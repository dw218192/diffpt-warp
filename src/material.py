import warp as wp
from pxr import UsdShade
from warp_math_utils import (
    concentric_disk_sample,
    hemisphere_sample,
    hemisphere_pdf,
)
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "Material",
    "create_material_from_usd_prim",
    "is_emissive",
    "mat_eval_bsdf",
    "mat_sample",
    "mat_pdf",
]

EPSILON = 1e-6

# here we'll partially implement the metallic-roughness part of the preview surface schema
# USD preview surface schema: https://openusd.org/dev/spec_usdpreviewsurface.html#preview-surface


@wp.struct
class Material:
    base_color: wp.vec3  # previewSurface:diffuseColor
    metallic: float  # previewSurface:metallic
    roughness: float  # previewSurface:roughness
    ior: float  # previewSurface:ior (default 1.5)
    emissive_color: wp.vec3  # previewSurface:emissiveColor
    emissive_intensity: float  # emissiveIntensity (custom attribute)


def create_material_from_usd_prim(prim: UsdShade.Material):
    """
    Create a material from a USD material prim.
    """
    mat = Material()
    # only one node is allowed
    surface_output = prim.GetSurfaceOutput()
    shader, _, _ = surface_output.GetConnectedSource()
    shader = UsdShade.Shader(shader)

    def get(attr, default):
        a = shader.GetInput(attr)
        val = a.Get()
        return val if val is not None else default

    mat.base_color = get("diffuseColor", wp.vec3(0.8, 0.8, 0.8))
    mat.metallic = get("metallic", 0.0)
    mat.roughness = get("roughness", 0.5)
    mat.ior = get("ior", 1.5)
    mat.emissive_color = get("emissiveColor", wp.vec3(0.0, 0.0, 0.0))

    # custom attributes
    mat.emissive_intensity = get("emissiveIntensity", 0.0)

    logger.debug(f"Material: {mat}")
    return mat


@wp.func
def fresnel_dielectric(cos_theta_i: float, ior: float) -> float:
    """
    Schlick's approximation for the Fresnel term, assuming air-dielectric interface.
    """
    cos_theta_i = wp.clamp(cos_theta_i, -1.0, 1.0)
    if cos_theta_i < 0.0:
        ior = 1.0 / ior
        cos_theta_i = -cos_theta_i
    F_0 = (ior - 1.0) / (ior + 1.0)
    F_0 = F_0 * F_0
    return F_0 + (1.0 - F_0) * wp.pow(1.0 - cos_theta_i, 5.0)


@wp.func
def fresnel_metal(cos_theta_i: float, F_0: wp.vec3) -> wp.vec3:
    """
    Fresnel term for a metallic material.
    """
    cos_theta_i = wp.clamp(cos_theta_i, -1.0, 1.0)
    if cos_theta_i < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    factor = wp.pow(1.0 - cos_theta_i, 5.0)
    result = wp.vec3(0.0, 0.0, 0.0)
    result.x = F_0.x + (1.0 - F_0.x) * factor
    result.y = F_0.y + (1.0 - F_0.y) * factor
    result.z = F_0.z + (1.0 - F_0.z) * factor
    return result


@wp.func
def ggx_d(h: wp.vec3, roughness: float) -> float:
    """
    GGX normal distribution function.
    h is the microfacet normal, assumed to be in normal space.
    """
    if h.z <= 0.0:
        return 0.0
    alpha = roughness * roughness
    return alpha / (wp.pi * wp.pow(h.z * h.z * (alpha - 1.0) + 1.0, 2.0))


@wp.func
def _ggx_smith_g1(cos_theta: float, roughness: float) -> float:
    if cos_theta <= 0.0:
        return 0.0
    alpha = roughness * roughness
    return (
        2.0
        * cos_theta
        / (cos_theta + wp.sqrt(alpha + (1.0 - alpha) * cos_theta * cos_theta))
    )


@wp.func
def ggx_g(wi: wp.vec3, wo: wp.vec3, roughness: float) -> float:
    """
    Smith's shadowing-masking function for GGX.
    wi is the incoming direction, assumed to be in normal space.
    wo is the outgoing direction, assumed to be in normal space.
    """
    return _ggx_smith_g1(wi.z, roughness) * _ggx_smith_g1(wo.z, roughness)


@wp.func
def ggx_visible_normal_pdf(wo: wp.vec3, h: wp.vec3, roughness: float) -> float:
    """
    Probability density function for the visible microfacet normals.
    wo is the outgoing direction, assumed to be in normal space.
    h is the microfacet normal, assumed to be in normal space.
    """
    return (
        _ggx_smith_g1(wo.z, roughness)
        / wp.abs(wo.z)
        * ggx_d(h, roughness)
        * wp.abs(wp.dot(wo, h))
    )


@wp.func
def ggx_visible_normal_sample(
    rand_state: wp.uint32, wo: wp.vec3, roughness: float
) -> wp.vec3:
    """
    Sample a microfacet normal for the GGX distribution using Heitz's method.
    wo is the outgoing direction (inverse viewing direction) in normal space.
    """
    alpha = roughness * roughness
    wh = wp.normalize(wp.vec3(alpha * wo.x, alpha * wo.y, wo.z))
    if wh.z < 0.0:
        wh = -wh

    # construct a coordinate system with wh as the z-axis
    T1 = (
        wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), wh))
        if wh.z < 0.99999
        else wp.vec3(1.0, 0.0, 0.0)
    )
    T2 = wp.cross(wh, T1)
    u = concentric_disk_sample(rand_state)

    # warp u to the hemisphere projection (half-disk foreshortened by the cosine of wh)
    h = wp.sqrt(1.0 - u.x * u.x)
    u.y = wp.lerp(h, u.y, (1.0 + wh.z) / 2.0)

    # project back to hemisphere and unstretch
    nh = u.x * T1 + u.y * T2 + wp.sqrt(wp.max(0.0, 1.0 - u.x * u.x - u.y * u.y)) * wh
    nh = wp.vec3(nh.x * alpha, nh.y * alpha, wp.max(1e-6, nh.z))
    nh = wp.normalize(nh)
    return nh


@wp.func
def is_perfect_lambertian(material: Material) -> wp.bool:
    return material.metallic <= EPSILON and material.roughness >= 1.0 - EPSILON


@wp.func
def is_perfect_mirror(material: Material) -> wp.bool:
    return material.metallic >= 1.0 - EPSILON and material.roughness <= EPSILON


@wp.func
def is_emissive(material: Material) -> wp.bool:
    return (
        material.emissive_color.x > EPSILON
        or material.emissive_color.y > EPSILON
        or material.emissive_color.z > EPSILON
    )


@wp.func
def mat_eval_bsdf(
    material: Material,
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> wp.vec3:
    """
    Evaluate the BRDF for a given material and an incoming and outgoing direction.
    Returns the BRDF value.
    """

    if is_perfect_lambertian(material):
        return material.base_color / wp.pi
    elif is_perfect_mirror(material):
        if (
            wi.z > 0.0
            and wp.abs(wi.x + wo.x) < EPSILON
            and wp.abs(wi.y + wo.y) < EPSILON
            and wp.abs(wi.z - wo.z) < EPSILON
        ):
            return fresnel_dielectric(wi.z, material.ior) * material.base_color / wi.z
        else:
            return wp.vec3(0.0, 0.0, 0.0)

    cos_theta_wi = wi.z
    cos_theta_wo = wo.z
    if cos_theta_wi * cos_theta_wo <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    h = wp.normalize(wi + wo)
    cos_theta_h = wp.clamp(wp.dot(wo, h), 0.0, 1.0)

    # TODO: handle metallic parameter

    # metal_f = fresnel_metal(cos_theta_h, material.base_color) * material.metallic
    # dielectric_f = fresnel_dielectric(cos_theta_h, material.ior) * (
    #     1.0 - material.metallic
    # )
    # F = wp.vec3(
    #     metal_f.x + dielectric_f,
    #     metal_f.y + dielectric_f,
    #     metal_f.z + dielectric_f,
    # )
    F = fresnel_metal(cos_theta_h, material.base_color)
    D = ggx_d(h, material.roughness)
    G = ggx_g(wi, wo, material.roughness)

    specular = F * D * G / (4.0 * cos_theta_wi * cos_theta_wo)
    # diffuse = material.base_color * (1.0 - material.metallic) / wp.pi

    # return specular + diffuse
    return specular

@wp.func
def mat_sample(
    material: Material,
    rand_state: wp.uint32,
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> wp.vec3:
    """
    Sample a BRDF for a given material and an outgoing direction.
    Returns the sampled incoming direction.

    The definition of incoming and outgoing directions is consistent with PBRT.
    """

    if is_perfect_lambertian(material):
        return hemisphere_sample(rand_state)
    elif is_perfect_mirror(material):
        return wp.vec3(-wo.x, -wo.y, wo.z)

    h = ggx_visible_normal_sample(rand_state, wo, material.roughness)
    wi = 2.0 * wp.dot(wo, h) * h - wo
    return wi


@wp.func
def mat_pdf(
    material: Material,
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> float:
    """
    Evaluate the PDF of a sampled incoming direction.
    """

    if is_perfect_lambertian(material):
        return hemisphere_pdf(wi)
    elif is_perfect_mirror(material):
        if (
            wp.abs(wi.x + wo.x) < EPSILON
            and wp.abs(wi.y + wo.y) < EPSILON
            and wp.abs(wi.z - wo.z) < EPSILON
        ):
            return 1.0
        return 0.0

    h = wp.normalize(wi + wo)
    pdf_wh = ggx_visible_normal_pdf(wo, h, material.roughness)
    pdf_wi = pdf_wh / (4.0 * wp.abs(wp.dot(wo, h)))

    return pdf_wi


# ---------------------------------------
# BRDF viewer
# ---------------------------------------
@wp.kernel
def gen_hemisphere_dirs(
    num_theta: int,
    num_phi: int,
    out_dirs: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # map flat index -> 2D grid (i, j)
    i = tid // num_phi  # theta index
    j = tid % num_phi  # phi index

    # normalized [0,1] coordinates
    u = (float(i) + 0.5) / float(num_theta)
    v = (float(j) + 0.5) / float(num_phi)

    # spherical angles
    theta = u * (wp.pi * 0.5)  # upper hemisphere
    phi = v * (wp.pi * 2.0)

    # convert to direction
    sin_theta = wp.sin(theta)

    x = sin_theta * wp.cos(phi)
    y = sin_theta * wp.sin(phi)
    z = wp.cos(theta)

    out_dirs[tid] = wp.vec3(x, y, z)


@wp.kernel
def sample_brdf(
    material: Material,
    wo: wp.vec3,  # in normal space, towards 'sensor'
    in_dirs: wp.array(dtype=wp.vec3),
    out_dirs: wp.array(dtype=wp.vec3),
):
    """
    Sample a BRDF for a given material and a set of hemisphere directions.
    It's consistent with the formulation of a typical path tracer, where an outgoing direction is given,
    and the incoming directions are sampled from the BRDF.
    """
    tid = wp.tid()
    wi = in_dirs[tid]
    brdf = mat_eval_bsdf(material, wi, wo)
    out_dirs[tid] = wi * wp.length(brdf)


@wp.kernel
def sample_brdf_stochastic(
    material: Material,
    wo: wp.vec3,  # in normal space, towards 'sensor'
    out_dirs: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    state = wp.rand_init(tid)
    wi = mat_sample(material, state, wo)
    brdf = mat_eval_bsdf(material, wi, wo)
    out_dirs[tid] = wi * wp.length(brdf)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    parser = argparse.ArgumentParser()
    material_arg_parser = parser.add_argument_group("Material")
    material_arg_parser.add_argument(
        "-c",
        "--base-color",
        type=float,
        nargs=3,
        default=[0.8, 0.8, 0.8],
        help="Base color of the material.",
    )
    material_arg_parser.add_argument(
        "-m", "--metallic", type=float, default=0.0, help="Metallic of the material."
    )
    material_arg_parser.add_argument(
        "-r",
        "--roughness",
        type=float,
        default=0.5,
        help="Roughness of the material.",
    )
    material_arg_parser.add_argument(
        "-i", "--ior", type=float, default=1.5, help="IOR of the material."
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=1000, help="Number of samples."
    )
    parser.add_argument(
        "-dir",
        "--direction",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Direction of the outgoing direction.",
    )
    parser.add_argument(
        "-s",
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling to generate samples for comparison",
    )

    args = parser.parse_args()

    # validate
    if args.metallic < 0.0 or args.metallic > 1.0:
        raise ValueError("Metallic must be between 0.0 and 1.0")
    if args.roughness < 0.0 or args.roughness > 1.0:
        raise ValueError("Roughness must be between 0.0 and 1.0")
    if args.ior < 1.0 or args.ior > 2.0:
        raise ValueError("IOR must be between 1.0 and 2.0")
    if args.base_color[0] < 0.0 or args.base_color[0] > 1.0:
        raise ValueError("Base color must be between 0.0 and 1.0")
    if args.base_color[1] < 0.0 or args.base_color[1] > 1.0:
        raise ValueError("Base color must be between 0.0 and 1.0")
    if args.base_color[2] < 0.0 or args.base_color[2] > 1.0:
        raise ValueError("Base color must be between 0.0 and 1.0")

    wp.init()

    # ---------------------------
    # Material setup
    # ---------------------------
    m = Material()
    m.metallic = args.metallic
    m.roughness = args.roughness
    m.ior = args.ior
    m.base_color = wp.vec3(args.base_color[0], args.base_color[1], args.base_color[2])
    m.emissive_color = wp.vec3(0.0, 0.0, 0.0)
    m.emissive_intensity = 0.0

    # Outgoing direction (looking straight up)
    wo = wp.vec3(args.direction[0], args.direction[1], args.direction[2])
    wo = wp.normalize(wo)

    # ---------------------------
    # Sample BRDF
    # ---------------------------
    # Compute grid dimensions: num_theta * num_phi should equal n_samples
    num_theta = int(np.sqrt(args.n_samples))
    num_phi = args.n_samples // num_theta

    # Always generate uniform grid samples for mesh plot
    in_dirs = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")
    mesh_samples = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")

    wp.launch(
        kernel=gen_hemisphere_dirs,
        dim=args.n_samples,
        inputs=[num_theta, num_phi, in_dirs],
    )
    wp.launch(
        kernel=sample_brdf, dim=args.n_samples, inputs=[m, wo, in_dirs, mesh_samples]
    )

    # Generate stochastic samples if requested
    stochastic_samples = None
    if args.stochastic:
        stochastic_samples = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")
        wp.launch(
            kernel=sample_brdf_stochastic,
            dim=args.n_samples,
            inputs=[m, wo, stochastic_samples],
        )

    mesh_samples_np = mesh_samples.numpy()
    stochastic_samples_np = (
        stochastic_samples.numpy() if stochastic_samples is not None else None
    )

    # ---------------------------
    # 3D Visualization
    # ---------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def build_triangles(num_theta: int, num_phi: int) -> np.ndarray:
        tris = []

        for i in range(num_theta - 1):
            for j in range(num_phi):
                i0 = i * num_phi + j
                i1 = i * num_phi + (j + 1) % num_phi
                i2 = (i + 1) * num_phi + j
                i3 = (i + 1) * num_phi + (j + 1) % num_phi

                # two triangles
                tris.append([i0, i1, i2])
                tris.append([i2, i1, i3])

        return np.array(tris)

    # Always plot mesh visualization
    tris = build_triangles(num_theta, num_phi)
    verts = mesh_samples_np[tris]  # shape = (F, 3, 3)

    mesh = Poly3DCollection(verts, alpha=0.6)
    vals = np.linalg.norm(mesh_samples_np, axis=1)
    mesh.set_array(vals)
    mesh.set_cmap("viridis")
    mesh.set_clim(vmin=vals.min(), vmax=vals.max())

    ax.add_collection3d(mesh)

    # Plot stochastic samples if provided
    if stochastic_samples_np is not None:
        # Plot sampled outgoing directions as vectors
        ax.scatter(
            stochastic_samples_np[:, 0],
            stochastic_samples_np[:, 1],
            stochastic_samples_np[:, 2],
            color="red",
            s=2,
            alpha=0.7,
            label="Stochastic samples",
        )

        # Also plot the reference outgoing direction wo
        origin = np.zeros((1, 3))
        ax.quiver(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            wo[0],
            wo[1],
            wo[2],
            length=1.0,
            normalize=True,
            color="green",
            label="Outgoing direction",
        )

        # Count number of degenerate samples
        degenerate_samples = np.sum(
            np.linalg.norm(stochastic_samples_np, axis=1) < EPSILON
        )
        logger.warning(
            f"Number of degenerate samples: {degenerate_samples}/{args.n_samples}"
        )

    # Set axis limits based on both datasets
    all_samples = (
        np.vstack([mesh_samples_np, stochastic_samples_np])
        if stochastic_samples_np is not None
        else mesh_samples_np
    )
    max_range = np.max(np.abs(all_samples))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range)

    if stochastic_samples_np is not None:
        ax.legend()

    title = f"BRDF Lobe Visualization ({args.n_samples} samples, metallic={args.metallic}, roughness={args.roughness}, ior={args.ior})"
    if stochastic_samples_np is not None:
        title += " [Mesh + Stochastic]"
    ax.set_title(title)
    plt.show()
