import warp as wp
from pxr import UsdShade
from warp_math_utils import (
    hemisphere_sample,
    hemisphere_pdf,
)

__all__ = [
    "MAT_ID_DIFFUSE",
    "MAT_ID_SPECULAR",
    "Material",
    "create_material_from_usd_prim",
    "mat_eval_bsdf",
    "mat_sample",
    "mat_pdf",
]

MAT_ID_DIFFUSE = 0
MAT_ID_SPECULAR = 1
MAT_NAME_TO_ID = {
    "diffuse": MAT_ID_DIFFUSE,
    "specular": MAT_ID_SPECULAR,
}
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
def is_perfect_lambertian(material: Material) -> wp.bool:
    return material.metallic <= EPSILON and material.roughness >= 1.0 - EPSILON


@wp.func
def is_perfect_mirror(material: Material) -> wp.bool:
    return material.metallic >= 1.0 - EPSILON and material.roughness <= EPSILON


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

    return wp.vec3(0.0, 0.0, 0.0)


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

    return wp.vec3(0.0, 0.0, 0.0)


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

    return 0.0


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
        "-r", "--roughness", type=float, default=0.5, help="Roughness of the material."
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
        help="Use stochastic sampling to generate samples instead of uniform grid",
    )
    parser.add_argument(
        "-q",
        "--quiver",
        action="store_true",
        help="Draw a quiver plot of the lobe instead of a mesh visualization",
    )

    args = parser.parse_args()

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

    # Outgoing direction (looking straight up)
    wo = wp.vec3(args.direction[0], args.direction[1], args.direction[2])
    wo = wp.normalize(wo)

    # ---------------------------
    # Sample BRDF
    # ---------------------------
    out_dirs = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")
    # Compute grid dimensions: num_theta * num_phi should equal n_samples
    num_theta = int(np.sqrt(args.n_samples))
    num_phi = args.n_samples // num_theta

    if args.stochastic:
        wp.launch(
            kernel=sample_brdf_stochastic, dim=args.n_samples, inputs=[m, wo, out_dirs]
        )
    else:
        in_dirs = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")

        wp.launch(
            kernel=gen_hemisphere_dirs,
            dim=args.n_samples,
            inputs=[num_theta, num_phi, in_dirs],
        )
        wp.launch(
            kernel=sample_brdf, dim=args.n_samples, inputs=[m, wo, in_dirs, out_dirs]
        )

    samples = out_dirs.numpy()

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

    if args.quiver:
        origin = np.zeros((args.n_samples, 3))
        # Plot sampled outgoing directions as vectors
        ax.quiver(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            length=1.0,
            normalize=True,
            color="blue",
        )

        # Also plot the reference outgoing direction wo
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
        )
        wo_np = np.array([wo[0], wo[1], wo[2]])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
    else:
        # 3D Mesh Visualization
        # Compute triangles
        tris = build_triangles(num_theta, num_phi)

        # Collect vertices
        verts = samples[tris]  # shape = (F, 3, 3)

        mesh = Poly3DCollection(verts, alpha=0.8)
        mesh.set_facecolor((0.8, 0.3, 0.3, 1.0))  # optional
        vals = np.linalg.norm(samples, axis=1)
        mesh.set_array(vals)
        mesh.set_cmap("viridis")
        mesh.set_clim(vmin=vals.min(), vmax=vals.max())

        ax.add_collection3d(mesh)
        max_range = np.max(np.abs(samples))
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, max_range)

    ax.set_title(
        f"BRDF Lobe Visualization ({args.n_samples} samples, metallic={args.metallic}, roughness={args.roughness}, ior={args.ior})"
    )
    plt.show()
