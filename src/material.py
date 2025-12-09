import logging
from typing import Any
import warp as wp
from pxr import UsdShade
from warp_math_utils import (
    concentric_disk_sample,
    hemisphere_sample,
    hemisphere_pdf,
    hash2,
)

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
GGX_MIN_ALPHA = 1e-4

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
def same_hemisphere(wi: wp.vec3, wo: wp.vec3) -> wp.bool:
    return wi.z * wo.z > 0.0


@wp.func
def f0_dielectric(ior: float) -> float:
    ret = (ior - 1.0) / (ior + 1.0)
    return ret * ret


@wp.func
def fresnel_schlick(cos_theta_i: float, F_0: Any) -> Any:
    """
    Fresnel term using Schlick's approximation.
    """
    cos_theta_i = wp.clamp(cos_theta_i, -1.0, 1.0)
    if cos_theta_i < 0.0:
        return type(F_0)(0.0)

    factor = wp.pow(1.0 - cos_theta_i, 5.0)
    return F_0 + (type(F_0)(1.0) - F_0) * factor


@wp.func
def _ggx_alpha(roughness: float) -> float:
    return wp.max(roughness * roughness, GGX_MIN_ALPHA)


@wp.func
def ggx_d(h: wp.vec3, roughness: float) -> float:
    """
    GGX normal distribution function.
    h is the microfacet normal, assumed to be in normal space.
    """
    if h.z <= 0.0:
        return 0.0
    alpha = _ggx_alpha(roughness)
    alpha2 = alpha * alpha
    denom = h.z * h.z * (alpha2 - 1.0) + 1.0
    return alpha2 / (wp.pi * denom * denom)


@wp.func
def _ggx_smith_g1(cos_theta: float, roughness: float) -> float:
    if cos_theta <= 0.0:
        return 0.0
    alpha = _ggx_alpha(roughness)
    alpha2 = alpha * alpha
    sqrt_term = wp.sqrt(alpha2 + (1.0 - alpha2) * cos_theta * cos_theta)
    return 2.0 * cos_theta / (cos_theta + sqrt_term)


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
    denom = wp.abs(wo.z)
    if denom < EPSILON:
        return 0.0
    return (
        _ggx_smith_g1(wo.z, roughness)
        * ggx_d(h, roughness)
        * wp.abs(wp.dot(wo, h))
        / denom
    )


@wp.func
def ggx_visible_normal_sample(
    rand_state: wp.uint32, wo: wp.vec3, roughness: float
) -> wp.vec3:
    """
    Sample a microfacet normal for the GGX distribution using Heitz's method.
    wo is the outgoing direction (inverse viewing direction) in normal space.
    """
    alpha = _ggx_alpha(roughness)
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
    cos_theta_wi = wi.z
    cos_theta_wo = wo.z
    if not same_hemisphere(wi, wo):
        return wp.vec3(0.0, 0.0, 0.0)

    h = wp.normalize(wi + wo)
    cos_theta_h = wp.clamp(wp.dot(wo, h), 0.0, 1.0)

    # TODO: handle metallic parameter
    F0 = wp.lerp(wp.vec3(0.4), material.base_color, material.metallic)
    F = fresnel_schlick(cos_theta_h, F0)
    D = ggx_d(h, material.roughness)
    G = ggx_g(wi, wo, material.roughness)

    # fraction of light reflected (specular)
    kS = F
    # fraction of light refracted (diffuse), attenuated by the metallic parameter
    kD = (wp.vec3(1.0) - kS) * (1.0 - material.metallic)

    specular = F * D * G / (4.0 * cos_theta_wi * cos_theta_wo)
    diffuse = wp.cw_mul(kD, material.base_color) / wp.pi
    return specular + diffuse


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
    prob_specular = wp.lerp(0.5, 1.0, material.metallic)
    choice = wp.randf(rand_state)

    if choice < prob_specular:
        h = ggx_visible_normal_sample(rand_state, wo, material.roughness)
        # reflect wo around the sampled microfacet normal
        return 2.0 * wp.dot(wo, h) * h - wo
    else:
        return hemisphere_sample(rand_state)


@wp.func
def mat_pdf(
    material: Material,
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> float:
    """
    Evaluate the PDF of a sampled incoming direction.
    """
    if not same_hemisphere(wi, wo):
        return 0.0

    h = wp.normalize(wi + wo)

    prob_specular = wp.lerp(0.5, 1.0, material.metallic)
    prob_diffuse = 1.0 - prob_specular

    pdf_wh = ggx_visible_normal_pdf(wo, h, material.roughness)
    pdf_wi = pdf_wh / (4.0 * wp.max(wp.abs(wp.dot(wo, h)), EPSILON))

    pdf_diffuse = hemisphere_pdf(wi)

    return prob_specular * pdf_wi + prob_diffuse * pdf_diffuse


# ---------------------------------------
# BSDF viewer/ debug helper
#
# Verification criteria:
# 1. (implemented) PDF normalization: \int p(w) dw=1 on the claimed domain (sphere or hemisphere).
#
# 2. Sampling correctness:
#   - sample() produces samples whose empirical distribution matches pdf() (via histograms / CDF tests).
#
# 3. (implemented) BSDF consistency & energy check:
#   - Using sample + pdf + eval to estimate the reflectance integral via Monte Carlo integration and importance sampling.
#
# 4. Reciprocity check:
#   - f_r(w_i, w_o) \approx f_r(w_o, w_i)
#
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
    """
    Sample a BRDF according to the corresponding sampling method.
    """
    tid = wp.tid()
    state = wp.rand_init(tid)
    wi = mat_sample(material, state, wo)
    brdf = mat_eval_bsdf(material, wi, wo)
    out_dirs[tid] = wi * wp.length(brdf)


@wp.kernel
def pdf_integral_one_iter(
    material: Material,
    cur_iter: int,
    wo: wp.vec3,
    out_samples: wp.array(dtype=wp.float32),
):
    """
    Check the validity of the PDF of a material, i.e. its spherical integral should be 1.
    """
    tid = wp.tid()
    rand_seed = hash2(tid, cur_iter)
    state = wp.rand_init(rand_seed)
    wi = wp.sample_unit_sphere_surface(state)
    inv_sample_pdf = 4.0 * wp.pi

    out_samples[tid] = mat_pdf(material, wi, wo) * inv_sample_pdf


@wp.kernel
def reflectance_integral_one_iter(
    material: Material,
    cur_iter: int,
    wo: wp.vec3,
    out_samples: wp.array(dtype=wp.vec3),
):
    """
    Evaluate the reflectance integral of a material.
    This is for checking the consistency of the sampler + pdf + eval.
    """
    tid = wp.tid()
    rand_seed = hash2(tid, cur_iter)
    state = wp.rand_init(rand_seed)

    wi = mat_sample(material, state, wo)
    brdf = mat_eval_bsdf(material, wi, wo)
    pdf = mat_pdf(material, wi, wo)
    if pdf < EPSILON:
        out_samples[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        out_samples[tid] = brdf * wi.z / pdf


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
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
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of iterations for PDF Monte Carlo integration.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without showing any plots.",
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
    # Run PDF Monte Carlo Integration
    # ---------------------------
    pdf_samples = wp.empty(args.n_samples, dtype=wp.float32, device="cuda")
    running_integral = []
    mean_estimate = 0.0
    total_samples = 0

    print(
        f"Running PDF Monte Carlo integration: {args.n_samples} samples/iter, {args.n_iter} iters"
    )
    for iter_idx in range(args.n_iter):
        wp.launch(
            kernel=pdf_integral_one_iter,
            dim=args.n_samples,
            inputs=[m, iter_idx, wo, pdf_samples],
        )

        batch_mean = np.mean(pdf_samples.numpy())
        mean_estimate += (batch_mean - mean_estimate) * (
            args.n_samples / (total_samples + args.n_samples)
        )
        total_samples += args.n_samples
        running_integral.append(mean_estimate)

    final_pdf = running_integral[-1]
    final_pdf_error = abs(final_pdf - 1.0)
    if running_integral:
        print(
            f"PDF Hemisphere integral estimate: {final_pdf}, expected value: 1.0, error: {final_pdf_error:.6f}"
        )
    else:
        print(f"Please ensure --n-iter is greater than 0.")
        exit(1)

    # ---------------------------
    # Run Reflectance Integral Monte Carlo Integration
    # ---------------------------
    reflectance_samples = wp.empty(args.n_samples, dtype=wp.vec3, device="cuda")
    running_reflectance = []
    mean_reflectance = np.array([0.0, 0.0, 0.0])
    total_samples = 0

    print(
        f"Running Reflectance Integral Monte Carlo integration: {args.n_samples} samples/iter, {args.n_iter} iters"
    )
    for iter_idx in range(args.n_iter):
        wp.launch(
            kernel=reflectance_integral_one_iter,
            dim=args.n_samples,
            inputs=[m, iter_idx, wo, reflectance_samples],
        )

        batch_mean = np.mean(reflectance_samples.numpy(), axis=0)
        mean_reflectance += (batch_mean - mean_reflectance) * (
            args.n_samples / (total_samples + args.n_samples)
        )
        total_samples += args.n_samples
        running_reflectance.append(mean_reflectance.copy())

    final_reflectance = running_reflectance[-1]
    final_reflectance_error = np.abs(final_reflectance - 1.0)
    if running_reflectance:
        print(
            f"Reflectance integral estimate: R={final_reflectance[0]:.6f}, G={final_reflectance[1]:.6f}, B={final_reflectance[2]:.6f}, it should be <= 1 for an energy-conserving BSDF"
        )

    # early return if in headless mode
    if args.headless:
        exit(0)

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
    # Combined Visualization (3D + PDF Convergence)
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pdf = fig.add_subplot(gs[0, 1])
    ax_pdf.set_xlabel("Iteration", fontsize=12)
    ax_pdf.set_ylabel("Integral Estimate", fontsize=12)
    ax_pdf.set_title(
        r"Monte Carlo Estimate of $\int_{\Omega} \text{pdf}(\omega) \ \mathrm{d}\omega$",
        fontsize=12,
    )
    ax_reflectance = fig.add_subplot(gs[1, 1])
    ax_reflectance.set_title("Reflectance")
    ax_reflectance.set_xlabel("Iteration", fontsize=12)
    ax_reflectance.set_ylabel("Reflectance", fontsize=12)
    ax_reflectance.set_title(
        r"Monte Carlo Importance Sampling Estimate of $\int_{\Omega+} f_r(\omega_i, \omega_o) \cos(\theta_i) \ \mathrm{d}\omega_i$",
        fontsize=12,
    )

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

    ax_3d.add_collection3d(mesh)

    # Plot stochastic samples if provided
    if stochastic_samples_np is not None:
        # Plot sampled outgoing directions as vectors
        ax_3d.scatter(
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
        ax_3d.quiver(
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
    ax_3d.set_xlim(-max_range, max_range)
    ax_3d.set_ylim(-max_range, max_range)
    ax_3d.set_zlim(0, max_range)

    if stochastic_samples_np is not None:
        ax_3d.legend()

    title_3d = f"Lobe Visualization ({args.n_samples} samples, metallic={args.metallic}, roughness={args.roughness}, ior={args.ior})"
    ax_3d.set_title(title_3d)

    # ---------------------------
    # Plot PDF Convergence
    # ---------------------------
    ax_pdf.plot(
        np.arange(1, args.n_iter + 1),
        running_integral,
        "b-",
        linewidth=2,
        label="Monte Carlo Estimate",
    )
    ax_pdf.axhline(
        y=1.0, color="r", linestyle="--", linewidth=2, label="Expected Value (1.0)"
    )
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.legend(fontsize=10)
    ax_pdf.text(
        0.02,
        0.98,
        f"Final: {final_pdf:.6f}\nError: {final_pdf_error:.6f}\nSamples: {total_samples:,}",
        transform=ax_pdf.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ---------------------------
    # Plot Reflectance Convergence
    # ---------------------------
    running_reflectance_np = np.array(running_reflectance)
    ax_reflectance.plot(
        np.arange(1, args.n_iter + 1),
        running_reflectance_np[:, 0],
        "r-",
        linewidth=2,
        label="R",
    )
    ax_reflectance.plot(
        np.arange(1, args.n_iter + 1),
        running_reflectance_np[:, 1],
        "g-",
        linewidth=2,
        label="G",
    )
    ax_reflectance.plot(
        np.arange(1, args.n_iter + 1),
        running_reflectance_np[:, 2],
        "b-",
        linewidth=2,
        label="B",
    )
    ax_reflectance.grid(True, alpha=0.3)
    ax_reflectance.legend(fontsize=10)
    ax_reflectance.text(
        0.02,
        0.98,
        f"Final R: {final_reflectance[0]:.6f}\nFinal G: {final_reflectance[1]:.6f}\nFinal B: {final_reflectance[2]:.6f}\nMax Diff: {np.max(final_reflectance_error):.6f}\nSamples: {total_samples:,}",
        transform=ax_reflectance.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()
