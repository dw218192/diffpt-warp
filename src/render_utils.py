import pathlib
from typing import Optional, TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import warp as wp
from mesh import Mesh
from warp_math_utils import get_triangle_points, triangle_sample

EPSILON = 1e-4

if TYPE_CHECKING:
    from main import Renderer

__all__ = [
    "tonemap",
    "save_image",
    "load_target_image",
    "hdr_to_png",
    "HitData",
    "scene_intersect",
    "light_sample",
    "compute_light_pdf",
    "is_visible",
]


@wp.struct
class HitData:
    mesh_idx: int  # -1 if miss
    face_idx: int
    hit_point: wp.vec3
    hit_normal: wp.vec3


@wp.func
def scene_intersect(
    bvh_id: wp.uint64,
    ro: wp.vec3,
    rd: wp.vec3,
    meshes: wp.array(dtype=Mesh),
    max_t: float,
    back_face: bool,
) -> HitData:
    """
    Find the closest intersection between a ray and the scene geometries (including lights).
    Returns the hit data.
    """
    t = wp.float32(max_t)
    hit_mesh_idx = wp.int32(-1)
    hit_normal = wp.vec3(0.0)
    hit_point = wp.vec3(0.0)
    hit_face_idx = wp.int32(-1)

    if bvh_id == wp.uint64(-1):
        for i in range(meshes.shape[0]):
            mesh_query = wp.mesh_query_ray(meshes[i].mesh_id, ro, rd, t)
            if mesh_query.result and mesh_query.t < t and mesh_query.t > 0.0:
                if not back_face and mesh_query.sign < 0.0:
                    continue

                t = mesh_query.t
                hit_mesh_idx = i
                hit_normal = mesh_query.normal
                hit_point = ro + t * rd
                hit_face_idx = mesh_query.face
    else:
        bvh_query = wp.bvh_query_ray(bvh_id, ro, rd)
        i = wp.int32(0)

        while wp.bvh_query_next(bvh_query, i):
            # The ray intersects the volume with index i
            mesh = meshes[i]
            mesh_query = wp.mesh_query_ray(mesh.mesh_id, ro, rd, t)
            if mesh_query.result and mesh_query.t < t and mesh_query.t > 0.0:
                if not back_face and mesh_query.sign < 0.0:
                    continue

                t = mesh_query.t
                hit_mesh_idx = i
                hit_normal = mesh_query.normal
                hit_point = ro + t * rd
                hit_face_idx = mesh_query.face

    return HitData(hit_mesh_idx, hit_face_idx, hit_point, hit_normal)


@wp.func
def light_sample(
    rand_state: wp.uint32,
    meshes: wp.array(dtype=Mesh),
    light_indices: wp.array(dtype=int),
) -> tuple[int, int, wp.vec3, wp.vec3, float]:
    """
    Sample a random point on the light sources in the scene.
    Returns the mesh index (not warp mesh id) of the light source, the point on the light source, the normal, and the probability.
    If there is no light in the scene, the mesh index is -1.
    """
    num_lights = len(light_indices)
    if num_lights == 0:
        return -1, -1, wp.vec3(0.0), wp.vec3(0.0), 0.0

    light_idx = wp.randi(rand_state, 0, num_lights)
    light_mesh = meshes[light_indices[light_idx]]

    mesh_id = light_mesh.mesh_id
    num_faces = light_mesh.num_faces
    face_idx = wp.randi(rand_state, 0, num_faces)

    normal = wp.mesh_eval_face_normal(mesh_id, face_idx)
    v1, v2, v3 = get_triangle_points(mesh_id, face_idx)
    pos, _, pdf = triangle_sample(rand_state, v1, v2, v3)

    # this is not uniform area sampling wrt. all light surfaces
    # but for simplicity we'll do it this way for now
    pdf = pdf / float(num_lights * num_faces)
    return light_indices[light_idx], face_idx, pos, normal, pdf


@wp.func
def compute_light_pdf(
    p: wp.vec3,
    face_idx: int,
    light_mesh: Mesh,
    light_indices: wp.array(dtype=int),
) -> float:
    """
    Given a point on a light source, compute the PDF of sampling that point.
    """
    num_lights = len(light_indices)
    if num_lights == 0:
        return 0.0
    num_faces = light_mesh.num_faces
    if num_faces == 0:
        return 0.0

    v1, v2, v3 = get_triangle_points(light_mesh.mesh_id, face_idx)
    area = 0.5 * wp.length(wp.cross(v2 - v1, v3 - v1))

    return 1.0 / (area * float(num_lights * num_faces))


@wp.func
def is_visible(
    bvh_id: wp.uint64,
    target_mesh_idx: int,
    target_face_idx: int,
    p1: wp.vec3,
    p2: wp.vec3,
    meshes: wp.array(dtype=Mesh),
):
    """
    Checks if a ray from p1 to p2 is occluded
    """
    p1p2 = p2 - p1
    max_t = wp.length(p1p2)
    rd = wp.normalize(p1p2)
    origin = p1 + rd * EPSILON
    hit_data = scene_intersect(bvh_id, origin, rd, meshes, max_t + EPSILON, True)
    hit_mesh_idx = hit_data.mesh_idx
    hit_face_idx = hit_data.face_idx
    return hit_mesh_idx == target_mesh_idx and hit_face_idx == target_face_idx


def _ensure_freeimage_available() -> None:
    """
    Ensure ImageIO's FreeImage plugin is available.

    We rely on it for Radiance .hdr read/write in many environments.
    The download is idempotent.
    """
    try:
        from imageio.plugins import freeimage

        freeimage.download()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"FreeImage is unavailable: {exc}") from exc


@wp.kernel
def tonemap_kernel(
    # Read
    radiances: wp.array(dtype=wp.vec3),
    # Write
    pixels: wp.array(dtype=wp.vec3),
):
    """
    Simple Reinhard-like tonemap + gamma, matching the original `main.py` behavior.
    Output is linear RGB in [0,1] after gamma correction.
    """
    tid = wp.tid()
    pixel = radiances[tid]
    pixel[0] /= 1.0 + pixel[0]
    pixel[1] /= 1.0 + pixel[1]
    pixel[2] /= 1.0 + pixel[2]

    # gamma correction
    p = 0.45454545454  # 1/2.2
    pixel[0] = wp.clamp(wp.pow(wp.max(pixel[0], EPSILON), p), 0.0, 1.0)
    pixel[1] = wp.clamp(wp.pow(wp.max(pixel[1], EPSILON), p), 0.0, 1.0)
    pixel[2] = wp.clamp(wp.pow(wp.max(pixel[2], EPSILON), p), 0.0, 1.0)
    pixels[tid] = pixel


def hdr_to_png(
    hdr_path: pathlib.Path,
    png_path: Optional[pathlib.Path] = None,
    *,
    flip_y: bool = False,
    device: Optional[str] = None,
) -> pathlib.Path:
    """
    Convert an HDR image (Radiance .hdr or other float formats supported by ImageIO)
    into an 8-bit PNG using Warp for tonemap/gamma and NumPy for uint8 packing.
    """
    hdr_path = pathlib.Path(hdr_path).resolve()
    if png_path is None:
        png_path = hdr_path.with_suffix(".png")
    png_path = pathlib.Path(png_path).resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure plugin availability for .hdr in common setups.
    if hdr_path.suffix.lower() == ".hdr":
        _ensure_freeimage_available()

    img = imageio.imread(hdr_path)
    if img.ndim == 2:
        raise ValueError("HDR image must be RGB, got grayscale.")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] != 3:
        raise ValueError("HDR image must have 3 channels (RGB).")

    img = img.astype(np.float32, copy=False)
    height, width = int(img.shape[0]), int(img.shape[1])

    with wp.ScopedDevice(device):
        radiances_wp = wp.array(img.reshape((-1, 3)), dtype=wp.vec3)
        pixels_wp = wp.zeros(width * height, dtype=wp.vec3)
        wp.launch(
            kernel=tonemap_kernel,
            dim=width * height,
            inputs=[radiances_wp],
            outputs=[pixels_wp],
        )
        pixels01 = pixels_wp.numpy().reshape((height, width, 3))

    if flip_y:
        pixels01 = pixels01[::-1, ...]

    out_u8 = (pixels01 * 255.0).clip(0.0, 255.0).astype(np.uint8)
    imageio.imwrite(png_path.as_posix(), out_u8)
    return png_path


def save_image(
    renderer: "Renderer", path_arg: pathlib.Path | None, save_raw: bool = False
):
    """
    Saves the current renderer output to a directory.

    - If save_raw=True: writes radiance as .hdr (float32) via ImageIO.
    - Else: writes tonemapped 8-bit .png (tonemap + packing done by Warp).
    """
    extension = "hdr" if save_raw else "png"
    stage_stem = (
        renderer.usd_path.stem
        if isinstance(getattr(renderer, "usd_path", None), pathlib.Path)
        else "render"
    )

    output_dir = (path_arg or pathlib.Path(__file__).parent.parent).resolve()
    output_path = (output_dir / f"{stage_stem}.{extension}").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_raw:
        # HDR writing is file I/O bound; radiance transfer is unavoidable here.
        _ensure_freeimage_available()
        radiances = renderer.get_pixels(use_tonemap=False)
        radiances = radiances[::-1, ...].astype(np.float32, copy=False)
        imageio.imwrite(output_path.as_posix(), radiances, format="HDR-FI")
        return output_path

    # PNG path: keep tonemap on GPU (Renderer.get_pixels), but pack to u8 in NumPy.
    pixels01 = renderer.get_pixels(use_tonemap=True)[::-1, ...]
    out_u8 = (pixels01 * 255.0).clip(0.0, 255.0).astype(np.uint8)
    imageio.imwrite(output_path.as_posix(), out_u8)
    return output_path


def load_target_image(target_path: pathlib.Path, renderer: "Renderer") -> np.ndarray:
    """
    Load and validate a target RGB image for learning/visualization.
    Ensures shape/channel match and aligns orientation with the renderer output.
    Only HDR inputs are accepted to avoid unintended 8-bit gamma/tonemap issues.
    """
    target_path = pathlib.Path(target_path).resolve()
    suffix = target_path.suffix.lower()
    allowed_hdr_exts = {".hdr", ".exr"}
    if suffix not in allowed_hdr_exts:
        raise ValueError("Target image must be HDR.")

    if suffix == ".hdr":
        _ensure_freeimage_available()

    img = imageio.imread(target_path)
    if img.ndim == 2:
        raise ValueError("Target image must be RGB, got grayscale.")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] != 3:
        raise ValueError("Target image must have 3 channels (RGB).")

    img = img.astype(np.float32, copy=False)

    if img.shape[0] != renderer.height or img.shape[1] != renderer.width:
        raise ValueError(
            f"Target image shape {img.shape[:2]} does not match renderer output "
            f"({renderer.height}, {renderer.width})."
        )

    # Invert and flatten to match renderer's in-memory layout.
    return img[::-1, ...].reshape(-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    hdr_to_png_parser = subparsers.add_parser(
        "hdr_to_png", help="Convert an HDR image (e.g. .hdr) to an 8-bit PNG."
    )
    hdr_to_png_parser.add_argument(
        "src_path", type=pathlib.Path, help="Path to the HDR image to convert."
    )
    hdr_to_png_parser.add_argument(
        "-o",
        "--output-path",
        type=pathlib.Path,
        default=None,
        help="Output PNG path. Defaults to <src_path>.png",
    )
    hdr_to_png_parser.add_argument(
        "-f",
        "--flip-y",
        action="store_true",
        default=False,
        help="Flip the image vertically.",
    )
    hdr_to_png_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override (e.g. 'cpu', 'cuda:0').",
    )

    args = parser.parse_args()

    if args.subcommand == "hdr_to_png":
        out = hdr_to_png(
            args.src_path,
            args.output_path,
            flip_y=bool(args.flip_y),
            device=args.device,
        )
        print(out.as_posix())
