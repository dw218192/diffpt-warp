import pathlib
from typing import Optional, TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import warp as wp

EPSILON = 1e-4

if TYPE_CHECKING:
    from main import Renderer

__all__ = [
    "tonemap",
    "save_image",
    "load_target_image",
    "hdr_to_png",
]


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
def tonemap(
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
            kernel=tonemap,
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
