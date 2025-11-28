"""
Quick and dirty PBR exercise -- using Nvidia's warp library and Pixar's OpenUSD to implement a simple path tracer.
Techniques:
    1. Progressive path tracing
    2. Multiple importance sampling/ Next Event Estimation using direct light sampling
    3. Russian Roulette for termination
    4. Monte Carlo integration

https://nvidia.github.io/warp/modules/functions.html
"""

from dataclasses import dataclass
import logging
import pathlib

import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdShade

import warp as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox

from usd_utils import extract_pos
from warp_math_utils import (
    get_coord_system,
    triangle_sample,
    power_heuristic,
    area_to_solid_angle,
    get_triangle_points,
)

from mesh import (
    Mesh,
    create_mesh_from_usd_prim,
)
from material import (
    Material,
    create_material_from_usd_prim,
    mat_eval_bsdf,
    mat_sample,
    mat_pdf,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EPSILON = 1e-4


@wp.struct
class PathSegment:
    radiance: wp.vec3  # estimated radiance
    throughput: wp.vec3  # current throughput of the path
    point: wp.vec3  # current hit point
    ray_dir: wp.vec3  # current ray direction
    pixel_idx: int  # associated pixel index
    depth: int  # current path depth
    prev_bsdf_pdf: float  # previous BSDF PDF, used by MIS for indirect paths
    debug_radiance: wp.vec3  # n-bounce radiance storage for debugging purposes


@wp.kernel
def init_path_segments(
    path_segments: wp.array(dtype=PathSegment),
    path_flags: wp.array(dtype=int),
    width: int,
    height: int,
    # camera
    cam_pos: wp.vec3,
    fov_x: float,
    fov_y: float,
):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    # offset to the center of the pixel and normalize to [-1,1]
    sx = 2.0 * (float(x) + 0.5) / float(width) - 1.0
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    u = sx * wp.tan(0.5 * fov_x)
    v = sy * wp.tan(0.5 * fov_y)

    ro = cam_pos
    rd = wp.normalize(wp.vec3(u, v, -1.0))

    path_segments[tid].radiance = wp.vec3(0.0, 0.0, 0.0)
    path_segments[tid].throughput = wp.vec3(1.0, 1.0, 1.0)
    path_segments[tid].point = ro
    path_segments[tid].ray_dir = rd
    path_segments[tid].pixel_idx = tid
    path_segments[tid].depth = 0
    path_segments[tid].prev_bsdf_pdf = 0.0
    path_flags[tid] = 0


@wp.func
def scene_intersect(
    ro: wp.vec3,
    rd: wp.vec3,
    meshes: wp.array(dtype=Mesh),
    min_t: float,
    max_t: float,
) -> tuple[int, wp.vec3, wp.vec3, int]:
    """
    Find the closest intersection between a ray and the scene geometries (including lights).
    Returns the index (not warp mesh id) of the hit mesh, the normal at the hit point, the hit point, and the hit face index.
    If no intersection is found, the hit mesh index is -1.
    """
    t = wp.float32(max_t)
    hit_mesh_idx = wp.int32(-1)
    hit_normal = wp.vec3(0.0, 0.0, 0.0)
    hit_point = wp.vec3(0.0, 0.0, 0.0)
    hit_face_idx = wp.int32(-1)

    for i in range(meshes.shape[0]):
        query = wp.mesh_query_ray(meshes[i].mesh_id, ro, rd, t)
        if query.result and query.t < t and query.t > min_t:
            t = query.t
            hit_mesh_idx = i
            hit_normal = query.normal
            hit_point = ro + t * rd
            hit_face_idx = query.face

    return hit_mesh_idx, hit_normal, hit_point, hit_face_idx


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
        return -1, -1, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0

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
    hit_mesh_idx, _, hit_point, hit_face_idx = scene_intersect(
        origin, rd, meshes, 0.0, max_t + EPSILON
    )
    return hit_mesh_idx == target_mesh_idx and hit_face_idx == target_face_idx


@wp.kernel
def draw(
    path_segments: wp.array(dtype=PathSegment),
    path_flags: wp.array(dtype=int),
    meshes: wp.array(dtype=Mesh),
    light_indices: wp.array(dtype=int),
    materials: wp.array(dtype=Material),
    min_t: float,
    max_t: float,
    max_depth: int,
    iteration: int,
    debug_radiance_depth: int,  # -1: no debug, 0: first bounce (direct), 1: second bounce (1-bounce lighting), etc.
):
    tid = wp.tid()
    if path_flags[tid] == 1:
        return

    path = path_segments[tid]
    ro = path.point
    rd = path.ray_dir
    rand_state = wp.rand_init(
        1469598103934665603
        ^ (tid * 1099511628211)
        ^ (path.depth * 1469598103934665603)
        ^ iteration
    )

    # BSDF sampling
    hit_mesh_idx, hit_normal, hit_point, hit_face_idx = scene_intersect(
        ro, rd, meshes, min_t, max_t
    )

    if hit_mesh_idx != -1:
        normal_to_world_space = get_coord_system(hit_normal)
        world_to_normal_space = wp.transpose(normal_to_world_space)

        # information about the hit point
        wo = world_to_normal_space * -rd
        mesh = meshes[hit_mesh_idx]
        mat = materials[mesh.material_id]

        # Terminate BSDF path (1 BSDF sample)
        if mesh.is_light:
            light_pdf_area = compute_light_pdf(
                hit_point, hit_face_idx, mesh, light_indices
            )
            light_pdf_solid_angle = (
                area_to_solid_angle(ro, hit_point, hit_normal) * light_pdf_area
            )
            if path.depth > 0:
                mis_weight = power_heuristic(path.prev_bsdf_pdf, light_pdf_solid_angle)
            else:
                # if depth is 0, this is a direct lighting path (P_bsdf = 0.0)
                mis_weight = 1.0

            contrib = wp.cw_mul(
                path.throughput, mesh.light_color * mesh.light_intensity
            )
            path.radiance += mis_weight * contrib

            if path.depth == debug_radiance_depth:
                path.debug_radiance += contrib

            path_segments[tid] = path
            path_flags[tid] = 1
            return

        # Light sampling (NEE) (1 Light sample)
        (
            light_mesh_idx,
            light_face_idx,
            light_sample_point,
            light_normal,
            light_pdf_area,
        ) = light_sample(rand_state, meshes, light_indices)
        if (
            light_mesh_idx != -1
            and light_pdf_area > 0
            and is_visible(
                light_mesh_idx, light_face_idx, hit_point, light_sample_point, meshes
            )
        ):
            wi = world_to_normal_space * wp.normalize(light_sample_point - hit_point)
            if wi.z > 0.0:
                bsdf_pdf = mat_pdf(mat, wi, wo)
                light_pdf_solid_angle = (
                    area_to_solid_angle(hit_point, light_sample_point, light_normal)
                    * light_pdf_area
                )
                mis_weight = power_heuristic(light_pdf_solid_angle, bsdf_pdf)
                light_mesh = meshes[light_mesh_idx]

                contrib = wp.cw_mul(
                    path.throughput,
                    mat_eval_bsdf(mat, wi, wo) * wi.z / light_pdf_solid_angle,
                )

                Le = light_mesh.light_intensity * light_mesh.light_color
                contrib = wp.cw_mul(contrib, Le)

                path.radiance += mis_weight * contrib

                if path.depth == debug_radiance_depth - 1:
                    path.debug_radiance += mis_weight * contrib

        # BSDF sampling
        wi = mat_sample(mat, rand_state, wo)
        pdf = mat_pdf(mat, wi, wo)
        if pdf <= 0.0 or wi.z < 0.0:
            path.radiance = wp.vec3(0.0, 0.0, 0.0)
            path.debug_radiance = wp.vec3(0.0, 0.0, 0.0)
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        path.point = hit_point + hit_normal * EPSILON
        path.ray_dir = normal_to_world_space * wi
        path.throughput = wp.cw_mul(
            path.throughput,
            mat_eval_bsdf(mat, wi, wo) * wi.z / pdf,
        )
        path.depth += 1

        # russain roulette
        # the brighter the path, the higher the chance of its survival
        throughput = path.throughput
        p_rr = wp.clamp(
            wp.max(throughput.x, wp.max(throughput.y, throughput.z)), 0.22, 1.0
        )

        if wp.randf(rand_state) > p_rr:
            path.radiance = wp.vec3(0.0, 0.0, 0.0)
            path.debug_radiance = wp.vec3(0.0, 0.0, 0.0)
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        path.throughput = throughput / p_rr
        # hard stop
        if path.depth >= max_depth:
            path.radiance = wp.vec3(0.0, 0.0, 0.0)
            path.debug_radiance = wp.vec3(0.0, 0.0, 0.0)
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        # cache the BSDF PDF
        path.prev_bsdf_pdf = pdf
        path_segments[tid] = path
    else:
        path.radiance = wp.vec3(0.0, 0.0, 0.0)
        path.debug_radiance = wp.vec3(0.0, 0.0, 0.0)
        path_segments[tid] = path
        path_flags[tid] = 1


# monte carlo integration
@wp.kernel
def accumulate(
    path_segments: wp.array(dtype=PathSegment),
    radiances: wp.array(dtype=wp.vec3),
    debug_radiances: wp.array(dtype=wp.vec3),
    iteration: int,
):
    tid = wp.tid()
    num_samples = iteration + 1
    pixel_idx = path_segments[tid].pixel_idx
    radiances[pixel_idx] += (
        path_segments[tid].radiance - radiances[pixel_idx]
    ) / float(num_samples)
    debug_radiances[pixel_idx] += (
        path_segments[tid].debug_radiance - debug_radiances[pixel_idx]
    ) / float(num_samples)


@wp.kernel
def tonemap(
    radiances: wp.array(dtype=wp.vec3),
    pixels: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    pixel = radiances[tid]
    pixel[0] /= 1.0 + pixel[0]
    pixel[1] /= 1.0 + pixel[1]
    pixel[2] /= 1.0 + pixel[2]

    # gamma correction
    p = 0.45454545454  # 1/2.2
    pixel[0] = wp.clamp(wp.pow(pixel[0], p), 0.0, 1.0)
    pixel[1] = wp.clamp(wp.pow(pixel[1], p), 0.0, 1.0)
    pixel[2] = wp.clamp(wp.pow(pixel[2], p), 0.0, 1.0)
    pixels[tid] = pixel


@dataclass
class CameraParams:
    pos: tuple[float] = (0, 1, 2)
    fov_x: float = np.deg2rad(60)
    fov_y: float = 2 * np.atan(9 / 16 * np.tan(np.deg2rad(60) / 2))
    clipping_range: tuple[float] = (0, 1.0e6)


class Renderer:
    """
    Simple USD-based path tracer.

    Args:
        width: The width of the output image in pixels.
        usd_path: The path to the USD file to render.
        max_iter: The maximum number of iterations to run.
        max_depth: The maximum depth of the path.

    Returns:
        The rendered image.

    Constraints (for the sake of simplicity):
        - The USD file must contain a camera prim.
        - The USD file must contain at least one material.
        - Non-triangular or degenerate faces are not supported.
        - Only implements USD material binding API, activeness
            - Material format is custom, not UsdShade
            - No support for UsdLux
        - Lights can only be
            - meshes with UsdLux.MeshLightAPI
            - UsdLux.RectLight
    """

    def __init__(
        self,
        width: int,
        usd_path: pathlib.Path,
        max_iter: int,
        max_depth: int,
    ):  # height will be computed from width and fov set in the USD stage's camera prim

        self.cam_params = CameraParams()

        asset_stage: Usd.Stage = Usd.Stage.Open(usd_path.as_posix())

        self.meshes = []
        self.wp_meshes = []
        self.light_indices = []
        self.materials = []
        mat_prim_path_to_mat_id = {}

        # collect all materials
        for prim in asset_stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material = create_material_from_usd_prim(prim)
                self.materials.append(material)
                mat_prim_path_to_mat_id[prim.GetPrimPath()] = len(self.materials) - 1

        # create rendering primitives
        for prim in asset_stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                usd_cam = UsdGeom.Camera(prim)
                cam_pos = extract_pos(prim)

                focal_length = usd_cam.GetFocalLengthAttr().Get()
                h_ap = usd_cam.GetHorizontalApertureAttr().Get()
                v_ap = usd_cam.GetVerticalApertureAttr().Get()
                clipping_range = usd_cam.GetClippingRangeAttr().Get()

                # field of view
                fov_x = 2.0 * np.arctan(0.5 * h_ap / focal_length)
                fov_y = 2.0 * np.arctan(0.5 * v_ap / focal_length)

                self.cam_params = CameraParams(
                    cam_pos, fov_x, fov_y, (clipping_range[0], clipping_range[1])
                )
            else:
                res = create_mesh_from_usd_prim(prim, mat_prim_path_to_mat_id)
                if res is None:
                    continue

                wp_mesh, mesh = res
                self.wp_meshes.append(wp_mesh)
                self.meshes.append(mesh)
                if mesh.is_light:
                    self.light_indices.append(len(self.meshes) - 1)

        self.width = width
        self.height = int(
            width
            * np.tan(self.cam_params.fov_y * 0.5)
            / np.tan(self.cam_params.fov_x * 0.5)
        )
        self.dev_pixels = wp.zeros(
            self.width * self.height, dtype=wp.vec3, device="cuda"
        )

        logger.info(f"Using Camera Params: {self.cam_params}")
        logger.info(
            f"Loaded {len(self.meshes)} meshes ({len(self.light_indices)} lights)"
        )

        # stores the running estimates of the radiances for each pixel
        self.radiances = wp.zeros(self.width * self.height, dtype=wp.vec3)
        # stores the running estimates of debug radiances (n-th term of the Neumann series representation of the rendering equation)
        self.debug_radiances = wp.zeros(self.width * self.height, dtype=wp.vec3)

        self.meshes = wp.array(self.meshes, dtype=Mesh)
        self.materials = wp.array(self.materials, dtype=Material)
        self.light_indices = wp.array(self.light_indices, dtype=int)
        self.max_iter = max_iter
        self.max_depth = max_depth

        self._debug_radiance_depth = -1
        self._last_debug_radiance_depth = -1

        self._num_iter = 0
        self._path_segments = wp.zeros(
            self.width * self.height, dtype=PathSegment, device="cuda"
        )
        self._path_flags = wp.zeros(self.width * self.height, dtype=int, device="cuda")

    @property
    def debug_radiance_depth(self):
        return self._debug_radiance_depth

    @debug_radiance_depth.setter
    def debug_radiance_depth(self, value: int):
        self._debug_radiance_depth = value

    def render(self):
        if self._num_iter >= self.max_iter:
            return

        with wp.ScopedTimer("render single iteration"):
            wp.launch(
                kernel=init_path_segments,
                dim=self.width * self.height,
                inputs=[
                    self._path_segments,
                    self._path_flags,
                    self.width,
                    self.height,
                    self.cam_params.pos,
                    self.cam_params.fov_x,
                    self.cam_params.fov_y,
                ],
            )
            while True:
                wp.launch(
                    kernel=draw,
                    dim=self.width * self.height,
                    inputs=[
                        self._path_segments,
                        self._path_flags,
                        self.meshes,
                        self.light_indices,
                        self.materials,
                        self.cam_params.clipping_range[0],
                        self.cam_params.clipping_range[1],
                        self.max_depth,
                        self._num_iter,
                        self._debug_radiance_depth,
                    ],
                )
                host_path_flags = self._path_flags.numpy()
                should_continue = False
                for i in range(self.width * self.height):
                    if host_path_flags[i] == 0:
                        should_continue = True
                        break
                if not should_continue:
                    break

            if self._last_debug_radiance_depth != self._debug_radiance_depth:
                self.debug_radiances.zero_()
                self._last_debug_radiance_depth = self._debug_radiance_depth

            wp.launch(
                kernel=accumulate,
                dim=self.width * self.height,
                inputs=[
                    self._path_segments,
                    self.radiances,
                    self.debug_radiances,
                    self._num_iter,
                ],
            )
            self._num_iter += 1

    def get_pixels(self) -> np.ndarray:
        wp.launch(
            kernel=tonemap,
            dim=self.width * self.height,
            inputs=[
                self.radiances,
                self.dev_pixels,
            ],
        )
        return self.dev_pixels.numpy().reshape((self.height, self.width, 3))

    def get_debug_pixels(self) -> np.ndarray:
        wp.launch(
            kernel=tonemap,
            dim=self.width * self.height,
            inputs=[
                self.debug_radiances,
                self.dev_pixels,
            ],
        )
        return self.dev_pixels.numpy().reshape((self.height, self.width, 3))

    def reset(self):
        self._num_iter = 0
        self._last_debug_radiance_depth = -1
        self._debug_radiance_depth = -1
        self.radiances.zero_()
        self.debug_radiances.zero_()


g_running = True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--usd-path",
        type=pathlib.Path,
        default=(
            pathlib.Path(__file__).parent.parent / "assets" / "cornell.usda"
        ).resolve(),
        help="Path to the USD file to render.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Output image width in pixels."
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum number of iterations."
    )
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum path depth.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        renderer = Renderer(
            width=args.width,
            usd_path=args.usd_path,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
        )

        plt.ion()  # turn on interactive mode
        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(
            1, 2, width_ratios=[2, 1], height_ratios=[1], figure=fig, wspace=0.05
        )

        ax = fig.add_subplot(gs[:, 0])  # big render view
        ax_direct_lights = fig.add_subplot(gs[0, 1])

        ax.set_title("PBR Renderer")
        ax.set_axis_off()

        ax_direct_lights.set_title("N-bounce Radiance")
        ax_direct_lights.set_axis_off()

        # --- Add TextBox widget at bottom of the figure ---
        axbox = fig.add_axes([0.5, 0.02, 0.2, 0.05])  # [left, bottom, width, height]
        textbox = TextBox(axbox, "Debug Radiance Depth: ", initial="-1")

        def on_submit(text):
            try:
                val = int(text)
                val = max(-1, min(val, renderer.max_depth))
                ax_direct_lights.set_title(f"{val}-bounce Radiance")

                logger.info(f"Setting debug radiance depth to {val}")
                renderer.debug_radiance_depth = val
            except ValueError:
                textbox.set_val(renderer.debug_radiance_depth)
                logger.error(f"'{text}' is not a valid integer!")

        textbox.on_submit(on_submit)

        im = ax.imshow(
            np.zeros((renderer.height, renderer.width, 3)),
            origin="lower",
            interpolation="antialiased",
            aspect="equal",
        )
        im_direct_lights = ax_direct_lights.imshow(
            np.zeros((renderer.height, renderer.width, 3)),
            origin="lower",
            interpolation="antialiased",
            aspect="equal",
        )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # add label in top-left corner
        label = ax.text(
            0.01,
            0.99,
            "Iteration: 0",
            color="white",
            fontsize=9,
            ha="left",
            va="top",
            transform=ax.transAxes,  # coordinates relative to axes (0–1)
        )
        plt.show(block=False)

        def handle_close(_):
            global g_running
            g_running = False

        fig.canvas.mpl_connect("close_event", handle_close)

        # main loop
        while g_running:
            renderer.render()
            im.set_data(renderer.get_pixels())
            im_direct_lights.set_data(renderer.get_debug_pixels())
            fig.canvas.draw()
            fig.canvas.flush_events()
            label.set_text(f"Iteration: {renderer._num_iter}")

        plt.close()
