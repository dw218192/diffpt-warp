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

from usd_utils import (
    extract_pos,
    get_world_transform,
    usdmesh_to_wpmesh,
)

from warp_math_utils import (
    hemisphere_sample,
    hemisphere_pdf,
    get_coord_system,
    triangle_sample,
    power_heuristic,
    area_to_solid_angle,
    get_triangle_points,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EPSILON = 1e-4


@wp.struct
class Material:
    type: wp.uint16  # 0: diffuse, 1: specular
    color: wp.vec3


@wp.struct
class Mesh:
    mesh_id: wp.uint64  # warp mesh id
    material_id: wp.uint64
    num_faces: wp.uint32
    is_light: wp.bool
    light_intensity: wp.float32
    light_color: wp.vec3


@wp.struct
class PathSegment:
    radiance: wp.vec3  # estimated radiance
    throughput: wp.vec3  # current throughput of the path
    point: wp.vec3  # current hit point
    ray_dir: wp.vec3  # current ray direction
    pixel_idx: int  # associated pixel index
    depth: int  # current path depth
    prev_bsdf_pdf: float  # previous BSDF PDF, used by MIS for indirect paths


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
def shade(
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
    mesh: Mesh,
    material: Material,
) -> wp.vec3:
    if material.type == 0:  # diffuse
        return material.color / wp.pi

    return wp.vec3(0.0, 0.0, 0.0)


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
    Returns the index of the hit mesh, the normal at the hit point, the hit point, and the hit face index.
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

    return hit_mesh_idx, hit_normal, hit_point, hit_face_idx


@wp.func
def light_sample(
    rand_state: wp.uint32,
    meshes: wp.array(dtype=Mesh),
    light_indices: wp.array(dtype=int),
) -> tuple[int, wp.vec3, wp.vec3, float]:
    """
    Sample a random point on the light sources in the scene.
    Returns the mesh id of the light source, the point on the light source, the normal, and the probability.
    If there is no light in the scene, returns -1, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), and 0.0.
    """
    num_lights = len(light_indices)
    if num_lights == 0:
        return -1, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0
    light_idx = wp.randi(rand_state, 0, num_lights)
    light_mesh = meshes[light_indices[light_idx]]

    mesh_id = light_mesh.mesh_id
    num_faces = wp.mesh_get(mesh_id).indices.shape[0] // 3
    face_idx = wp.randi(rand_state, 0, num_faces)

    normal = wp.mesh_eval_face_normal(mesh_id, face_idx)
    v1, v2, v3 = get_triangle_points(mesh_id, face_idx)
    pos, _, pdf = triangle_sample(rand_state, v1, v2, v3)

    # this is not uniform area sampling wrt. all light surfaces
    # but for simplicity we'll do it this way for now
    pdf = pdf / float(num_lights * num_faces)
    return light_indices[light_idx], pos, normal, pdf


@wp.func
def compute_light_pdf(
    p: wp.vec3,
    mesh_id: wp.uint64,
    face_idx: int,
    meshes: wp.array(dtype=Mesh),
    light_indices: wp.array(dtype=int),
) -> float:
    num_lights = len(light_indices)
    if num_lights == 0:
        return 0.0
    num_faces = wp.mesh_get(mesh_id).indices.shape[0] // 3
    if num_faces == 0:
        return 0.0

    v1, v2, v3 = get_triangle_points(mesh_id, face_idx)
    area = 0.5 * wp.length(wp.cross(v2 - v1, v3 - v1))
    if area <= 1e-5:
        return 0.0

    return 1.0 / (area * float(num_lights * num_faces))


@wp.func
def is_visible(
    target_light_mesh_idx: int, p1: wp.vec3, p2: wp.vec3, meshes: wp.array(dtype=Mesh)
):
    """
    Checks if a ray from p1 to p2 is occluded
    """
    p1p2 = p2 - p1
    max_t = wp.length(p1p2)
    rd = wp.normalize(p1p2)
    origin = p1 + rd * EPSILON
    hit_mesh_idx, _, __, ___ = scene_intersect(origin, rd, meshes, 0.0, max_t + EPSILON)
    return hit_mesh_idx == target_light_mesh_idx


# path tracing
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
            light_pdf = compute_light_pdf(
                hit_point, mesh.mesh_id, hit_face_idx, meshes, light_indices
            )
            light_pdf_solid_angle = (
                area_to_solid_angle(ro, hit_point, hit_normal) * light_pdf
            )
            mis_weight = power_heuristic(path.prev_bsdf_pdf, light_pdf_solid_angle)
            path.radiance += mis_weight * wp.cw_mul(
                path.throughput, mesh.light_color * mesh.light_intensity
            )
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        # Light sampling (NEE) (1 Light sample)
        light_mesh_idx, light_sample_point, light_normal, light_pdf = light_sample(
            rand_state, meshes, light_indices
        )
        light_pdf_solid_angle = (
            area_to_solid_angle(hit_point, light_sample_point, light_normal) * light_pdf
        )

        if (
            light_mesh_idx != -1
            and light_pdf_solid_angle > 0
            and is_visible(light_mesh_idx, hit_point, light_sample_point, meshes)
        ):
            wi = world_to_normal_space * wp.normalize(light_sample_point - hit_point)
            if wi.z < 0.0:
                path.radiance = wp.vec3(0.0, 0.0, 0.0)
                path_segments[tid] = path
                path_flags[tid] = 1
                return

            bsdf_pdf = hemisphere_pdf(wi)
            mis_weight = power_heuristic(light_pdf_solid_angle, bsdf_pdf)
            path.radiance += mis_weight * wp.cw_mul(
                path.throughput,
                shade(wi, wo, mesh, mat) * wi.z / light_pdf_solid_angle,
            )

        # BSDF sampling
        wi = hemisphere_sample(rand_state)
        pdf = hemisphere_pdf(wi)
        if pdf <= 0.0 or wi.z < 0.0:
            path.radiance = wp.vec3(0.0, 0.0, 0.0)
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        path.point = hit_point + hit_normal * EPSILON
        path.ray_dir = normal_to_world_space * wi
        path.throughput = wp.cw_mul(
            path.throughput,
            shade(wi, wo, mesh, mat) * wi.z / pdf,
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
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        path.throughput = throughput / p_rr
        # hard stop
        if path.depth >= max_depth:
            path.radiance = wp.vec3(0.0, 0.0, 0.0)
            path_segments[tid] = path
            path_flags[tid] = 1
            return

        # cache the BSDF PDF
        path.prev_bsdf_pdf = pdf
        path_segments[tid] = path
    else:
        path.radiance = wp.vec3(0.0, 0.0, 0.0)
        path_segments[tid] = path
        path_flags[tid] = 1


# monte carlo integration
@wp.kernel
def accumulate(
    path_segments: wp.array(dtype=PathSegment),
    pixels: wp.array(dtype=wp.vec3),
    iteration: int,
):
    tid = wp.tid()
    num_samples = iteration + 1
    pixel_idx = path_segments[tid].pixel_idx
    pixels[pixel_idx] += (path_segments[tid].radiance - pixels[pixel_idx]) / float(
        num_samples
    )


@dataclass
class CameraParams:
    pos: tuple[float] = (0, 1, 2)
    fov_x: float = np.deg2rad(60)
    fov_y: float = 2 * np.atan(9 / 16 * np.tan(np.deg2rad(60) / 2))
    clipping_range: tuple[float] = (0, 1.0e6)


class Renderer:
    def __init__(
        self, width: int, usd_path: pathlib.Path, max_iter: int, max_depth: int
    ):
        self.cam_params = CameraParams()

        asset_stage: Usd.Stage = Usd.Stage.Open(usd_path.as_posix())

        self.meshes = []
        self.wp_meshes = []
        self.light_indices = []
        self.materials = []
        mat_prim_path_to_mat_id = {}

        def create_material(prim: Usd.Prim):
            type = prim.GetAttribute("pbr:type").Get()
            color = prim.GetAttribute("pbr:color").Get()
            material = Material()

            if type == "diffuse":
                material.type = 0
            elif type == "specular":
                material.type = 1
            else:
                raise ValueError(f"Unsupported material type: {type}")

            material.color = color
            return material

        def create_mesh(wp_mesh: wp.Mesh, prim: Usd.Prim, is_light: bool):
            mesh = Mesh()
            mesh.mesh_id = wp_mesh.id
            mesh.is_light = is_light
            if is_light:
                mesh.light_intensity = prim.GetAttribute("inputs:intensity").Get()
                mesh.light_color = prim.GetAttribute("inputs:color").Get()

            if prim.HasAPI(UsdShade.MaterialBindingAPI):
                material_binding = UsdShade.MaterialBindingAPI(prim)
                for binding in material_binding.GetDirectBindingRel().GetTargets():
                    # only the first material is used
                    material_path = binding.GetPrimPath()
                    mesh.material_id = wp.uint64(mat_prim_path_to_mat_id[material_path])
            else:
                mesh.material_id = wp.uint64(0)

            return mesh

        # collect all materials
        for prim in asset_stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material = create_material(prim)
                self.materials.append(material)
                mat_prim_path_to_mat_id[prim.GetPrimPath()] = len(self.materials) - 1

        # create rendering primitives
        for prim in asset_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_geom = UsdGeom.Mesh(prim)
                self.wp_meshes.append(usdmesh_to_wpmesh(mesh_geom))

                if prim.HasAPI(UsdLux.MeshLightAPI):
                    self.meshes.append(create_mesh(self.wp_meshes[-1], prim, True))
                    self.light_indices.append(len(self.meshes) - 1)
                else:
                    self.meshes.append(create_mesh(self.wp_meshes[-1], prim, False))

            elif prim.IsA(UsdLux.RectLight):
                rect_light = UsdLux.RectLight(prim)
                hx = rect_light.GetWidthAttr().Get() * 0.5
                hy = rect_light.GetHeightAttr().Get() * 0.5
                # vertices in local space (XY plane)
                transform = get_world_transform(prim)
                verts = [
                    transform.TransformAffine((-hx, -hy, 0.0)),  # bottom-left
                    transform.TransformAffine((hx, -hy, 0.0)),  # bottom-right
                    transform.TransformAffine((hx, hy, 0.0)),  # top-right
                    transform.TransformAffine((-hx, hy, 0.0)),  # top-left
                ]
                # two triangles (indices into verts)
                indices = [0, 1, 2, 0, 2, 3]
                self.wp_meshes.append(
                    wp.Mesh(
                        points=wp.array(verts, dtype=wp.vec3),
                        indices=wp.array(indices, dtype=int),
                    )
                )
                self.meshes.append(create_mesh(self.wp_meshes[-1], prim, True))
                self.light_indices.append(len(self.meshes) - 1)

            elif prim.IsA(UsdGeom.Camera):
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

        self.width = width
        self.height = int(
            width
            * np.tan(self.cam_params.fov_y * 0.5)
            / np.tan(self.cam_params.fov_x * 0.5)
        )
        logger.info(f"Using Camera Params: {self.cam_params}")
        logger.info(f"Loaded {len(self.meshes)} meshes")

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.meshes = wp.array(self.meshes, dtype=Mesh)
        self.materials = wp.array(self.materials, dtype=Material)
        self.light_indices = wp.array(self.light_indices, dtype=int)
        self.max_iter = max_iter
        self.max_depth = max_depth

        self._num_iter = 0
        self._path_segments = wp.zeros(
            self.width * self.height, dtype=PathSegment, device="cuda"
        )
        self._path_flags = wp.zeros(self.width * self.height, dtype=int, device="cuda")

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

            wp.launch(
                kernel=accumulate,
                dim=self.width * self.height,
                inputs=[
                    self._path_segments,
                    self.pixels,
                    self._num_iter,
                ],
            )
            self._num_iter += 1

    def reset(self):
        self._num_iter = 0


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
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum path depth.")

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
        ax.set_title("PBR Renderer")
        ax.set_axis_off()

        im = ax.imshow(
            renderer.pixels.numpy().reshape((renderer.height, renderer.width, 3)),
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

            def tonemap(x):
                y = x / (1.0 + x)
                return np.power(y, 1 / 2.2)

            im.set_data(
                tonemap(
                    renderer.pixels.numpy().reshape(
                        (renderer.height, renderer.width, 3)
                    )
                )
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            label.set_text(f"Iteration: {renderer._num_iter}")

        plt.close()
