"""
Use Nvidia's warp library to render a simple PBR scene.

https://nvidia.github.io/warp/modules/functions.html
"""

from dataclasses import dataclass
import logging
import os

import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdShade

import warp as wp
import matplotlib.pyplot as plt

from usd_utils import (
    extract_pos,
    get_world_transform,
    usdmesh_to_wpmesh,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@wp.struct
class Material:
    type: wp.uint16  # 0: diffuse, 1: specular, 2: light
    color: wp.vec3


@wp.struct
class Mesh:
    mesh_id: wp.uint64  # warp mesh id
    material_id: wp.uint64
    is_light: wp.bool
    light_intensity: wp.float32
    light_color: wp.vec3


@wp.struct
class PathSegment:
    throughput: wp.vec3
    point: wp.vec3  # current hit point
    ray_dir: wp.vec3  # current ray direction
    normal: wp.vec3  # current hit normal
    pixel_idx: int  # associated pixel index
    finished: wp.bool  # whether the path has finished
    depth: int  # current path depth


@wp.kernel
def init_path_segments(
    path_segments: wp.array(dtype=PathSegment),
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

    path_segments[tid].throughput = wp.vec3(1.0, 1.0, 1.0)
    path_segments[tid].point = ro
    path_segments[tid].ray_dir = rd
    path_segments[tid].normal = wp.vec3(0.0, 1.0, 0.0)
    path_segments[tid].pixel_idx = tid
    path_segments[tid].finished = wp.bool(False)


@wp.func
def hemisphere_sample(
    normal: wp.vec3,
) -> wp.vec3:
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def hemisphere_pdf(
    normal: wp.vec3,
) -> wp.float32:
    return 1.0 / (2.0 * wp.pi)


@wp.func
def shade(
    wi: wp.vec3,
    wo: wp.vec3,
    n: wp.vec3,
    mesh: Mesh,
    material: Material,
) -> wp.vec3:
    return wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def draw(
    path_segments: wp.array(dtype=PathSegment),
    meshes: wp.array(dtype=Mesh),
    materials: wp.array(dtype=Material),
    min_t: float,
    max_t: float,
    max_depth: int,
):
    tid = wp.tid()
    if path_segments[tid].finished:
        return

    ro = path_segments[tid].point
    rd = path_segments[tid].ray_dir

    t = wp.float32(max_t)
    hit_mesh_idx = wp.int32(-1)
    hit_normal = wp.vec3(0.0, 0.0, 0.0)
    hit_point = wp.vec3(0.0, 0.0, 0.0)

    for i in range(meshes.shape[0]):
        query = wp.mesh_query_ray(meshes[i].mesh_id, ro, rd, max_t)
        if query.result and query.t < t and query.t > min_t:
            t = query.t
            hit_mesh_idx = i
            hit_normal = query.normal
            hit_point = ro + t * rd

    if hit_mesh_idx != -1:
        mesh = meshes[hit_mesh_idx]
        mat = materials[mesh.material_id]
        if mesh.is_light:
            path_segments[tid].throughput = wp.cw_mul(
                path_segments[tid].throughput, mesh.light_color * mesh.light_intensity
            )
            path_segments[tid].finished = wp.bool(True)
            return
        else:
            path_segments[tid].finished = wp.bool(True)
            if hit_mesh_idx % 5 == 0:
                path_segments[tid].throughput = wp.vec3(1.0, 0.0, 0.0)
            elif hit_mesh_idx % 5 == 1:
                path_segments[tid].throughput = wp.vec3(0.0, 1.0, 0.0)
            elif hit_mesh_idx % 5 == 2:
                path_segments[tid].throughput = wp.vec3(0.0, 0.0, 1.0)
            elif hit_mesh_idx % 5 == 3:
                path_segments[tid].throughput = wp.vec3(1.0, 1.0, 0.0)
            elif hit_mesh_idx % 5 == 4:
                path_segments[tid].throughput = wp.vec3(1.0, 0.0, 1.0)
            return

        wi = -path_segments[tid].ray_dir

        path_segments[tid].normal = hit_normal
        path_segments[tid].point = hit_point
        path_segments[tid].ray_dir = hemisphere_sample(hit_normal)

        wo = path_segments[tid].ray_dir

        pdf = hemisphere_pdf(hit_normal)
        path_segments[tid].throughput = wp.cw_mul(
            path_segments[tid].throughput, shade(wi, wo, hit_normal, mesh, mat) / pdf
        )
        path_segments[tid].depth += 1
        if path_segments[tid].depth >= max_depth:
            path_segments[tid].finished = wp.bool(True)
    else:
        path_segments[tid].throughput = wp.vec3(0.0, 0.0, 0.0)
        path_segments[tid].finished = wp.bool(True)


@wp.kernel
def accumulate(
    path_segments: wp.array(dtype=PathSegment),
    pixels: wp.array(dtype=wp.vec3),
    iteration: int,
):
    tid = wp.tid()
    num_samples = iteration + 1
    pixel_idx = path_segments[tid].pixel_idx
    pixels[pixel_idx] += (path_segments[tid].throughput - pixels[pixel_idx]) / float(
        num_samples
    )


@dataclass
class CameraParams:
    pos: tuple[float] = (0, 1, 2)
    fov_x: float = np.deg2rad(60)
    fov_y: float = 2 * np.atan(9 / 16 * np.tan(np.deg2rad(60) / 2))
    clipping_range: tuple[float] = (0, 1.0e6)


class Renderer:
    def __init__(self, width=1024, usd_path=None, max_iter=100, max_depth=10):
        self.cam_params = CameraParams()

        asset_stage: Usd.Stage = Usd.Stage.Open(usd_path)

        self.meshes = []
        self.wp_meshes = []

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
            elif type == "light":
                material.type = 2
            else:
                raise ValueError(f"Unknown material type: {type}")

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

        for prim in asset_stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material = create_material(prim)
                self.materials.append(material)
                mat_prim_path_to_mat_id[prim.GetPrimPath()] = len(self.materials) - 1

        for prim in asset_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_geom = UsdGeom.Mesh(prim)
                self.wp_meshes.append(usdmesh_to_wpmesh(mesh_geom))
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
        self.max_iter = max_iter
        self.max_depth = max_depth

        self._num_iter = 0
        self._path_segments = wp.zeros(
            self.width * self.height, dtype=PathSegment, device="cuda"
        )

    def render(self):
        if self._num_iter >= self.max_iter:
            return

        if self._num_iter == 0:
            with wp.ScopedTimer("init path segments"):
                wp.launch(
                    kernel=init_path_segments,
                    dim=self.width * self.height,
                    inputs=[
                        self._path_segments,
                        self.width,
                        self.height,
                        self.cam_params.pos,
                        self.cam_params.fov_x,
                        self.cam_params.fov_y,
                    ],
                )

        with wp.ScopedTimer("render single iteration"):
            while True:
                wp.launch(
                    kernel=draw,
                    dim=self.width * self.height,
                    inputs=[
                        self._path_segments,
                        self.meshes,
                        self.materials,
                        self.cam_params.clipping_range[0],
                        self.cam_params.clipping_range[1],
                        self.max_depth,
                    ],
                )
                host_path_segments = self._path_segments.numpy()
                should_continue = False
                for i in range(self.width * self.height):
                    if not host_path_segments[i]["finished"]:
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
        type=str,
        default=os.path.join(os.path.dirname(__file__), "assets", "cornell.usda"),
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
            im.set_data(
                renderer.pixels.numpy().reshape((renderer.height, renderer.width, 3))
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            label.set_text(f"Iteration: {renderer._num_iter}")

        plt.close()
