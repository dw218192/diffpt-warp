"""
Quick and dirty PBR exercise -- using Nvidia's warp library and Pixar's OpenUSD to implement a simple path tracer.
Techniques:
    1. Progressive path tracing
    2. Multiple importance sampling/ Next Event Estimation using direct light sampling
    3. Russian Roulette for termination
    4. Monte Carlo integration

https://nvidia.github.io/warp/modules/functions.html
"""

import contextlib
from dataclasses import dataclass
import logging
import pathlib
import time
import imageio.v2 as imageio
import numpy as np
from pxr import Usd, UsdGeom, UsdShade

import warp as wp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from usd_utils import extract_pos
from warp_math_utils import (
    get_coord_system,
    triangle_sample,
    power_heuristic,
    area_to_solid_angle,
    get_triangle_points,
    hash3,
)

from mesh import (
    Mesh,
    create_mesh_from_usd_prim,
)
from material import (
    Material,
    create_material_from_usd_prim,
    is_emissive,
    mat_eval_bsdf,
    mat_sample,
    mat_pdf,
)
from learning import LearningSession

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EPSILON = 1e-4
BVH_THRESHOLD = (
    20  # if less than this number of meshes, use linear search instead of bvh
)


# per-pixel
@wp.struct
class PathSegment:
    point: wp.vec3  # current hit point
    ray_dir: wp.vec3  # current ray direction
    pixel_idx: int  # associated pixel index
    depth: int  # current path depth, -1 if terminated
    prev_bsdf_pdf: float  # previous BSDF PDF, used by MIS for indirect paths
    throughput: wp.vec3  # current throughput
    radiance: wp.vec3  # current radiance


# per-pixel
@wp.struct
class ReplayPathData:
    pixel_idx: int  # associated pixel index
    path_len: int  # length of the path, 0 if invalid
    terminal_contrib: wp.vec3  # contribution at terminal light-hit


# per-depth/bounce
# i-th bounce data is logically stored at bounce_data[tid, i - 1]
@wp.struct
class ReplayBounceData:
    hit_mat_id: int
    wo_local: wp.vec3

    # BSDF-sampled continuation
    wi_bsdf_local: wp.vec3
    pdf_bsdf: float
    throughput_before: wp.vec3

    # NEE (optional per bounce)
    nee_valid: int
    wi_nee_local: wp.vec3
    light_pdf_solid_angle: float
    mis_w_nee: float
    Le_nee: wp.vec3  # light emission at sampled light (rgb)

    # emissive add at surface after throughput update (optional)
    add_emissive: int  # 1 if is_emissive(mat) was applied in forward

    # russian roulette
    inv_p_rr: float  # 1.0 / p_rr


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


# ------------------------------------------------------------------------------------------------
# Wavefront Path Tracing kernels


@wp.kernel
def init_path_segments(
    # Read
    width: int,
    height: int,
    cam_pos: wp.vec3,
    fov_x: float,
    fov_y: float,
    # Write
    path_segments: wp.array(dtype=PathSegment),
):
    """
    Step 0: Initialize path segments with camera rays.
    """
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

    path_segments[tid].point = ro
    path_segments[tid].ray_dir = rd
    path_segments[tid].pixel_idx = tid
    path_segments[tid].depth = 0
    path_segments[tid].prev_bsdf_pdf = 0.0
    path_segments[tid].throughput = wp.vec3(1.0)
    path_segments[tid].radiance = wp.vec3(0.0)


@wp.kernel
def intersect(
    # Read
    bvh_id: wp.uint64,
    path_segments: wp.array(dtype=PathSegment),
    meshes: wp.array(dtype=Mesh),
    max_t: float,
    # Write
    out_hits: wp.array(dtype=HitData),
):
    """
    Step 1: Intersect path segments with the scene geometries (including lights).
    """
    tid = wp.tid()
    if path_segments[tid].depth == -1:
        return
    path = path_segments[tid]
    out_hits[tid] = scene_intersect(
        bvh_id, path.point, path.ray_dir, meshes, max_t, False
    )


@wp.kernel
def shade(
    # Read
    record_path_replay_data: bool,
    bvh_id: wp.uint64,
    hits: wp.array(dtype=HitData),
    meshes: wp.array(dtype=Mesh),
    light_indices: wp.array(dtype=int),
    materials: wp.array(dtype=Material),
    max_depth: int,
    iteration: int,
    # Read/Write
    path_segments: wp.array(dtype=PathSegment),
    num_finished_paths: wp.array(dtype=int),
    # dummy value if not recording replay data
    # if replay_bounce_data is true, they are expected to be of correct shapes
    replay_bounce_data: wp.array(dtype=ReplayBounceData),
    replay_path_data: wp.array(dtype=ReplayPathData),
):
    """
    Step 2: Shade the path segments.
    """
    tid = wp.tid()
    path = path_segments[tid]
    if path.depth == -1:
        return

    ro = path.point
    rd = path.ray_dir
    rand_state = wp.rand_init(hash3(tid, path.depth, iteration))
    radiance = path.radiance
    throughput = path.throughput

    hit_data = hits[tid]
    hit_mesh_idx = hit_data.mesh_idx
    hit_face_idx = hit_data.face_idx
    hit_point = hit_data.hit_point
    hit_normal = hit_data.hit_normal

    replay_bounce_data_idx = tid * max_depth + path.depth

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

            contrib = mis_weight * wp.cw_mul(
                throughput, mesh.light_color * mesh.light_intensity
            )
            radiance += contrib

            # finalize replay data for terminal light hit
            if record_path_replay_data:
                path_data = replay_path_data[tid]
                path_data.terminal_contrib = contrib
                path_data.pixel_idx = path.pixel_idx
                path_data.path_len = path.depth + 1
                replay_path_data[tid] = path_data

            path.radiance = radiance
            path.throughput = throughput
            path.depth = -1
            path_segments[tid] = path

            wp.atomic_add(num_finished_paths, 0, 1)

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
                bvh_id,
                light_mesh_idx,
                light_face_idx,
                hit_point,
                light_sample_point,
                meshes,
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
                    throughput,
                    mat_eval_bsdf(mat, wi, wo) * wi.z / light_pdf_solid_angle,
                )

                Le = light_mesh.light_intensity * light_mesh.light_color
                contrib = wp.cw_mul(contrib, Le)

                radiance += mis_weight * contrib

                # record NEE contribution for replay
                if record_path_replay_data:
                    bounce_data = replay_bounce_data[replay_bounce_data_idx]
                    bounce_data.nee_valid = 1
                    bounce_data.wi_nee_local = wi
                    bounce_data.light_pdf_solid_angle = light_pdf_solid_angle
                    bounce_data.mis_w_nee = mis_weight
                    bounce_data.Le_nee = Le
                    replay_bounce_data[replay_bounce_data_idx] = bounce_data

        # BSDF sampling
        wi = mat_sample(mat, rand_state, wo)
        pdf = mat_pdf(mat, wi, wo)
        if pdf <= 0.0 or wi.z < 0.0:
            path.radiance = wp.vec3(0.0)
            path.throughput = wp.vec3(1.0)
            path.depth = -1
            path_segments[tid] = path
            wp.atomic_add(num_finished_paths, 0, 1)

            # invalidate replay data
            if record_path_replay_data:
                replay_path_data[tid].path_len = 0

            return

        # record bsdf continuation for replay
        if record_path_replay_data:
            bounce_data = replay_bounce_data[replay_bounce_data_idx]
            bounce_data.wi_bsdf_local = wi
            bounce_data.pdf_bsdf = pdf
            bounce_data.throughput_before = throughput
            replay_bounce_data[replay_bounce_data_idx] = bounce_data

        path.point = hit_point + hit_normal * EPSILON
        path.ray_dir = normal_to_world_space * wi
        throughput = wp.cw_mul(throughput, mat_eval_bsdf(mat, wi, wo) * wi.z / pdf)

        if is_emissive(mat):
            radiance += wp.cw_mul(
                throughput, mat.emissive_color * mat.emissive_intensity
            )

            # record emissive contribution for replay
            if record_path_replay_data:
                bounce_data = replay_bounce_data[replay_bounce_data_idx]
                bounce_data.add_emissive = 1
                replay_bounce_data[replay_bounce_data_idx] = bounce_data

        # russain roulette
        # the brighter the path, the higher the chance of its survival
        p_rr = wp.clamp(
            wp.max(throughput.x, wp.max(throughput.y, throughput.z)), 0.22, 1.0
        )

        if wp.randf(rand_state) > p_rr:
            radiance = wp.vec3(0.0)
            path.radiance = radiance
            path.throughput = throughput
            path.depth = -1
            wp.atomic_add(num_finished_paths, 0, 1)

            # invalidate replay data
            if record_path_replay_data:
                replay_path_data[tid].path_len = 0

            return

        inv_p_rr = 1.0 / p_rr

        # record russian roulette for replay
        if record_path_replay_data:
            bounce_data = replay_bounce_data[replay_bounce_data_idx]
            bounce_data.inv_p_rr = inv_p_rr
            replay_bounce_data[replay_bounce_data_idx] = bounce_data

        throughput = throughput * inv_p_rr

        # hard stop
        if path.depth == max_depth - 1:
            radiance = wp.vec3(0.0)
            path.radiance = radiance
            path.throughput = throughput
            path.depth = -1
            path_segments[tid] = path
            wp.atomic_add(num_finished_paths, 0, 1)

            # invalidate replay data
            if record_path_replay_data:
                replay_path_data[tid].path_len = 0

            return

        # cache the BSDF PDF
        path.prev_bsdf_pdf = pdf
        path.radiance = radiance
        path.throughput = throughput
        path.depth = path.depth + 1
        path_segments[tid] = path
    else:
        path.radiance = wp.vec3(0.0)
        path.throughput = wp.vec3(1.0)
        path.depth = -1
        path_segments[tid] = path
        wp.atomic_add(num_finished_paths, 0, 1)


@wp.kernel
def replay_path(
    # Read
    max_depth: int,
    replay_bounce_data: wp.array(dtype=ReplayBounceData),
    replay_path_data: wp.array(dtype=ReplayPathData),
    materials: wp.array(dtype=Material),
    # Write
    out_radiance: wp.array(dtype=wp.vec3),
):
    """
    Replays the path through the recorded bounce data to approximate gradients with respect to material parameters.
    """
    tid = wp.tid()
    radiance = wp.vec3(0.0)
    throughput = wp.vec3(1.0)

    path_data = replay_path_data[tid]
    if path_data.path_len <= 0:
        out_radiance[tid] = wp.vec3(0.0)
        return

    for d in range(path_data.path_len - 1):
        rec = replay_bounce_data[tid * max_depth + d]

        mat = materials[rec.hit_mat_id]
        wo = rec.wo_local

        # --- NEE term at this bounce (if recorded) ---
        if rec.nee_valid != 0:
            wiL = rec.wi_nee_local
            # replay same estimator form as shade():
            # contrib = throughput_before * (bsdf * cos / light_pdf_solid_angle) * Le * mis_w
            fL = mat_eval_bsdf(mat, wiL, wo)
            cL = wp.cw_mul(
                rec.throughput_before, fL * wiL.z / rec.light_pdf_solid_angle
            )
            cL = wp.cw_mul(cL, rec.Le_nee)
            radiance += rec.mis_w_nee * cL

        # --- BSDF continuation throughput update ---
        wi = rec.wi_bsdf_local
        pdf = rec.pdf_bsdf

        if pdf <= 0.0 or wi.z <= 0.0:
            out_radiance[tid] = wp.vec3(0.0)
            return

        f = mat_eval_bsdf(mat, wi, wo)
        throughput = wp.cw_mul(rec.throughput_before, f * wi.z / pdf)

        if rec.add_emissive != 0:
            radiance += wp.cw_mul(
                throughput, mat.emissive_color * mat.emissive_intensity
            )

        throughput = throughput * rec.inv_p_rr

    # --- Terminal mesh-light hit contribution ---
    radiance += path_data.terminal_contrib
    out_radiance[path_data.pixel_idx] = radiance


# monte carlo integration
@wp.kernel
def accumulate(
    # Read
    num_samples: int,
    path_segments: wp.array(dtype=PathSegment),
    # Write
    radiance_sum: wp.array(dtype=wp.vec3),
    radiance: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    pixel_idx = path_segments[tid].pixel_idx
    radiance_sum[pixel_idx] += path_segments[tid].radiance
    radiance[pixel_idx] = radiance_sum[pixel_idx] / float(num_samples)


@wp.kernel
def tonemap(
    # Read
    radiances: wp.array(dtype=wp.vec3),
    # Write
    pixels: wp.array(dtype=wp.vec3),
):
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


@dataclass
class CameraParams:
    pos: tuple[float] = (0, 1, 2)
    fov_x: float = np.deg2rad(60)
    fov_y: float = 2 * np.atan(9 / 16 * np.tan(np.deg2rad(60) / 2))
    clipping_range: tuple[float] = (0, 1.0e6)


class Renderer:
    @dataclass
    class ReplayData:
        is_valid: bool
        bounce_data: wp.array(dtype=ReplayBounceData)
        path_data: wp.array(dtype=ReplayPathData)

    """
    Simple USD-based path tracer.

    Args:
        width: The width of the output image in pixels.
        usd_path: The path to the USD file to render.
        spp: The number of samples per pixel.
        max_depth: The maximum depth of the path.
        force_bvh: Whether to force the use of a BVH.

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
        spp: int,
        max_depth: int,
        force_bvh: bool,
    ):  # height will be computed from width and fov set in the USD stage's camera prim

        self.cam_params = CameraParams()
        self.usd_path = usd_path.resolve()

        asset_stage: Usd.Stage = Usd.Stage.Open(usd_path.as_posix())

        self.meshes = []
        self.wp_meshes = []
        self.light_indices = []
        self.materials = []

        # aabb values for bvh construction
        self.bvh_lower_bounds = []
        self.bvh_upper_bounds = []

        mat_prim_path_to_mat_id = {}

        # collect all materials
        for prim in asset_stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material = create_material_from_usd_prim(UsdShade.Material(prim))
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

                wp_mesh = res.warp_mesh
                mesh = res.mesh
                aabb = res.aabb
                self.bvh_lower_bounds.append(aabb[0])
                self.bvh_upper_bounds.append(aabb[1])
                self.wp_meshes.append(wp_mesh)
                self.meshes.append(mesh)
                if mesh.is_light:
                    self.light_indices.append(len(self.meshes) - 1)

        if len(self.meshes) > BVH_THRESHOLD or force_bvh:
            self.bvh = wp.Bvh(
                wp.array(self.bvh_lower_bounds, dtype=wp.vec3),
                wp.array(self.bvh_upper_bounds, dtype=wp.vec3),
                "lbvh",
            )
        else:
            self.bvh = None

        self.width = width
        self.height = int(
            width
            * np.tan(self.cam_params.fov_y * 0.5)
            / np.tan(self.cam_params.fov_x * 0.5)
        )
        logger.info(f"Using Camera Params: {self.cam_params}")
        logger.info(
            f"Loaded {len(self.meshes)} meshes ({len(self.light_indices)} lights)"
        )

        # stores the running estimates of the radiances for each pixel
        self.radiances = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.radiance_sum = wp.zeros(self.width * self.height, dtype=wp.vec3)

        self.meshes = wp.array(self.meshes, dtype=Mesh)
        self.materials = wp.array(self.materials, dtype=Material)
        self.light_indices = wp.array(self.light_indices, dtype=int)
        self.max_iter = spp
        self.max_depth = max_depth

        self._num_iter = 0
        self._path_segments = wp.zeros(self.width * self.height, dtype=PathSegment)
        self._num_finished_paths = wp.zeros(1, dtype=int)
        self._hits = wp.zeros(self.width * self.height, dtype=HitData)
        self._first_hits = wp.zeros_like(self._hits)

    @property
    def bvh_id(self):
        return self.bvh.id if self.bvh else wp.uint64(-1)

    @property
    def num_iter(self):
        return self._num_iter

    def render(self, record_path_replay_data: bool = False):
        """
        Render a single iteration of the path tracer.
        If record_path_replay_data is True, extra data will be recorded to allow path replay backpropagation.
        """
        if self._num_iter >= self.max_iter:
            return False

        if record_path_replay_data:
            replay_data = self.ReplayData(
                is_valid=True,
                bounce_data=wp.zeros(
                    self.width * self.height * self.max_depth, dtype=ReplayBounceData
                ),
                path_data=wp.zeros(self.width * self.height, dtype=ReplayPathData),
            )
        else:
            replay_data = self.ReplayData(
                is_valid=False,
                bounce_data=wp.zeros(0, dtype=ReplayBounceData),
                path_data=wp.zeros(0, dtype=ReplayPathData),
            )

        with wp.ScopedTimer("render single iteration") as timer:
            timer.extra_msg = f"iteration {self._num_iter + 1}/{self.max_iter}"
            wp.launch(
                kernel=init_path_segments,
                dim=self.width * self.height,
                inputs=[
                    self.width,
                    self.height,
                    self.cam_params.pos,
                    self.cam_params.fov_x,
                    self.cam_params.fov_y,
                ],
                outputs=[
                    self._path_segments,
                ],
            )

            # cache first hits since we're sending just 1 ray per pixel (no measurement integral involved)
            if self._num_iter == 0:
                wp.launch(
                    kernel=intersect,
                    dim=self.width * self.height,
                    inputs=[
                        self.bvh_id,
                        self._path_segments,
                        self.meshes,
                        1e6,
                    ],
                    outputs=[self._first_hits],
                )

            self._num_finished_paths.zero_()
            for depth in range(self.max_depth):
                current_hits = self._first_hits
                if depth > 0:
                    wp.launch(
                        kernel=intersect,
                        dim=self.width * self.height,
                        inputs=[
                            self.bvh_id,
                            self._path_segments,
                            self.meshes,
                            1e6,
                        ],
                        outputs=[self._hits],
                    )
                    current_hits = self._hits

                wp.launch(
                    kernel=shade,
                    dim=self.width * self.height,
                    inputs=[
                        record_path_replay_data,
                        self.bvh_id,
                        current_hits,
                        self.meshes,
                        self.light_indices,
                        self.materials,
                        self.max_depth,
                        self._num_iter,
                    ],
                    outputs=[
                        self._path_segments,
                        self._num_finished_paths,
                        replay_data.bounce_data,
                        replay_data.path_data,
                    ],
                )
                if self._num_finished_paths.numpy()[0] == self.width * self.height:
                    break

            wp.launch(
                kernel=accumulate,
                dim=self.width * self.height,
                inputs=[
                    self._num_iter + 1,
                    self._path_segments,
                ],
                outputs=[self.radiance_sum, self.radiances],
            )
            self._num_iter += 1

        return self._num_iter < self.max_iter

    def replay(self):
        pass

    def get_pixels(self, use_tonemap: bool = True) -> np.ndarray:
        if use_tonemap:
            pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

            wp.launch(
                kernel=tonemap,
                dim=self.width * self.height,
                inputs=[self.radiances],
                outputs=[pixels],
            )
            return pixels.numpy().reshape((self.height, self.width, 3))
        else:
            return self.radiances.numpy().reshape((self.height, self.width, 3))

    def reset(self):
        self._num_iter = 0
        self.radiances.zero_()
        self.radiance_sum.zero_()


g_running = True


@contextlib.contextmanager
def record_frame(frame_times_list):
    info = {"elapsed": 0.0}
    t0 = time.perf_counter()
    yield info
    elapsed = time.perf_counter() - t0
    info["elapsed"] = elapsed
    frame_times_list.append(elapsed)


def log_frame_time_stats(frame_times: list[float], spp: int):
    if not frame_times:
        return
    avg_ms = 1000.0 * sum(frame_times) / len(frame_times)
    min_ms = 1000.0 * min(frame_times)
    max_ms = 1000.0 * max(frame_times)
    logger.info(
        "Frame time stats (spp=%d): avg=%.2f ms, min=%.2f ms, max=%.2f ms",
        spp,
        avg_ms,
        min_ms,
        max_ms,
    )


def save_image(
    renderer: Renderer, path_arg: pathlib.Path | None, save_raw: bool = False
):
    """
    Saves the current image to a directory. The filename matches the USD stage
    name and the extension is selected via save_raw. If path_arg is None, the
    repo root is used as the output directory. Returns the resolved output path
    if saved, otherwise None.
    """
    extension = "hdr" if save_raw else "png"
    stage_stem = (
        renderer.usd_path.stem
        if isinstance(getattr(renderer, "usd_path", None), pathlib.Path)
        else "render"
    )

    output_dir = (path_arg or pathlib.Path(__file__).parent.parent).resolve()
    output_path = (output_dir / f"{stage_stem}.{extension}").resolve()

    pixels = renderer.get_pixels(use_tonemap=not save_raw)
    pixels = pixels[::-1, ...]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_raw:
        hdr_pixels = pixels.astype(np.float32)
        imageio.imwrite(output_path.as_posix(), hdr_pixels, format="HDR-FI")
    else:
        imageio.imwrite(
            output_path.as_posix(),
            (pixels * 255.0).clip(0.0, 255.0).astype(np.uint8),
        )


def load_target_image(target_path: pathlib.Path, renderer: Renderer) -> np.ndarray:
    """
    Load and validate a target RGB image for learning/visualization.
    Ensures shape/channel match and aligns orientation with the renderer output.
    Only HDR inputs are accepted to avoid unintended 8-bit gamma/tonemap issues.
    """
    suffix = target_path.suffix.lower()
    allowed_hdr_exts = {".hdr", ".exr"}
    if suffix and suffix not in allowed_hdr_exts:
        raise ValueError(
            f"Target image must be HDR ({', '.join(sorted(allowed_hdr_exts))}); got {suffix or 'no extension'}."
        )

    img = imageio.imread(target_path)
    if img.ndim == 2:
        raise ValueError("Target image must be RGB, got grayscale.")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] != 3:
        raise ValueError("Target image must have 3 channels (RGB).")

    img = img.astype(np.float32)
    if suffix not in allowed_hdr_exts:
        # Non-HDR inputs (if ever allowed) would be rescaled from 0-255 range.
        if img.max() > 1.5:
            img /= 255.0

    if img.shape[0] != renderer.height or img.shape[1] != renderer.width:
        raise ValueError(
            f"Target image shape {img.shape[:2]} does not match renderer output "
            f"({renderer.height}, {renderer.width})."
        )

    # Invert and flatten to match renderer's in-memory layout.
    return img[::-1, ...].reshape(-1)


def do_render_loop(
    renderer: Renderer,
    *,
    save_path: pathlib.Path | None,
    save_raw: bool = False,
    fig: plt.Figure | None = None,
    render_image: mpimg.AxesImage | None = None,
    label: plt.Text | None = None,
):
    frame_times = []
    while g_running:
        with record_frame(frame_times) as info:
            can_continue = renderer.render()
        frame_ms = info["elapsed"] * 1000.0
        if save_path and not can_continue:
            save_image(renderer, save_path, save_raw=save_raw)
            break

        if render_image:
            render_image.set_data(renderer.get_pixels())

        if fig:
            fig.canvas.draw()
            fig.canvas.flush_events()

        if label:
            label.set_text(f"Iteration: {renderer.num_iter} ({frame_ms:.1f} ms)")
        if not can_continue:
            break

    log_frame_time_stats(frame_times, renderer.num_iter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--usd-path",
        type=pathlib.Path,
        default=(
            pathlib.Path(__file__).parent.parent / "stages" / "cornell.usda"
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
        "--spp", type=int, default=100, help="Number of samples per pixel."
    )
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum path depth.")
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=None,
        help="Directory to save the rendered image; filename matches the USD stage.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save the raw radiance output as an HDR image.",
    )
    parser.add_argument(
        "--force-bvh",
        action="store_true",
        help=f"Force the use of a BVH. By default, a BVH is used if the number of meshes is greater than {BVH_THRESHOLD}.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window; render to completion and save the image.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )

    diffrt_group = parser.add_argument_group("Differentiable Rendering")
    diffrt_group.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate."
    )
    diffrt_group.add_argument(
        "--learning-iter",
        type=int,
        default=10,
        help="Number of learning iterations.",
    )
    diffrt_group.add_argument(
        "--batch-spp",
        type=int,
        default=1,
        help="Number of samples per pixel during learning.",
    )
    diffrt_group.add_argument(
        "--target-image",
        type=pathlib.Path,
        default=None,
        help="Path to the target image. If not provided, learning will be disabled.",
    )

    args = parser.parse_args()

    if args.save_path:
        # Require a directory without an explicit filename/extension.
        if args.save_path.suffix:
            parser.error(
                "--save-path should be a directory (omit filename and extension)."
            )
        if args.save_path.exists() and not args.save_path.is_dir():
            parser.error("--save-path points to a file; provide a directory.")
        args.save_path = args.save_path.resolve()

    if args.save_raw:
        # Ensure the HDR writer is available up front; fail fast if not.
        try:
            from imageio.plugins import freeimage

            freeimage.download()  # idempotent; ensures plugin/DLL is present
        except Exception as exc:
            parser.error(f"HDR output requested but FreeImage is unavailable: {exc}")

    if args.debug:
        wp.config.verify_autodiff = True
        wp.config.verify_autograd_array_access = True

    with wp.ScopedDevice(args.device):
        renderer = Renderer(
            width=args.width,
            usd_path=args.usd_path,
            spp=args.spp,
            max_depth=args.max_depth,
            force_bvh=args.force_bvh,
        )

        if args.headless:
            logger.info("Running in headless mode (no UI).")

            if args.target_image is None:
                do_render_loop(
                    renderer, save_path=args.save_path, save_raw=args.save_raw
                )
            else:
                target_image = load_target_image(args.target_image, renderer)
                with LearningSession(
                    renderer,
                    target_image=target_image,
                    learning_rate=args.learning_rate,
                    max_epochs=args.learning_iter,
                ) as session:
                    while True:
                        loss = session.step()
                        if loss is None:
                            break
                        logger.info(
                            "Learning step %d/%d: loss=%.6f",
                            session.epoch,
                            session.max_epochs,
                            session.loss_value,
                        )
                do_render_loop(
                    renderer, save_path=args.save_path, save_raw=args.save_raw
                )
        else:
            plt.ion()  # turn on interactive mode
            fig = plt.figure()

            def handle_close(_):
                global g_running
                g_running = False

            fig.canvas.mpl_connect("close_event", handle_close)

            # main loop
            if args.target_image is None:
                ax = fig.add_subplot()
                # layout
                ax.set_title("PBR Renderer")
                ax.set_axis_off()

                render_image = ax.imshow(
                    np.zeros((renderer.height, renderer.width, 3)),
                    origin="lower",
                    interpolation="antialiased",
                    aspect="equal",
                )
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                # add label in top-left corner
                render_image_label = ax.text(
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

                # normal rendering loop
                do_render_loop(
                    renderer,
                    save_path=args.save_path,
                    save_raw=args.save_raw,
                    fig=fig,
                    render_image=render_image,
                    label=render_image_label,
                )
            else:
                # layout with render, per-pixel loss heatmap, and total loss curve
                gs = fig.add_gridspec(2, 2)
                ax_render = fig.add_subplot(gs[0, 0])
                ax_render.set_title("Render")
                ax_render.set_axis_off()
                render_image = ax_render.imshow(
                    np.zeros((renderer.height, renderer.width, 3)),
                    origin="lower",
                    interpolation="antialiased",
                    aspect="equal",
                )
                render_image_label = ax_render.text(
                    0.01,
                    0.99,
                    "Epoch: 0",
                    color="white",
                    fontsize=9,
                    ha="left",
                    va="top",
                    transform=ax_render.transAxes,
                )

                ax_per_pixel_loss = fig.add_subplot(gs[1, 0])
                ax_per_pixel_loss.set_title("Per-pixel Loss")
                ax_per_pixel_loss.set_axis_off()
                ax_per_pixel_loss_image = ax_per_pixel_loss.imshow(
                    np.zeros((renderer.height, renderer.width, 1)),
                    origin="lower",
                    interpolation="antialiased",
                    aspect="equal",
                    cmap="viridis",
                )

                ax_total_loss = fig.add_subplot(gs[1, 1])
                ax_total_loss.set_title("Total Loss")
                ax_total_loss.set_xlabel("Epoch")
                ax_total_loss.set_ylabel("Loss")
                (ax_total_loss_line,) = ax_total_loss.plot([], [])

                plt.show(block=False)

                target_image = load_target_image(args.target_image, renderer)
                losses = []

                with LearningSession(
                    renderer,
                    target_image=target_image,
                    learning_rate=args.learning_rate,
                    batch_spp=args.batch_spp,
                    max_epochs=args.learning_iter,
                ) as session:
                    while g_running:
                        loss = session.step()
                        if loss is None:
                            break

                        losses.append(session.loss_value)
                        render_image.set_data(renderer.get_pixels())
                        ax_per_pixel_loss_image.set_data(session.per_pixel_mse_host)
                        ax_per_pixel_loss_image.autoscale()

                        ax_total_loss_line.set_data(np.arange(len(losses)), losses)
                        ax_total_loss.relim()
                        ax_total_loss.autoscale_view()

                        render_image_label.set_text(
                            f"Epoch: {session.epoch}/{session.max_epochs} Loss: {session.loss_value:.4f}"
                        )
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                do_render_loop(
                    renderer,
                    save_path=args.save_path,
                    save_raw=args.save_raw,
                    fig=fig,
                    render_image=render_image,
                    label=render_image_label,
                )
            plt.close()
