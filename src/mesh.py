from dataclasses import dataclass
import warp as wp
from pxr import Usd, UsdGeom, UsdShade, UsdLux
import numpy as np
from usd_utils import get_world_transform
import logging

logger = logging.getLogger(__name__)


@wp.struct
class Mesh:
    mesh_id: wp.uint64  # warp mesh id
    material_id: wp.uint64
    num_faces: wp.int32
    is_light: wp.bool
    light_intensity: wp.float32
    light_color: wp.vec3


@dataclass
class MeshCreationInfo:
    warp_mesh: wp.Mesh
    mesh: Mesh
    aabb: tuple[wp.vec3, wp.vec3]


def get_mesh_points_and_indices(
    mesh_geom: UsdGeom.Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    # https://openusd.org/release/api/usd_geom_page_front.html#UsdGeom_WindingOrder
    # both USD and warp expect right-handed coordinate system
    # warp uses ccw winding order for some ray intersection tests

    transform = get_world_transform(mesh_geom.GetPrim())

    # get and transform vertex positions
    points = np.array(
        [transform.TransformAffine(p) for p in mesh_geom.GetPointsAttr().Get()],
        dtype=np.float32,
    )
    assert points.shape[1] == 3

    counts = np.array(mesh_geom.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    if not np.all(counts == 3):
        raise ValueError(
            f"Non-triangular faces detected; this is not supported: expected all counts == 3, got unique counts {np.unique(counts)}"
        )

    # sanity check for index count alignment
    assert indices.size == 3 * counts.size, "Indices don't align with face count!"

    # check degenerate triangles
    for i in range(indices.size // 3):
        v1 = points[indices[3 * i]]
        v2 = points[indices[3 * i + 1]]
        v3 = points[indices[3 * i + 2]]
        if np.allclose(v1, v2) or np.allclose(v1, v3) or np.allclose(v2, v3):
            raise ValueError(f"Degenerate triangle detected: {v1}, {v2}, {v3}")

    # check all indices are valid
    for i in range(indices.size):
        if indices[i] < 0 or indices[i] >= points.shape[0]:
            raise ValueError(
                f"Invalid index: {indices[i]}, expected range [0, {points.shape[0]})"
            )

    return points, indices


def create_mesh_from_usd_prim(
    prim: Usd.Prim,
    mat_prim_path_to_mat_id: dict[str, int],
) -> MeshCreationInfo | None:
    if prim.IsA(UsdGeom.Mesh):
        mesh_geom = UsdGeom.Mesh(prim)
        is_light = prim.HasAPI(UsdLux.MeshLightAPI)
        points, indices = get_mesh_points_and_indices(mesh_geom)
    elif prim.IsA(UsdLux.RectLight):
        rect_light = UsdLux.RectLight(prim)
        hx = rect_light.GetWidthAttr().Get() * 0.5
        hy = rect_light.GetHeightAttr().Get() * 0.5
        # vertices in local space (XY plane)
        transform = get_world_transform(prim)
        points = np.array(
            [
                transform.TransformAffine((-hx, -hy, 0.0)),  # bottom-left
                transform.TransformAffine((hx, -hy, 0.0)),  # bottom-right
                transform.TransformAffine((hx, hy, 0.0)),  # top-right
                transform.TransformAffine((-hx, hy, 0.0)),  # top-left
            ],
            dtype=np.float32,
        )
        assert points.shape[1] == 3
        # two triangles (indices into points)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        is_light = True
    else:
        return None

    wp_mesh = wp.Mesh(
        points=wp.array(np.ascontiguousarray(points), dtype=wp.vec3),
        velocities=None,
        indices=wp.array(np.ascontiguousarray(indices), dtype=wp.int32),
        bvh_constructor="sah",
    )

    assert len(indices) % 3 == 0

    mesh = Mesh()
    mesh.mesh_id = wp_mesh.id
    mesh.num_faces = len(indices) // 3
    mesh.is_light = is_light
    if is_light:
        mesh.light_intensity = prim.GetAttribute("inputs:intensity").Get()
        mesh.light_color = prim.GetAttribute("inputs:color").Get()

    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding = UsdShade.MaterialBindingAPI(prim)
        for binding in material_binding.GetDirectBindingRel().GetTargets():
            # only the first material is used
            material_path = binding.GetPrimPath()
            if material_path not in mat_prim_path_to_mat_id:
                raise ValueError(f"Material not found: {material_path}")
            mesh.material_id = wp.uint64(mat_prim_path_to_mat_id[material_path])
    else:
        if len(mat_prim_path_to_mat_id) == 0:
            raise ValueError(
                f"No materials found and the mesh {prim.GetPrimPath()} has no material binding"
            )
        logger.warning(
            f"No material binding found for {prim.GetPrimPath()}, using the material with the smallest id"
        )
        mesh.material_id = wp.uint64(min(mat_prim_path_to_mat_id.values()))

    return MeshCreationInfo(
        wp_mesh, mesh, (wp.vec3(np.min(points)), wp.vec3(np.max(points)))
    )
