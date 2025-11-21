import typing
from pxr import Gf, Usd, UsdGeom
import numpy as np

import logging

logger = logging.getLogger(__name__)


def decompose_world_transform(
    prim: Usd.Prim,
) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in world_transform.ExtractRotationMatrix())
    )
    return translation, rotation, scale


def extract_pos(prim: Usd.Prim) -> Gf.Vec3d:
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    return translation


def get_world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    return xform.ComputeLocalToWorldTransform(time)


def get_mesh_points_and_indices(
    mesh_geom: UsdGeom.Mesh,
) -> tuple[np.ndarray, np.ndarray]:
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


def compute_mesh_surface_area(
    points: np.ndarray[np.float32], indices: np.ndarray[np.int32]
) -> float:
    if len(points) == 0 or len(indices) == 0:
        logger.warning("Points or indices are empty")
        return 0.0
    elif len(indices) % 3 != 0:
        logger.warning("Num of indices is not a multiple of 3: %d", len(indices))
        return 0.0

    num_faces = indices.shape[0] // 3
    surface_area = 0.0
    for i in range(num_faces):
        i1 = indices[3 * i]
        i2 = indices[3 * i + 1]
        i3 = indices[3 * i + 2]
        v1 = points[i1]
        v2 = points[i2]
        v3 = points[i3]
        surface_area += 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    return surface_area
