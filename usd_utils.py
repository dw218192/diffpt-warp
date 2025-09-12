import typing
from pxr import Gf, Usd, UsdGeom
import numpy as np
import warp as wp


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


def usdmesh_to_wpmesh(mesh_geom: UsdGeom.Mesh) -> wp.Mesh:
    transform = get_world_transform(mesh_geom.GetPrim())
    points = mesh_geom.GetPointsAttr().Get()
    points = np.array([transform.TransformAffine(p) for p in points], dtype=np.float32)
    counts = np.array(mesh_geom.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    # triangulate faces
    tri_indices = []
    cursor = 0
    for n in counts:
        face = indices[cursor : cursor + n]
        cursor += n
        # fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
        for i in range(1, n - 1):
            tri_indices.extend([face[0], face[i], face[i + 1]])

    tri_indices = np.array(tri_indices, dtype=np.int32)

    return wp.Mesh(
        points=wp.array(points, dtype=wp.vec3),
        velocities=None,
        indices=wp.array(tri_indices, dtype=int),
    )
