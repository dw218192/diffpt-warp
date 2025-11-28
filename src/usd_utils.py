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
