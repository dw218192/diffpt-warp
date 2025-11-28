import warp as wp
from pxr import Usd
from warp_math_utils import (
    hemisphere_sample,
    hemisphere_pdf,
)
from mesh import Mesh

__all__ = [
    "MAT_ID_DIFFUSE",
    "MAT_ID_SPECULAR",
    "Material",
    "create_material_from_usd_prim",
    "mat_eval_bsdf",
    "mat_sample",
    "mat_pdf",
]

MAT_ID_DIFFUSE = 0
MAT_ID_SPECULAR = 1


@wp.struct
class Material:
    type: wp.uint16  # 0: diffuse, 1: specular
    color: wp.vec3


def create_material_from_usd_prim(prim: Usd.Prim):
    type = prim.GetAttribute("pbr:type").Get()
    color = prim.GetAttribute("pbr:color").Get()
    material = Material()

    if type == "diffuse":
        material.type = MAT_ID_DIFFUSE
    elif type == "specular":
        material.type = MAT_ID_SPECULAR
    else:
        raise ValueError(f"Unsupported material type: {type}")

    material.color = color
    return material


@wp.func
def mat_eval_bsdf(
    material: Material,
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> wp.vec3:
    """
    Evaluate the BRDF for a given material and an incoming and outgoing direction.
    Returns the BRDF value.
    """
    if material.type == MAT_ID_DIFFUSE:
        return material.color / wp.pi

    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def mat_sample(
    material: Material,
    rand_state: wp.uint32,
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> wp.vec3:
    """
    Sample a BRDF for a given material and an outgoing direction.
    Returns the sampled incoming direction.

    The definition of incoming and outgoing directions is consistent with PBRT.
    """

    if material.type == MAT_ID_DIFFUSE:
        return hemisphere_sample(rand_state)

    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def mat_pdf(
    material: Material,
    wi: wp.vec3,  # in normal space
    wo: wp.vec3,  # in normal space, towards 'sensor'
) -> float:
    """
    Evaluate the PDF of a sampled incoming direction.
    """

    if material.type == MAT_ID_DIFFUSE:
        return hemisphere_pdf(wi)

    return 0.0


# ---------------------------------------
# BRDF viewer
# ---------------------------------------
@wp.kernel
def sample_brdf(
    material: Material,
    wo: wp.vec3,  # in normal space, towards 'sensor'
    out_dirs: wp.array(dtype=wp.vec3),
    n_samples: int,
):
    """
    Sample a BRDF for a given material and a fixed outgoing direction.
    Returns the sampled directions.
    """
    tid = wp.tid()
    rand_state = wp.rand_init(tid)
    mat_sample(
        material,
        rand_state,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material-type", type=str, default="diffuse", choices=["diffuse", "specular"]
    )
    args = parser.parse_args()
