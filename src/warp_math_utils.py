import warp as wp
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Misc.
# ----------------------------
@wp.func
def get_triangle_points(
    mesh_id: wp.uint64, face_idx: int
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    v1 = wp.mesh_get_point(mesh_id, 3 * face_idx)
    v2 = wp.mesh_get_point(mesh_id, 3 * face_idx + 1)
    v3 = wp.mesh_get_point(mesh_id, 3 * face_idx + 2)
    return v1, v2, v3


# ----------------------------
# MIS, Eric Veach Thesis
# ----------------------------
@wp.func
def power_heuristic(p1: float, p2: float):
    psum = p1 + p2
    return p1 * p1 / (psum * psum)


@wp.func
def solid_angle_to_area(p1: wp.vec3, p2: wp.vec3, n2: wp.vec3):
    """
    Computes dΩ/dA: the area on the surface (at p2 with normal n2)
    that corresponds to one unit solid angle as seen from p1.
    """
    p2p1 = p1 - p2
    return wp.abs(wp.dot(wp.normalize(p2p1), n2)) / wp.dot(p2p1, p2p1)


@wp.func
def area_to_solid_angle(p1: wp.vec3, p2: wp.vec3, n2: wp.vec3):
    """
    Computes dA/dΩ: the area on the surface (at p2 with normal n2)
    that corresponds to one unit solid angle as seen from p1.
    """
    p2p1 = p1 - p2
    return wp.dot(p2p1, p2p1) / wp.abs(wp.dot(wp.normalize(p2p1), n2))


# ---------------------------
# Concentric disk mapping, Shirley & Chiu’s concentric map (1997)
# ---------------------------
@wp.func
def concentric_disk_sample(rand_state: wp.uint32) -> wp.vec2:
    # map [0,1] -> [-1,1]
    u1 = wp.randf(rand_state)
    u2 = wp.randf(rand_state)

    sx = 2.0 * u1 - 1.0
    sy = 2.0 * u2 - 1.0

    if sx == 0.0 and sy == 0.0:
        return wp.vec2(0.0, 0.0)

    if wp.abs(sx) > wp.abs(sy):
        r = sx
        theta = (wp.pi / 4.0) * (sy / sx)
    else:
        r = sy
        theta = (wp.pi / 2.0) - (wp.pi / 4.0) * (sx / sy)

    return wp.vec2(r * wp.cos(theta), r * wp.sin(theta))


@wp.func
def get_coord_system(normal: wp.vec3):
    """
    Return an orthonormal basis for the given normal vector as a column-major matrix
    https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    """
    sign = wp.sign(normal.z)
    a = -1.0 / (sign + normal.z)
    b = normal.x * normal.y * a
    b1 = wp.vec3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x)
    b2 = wp.vec3(b, sign + normal.y * normal.y * a, -normal.y)
    return wp.matrix_from_cols(b1, b2, normal)


# ---------------------------
# Cosine-weighted hemisphere, Malley’s Method
# ---------------------------
@wp.func
def hemisphere_sample(rand_state: wp.uint32) -> wp.vec3:
    d = concentric_disk_sample(rand_state)
    x = d[0]
    y = d[1]
    z = wp.sqrt(wp.max(0.0, 1.0 - x * x - y * y))

    return wp.normalize(wp.vec3(x, y, z))


@wp.func
def hemisphere_pdf(w: wp.vec3):
    z = w.z
    return z / wp.pi if z > 0.0 else 0.0


@wp.kernel
def visualize_hemisphere(rand_seed: int, dirs: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    state = wp.rand_init(rand_seed + tid)
    dir = hemisphere_sample(state)
    dirs[tid] = dir


@wp.kernel
def visualize_triangle(
    rand_seed: int,
    points: wp.array(dtype=wp.vec3),
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
):
    tid = wp.tid()
    state = wp.rand_init(rand_seed + tid)
    point, _, __ = triangle_sample(state, v0, v1, v2)
    points[tid] = point


# ---------------------------
# Triangle sampling
# ---------------------------
@wp.func
def triangle_sample(
    rand_state: wp.uint32, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3
) -> tuple[wp.vec3, wp.vec2, float]:
    """
    Uniformly samples a triagnle bounded by v0, v1, and v2
    Returns the point sampled, the barycentric coordinate, and the probability
    """
    weights = triangle_sample_bary(rand_state)
    w1, w2 = weights[0], weights[1]

    # triangle area
    e1 = v1 - v0
    e2 = v2 - v0
    area = 0.5 * wp.length(wp.cross(e1, e2))

    return (
        w1 * v0 + w2 * v1 + (1.0 - w1 - w2) * v2,
        weights,
        1.0 / area if area > 0.0 else 0.0,
    )


@wp.func
def triangle_sample_bary(rand_state: wp.uint32) -> wp.vec2:
    """
    Returns a uniformly sampled barycentric coordinate
    """
    u = wp.randf(rand_state)
    v = wp.randf(rand_state)

    sqrt_u = wp.sqrt(u)
    w1 = 1.0 - sqrt_u
    w2 = sqrt_u - sqrt_u * v

    return wp.vec2(w1, w2)


# ---------------------------
# Visualization
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=5000, help="Number of samples."
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=["triangle", "hemisphere"],
        default="triangle",
        help="The target to visualize.",
    )
    args = parser.parse_known_args()[0]
    n_samples = args.num_samples
    target = args.target

    # 3D scatter
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if target == "hemisphere":
        # collect hemisphere samples
        dirs = wp.array(dtype=wp.vec3, shape=(n_samples,))
        with wp.ScopedTimer("sample hemisphere"):
            wp.launch(
                kernel=visualize_hemisphere,
                dim=n_samples,
                inputs=[0, dirs],
            )
        dirs = dirs.numpy()

        ax.scatter(dirs[:, 0], dirs[:, 1], dirs[:, 2], s=2, alpha=0.3)

        # hemisphere outline for reference
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi / 2, 20)
        X = np.outer(np.cos(u), np.sin(v))
        Y = np.outer(np.sin(u), np.sin(v))
        Z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(X, Y, Z, color="lightgray", alpha=0.3)

        ax.set_title("Cosine-weighted Hemisphere Sampling (3D)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])

    elif target == "triangle":
        # Define triangle vertices
        v0 = wp.vec3(0.0, 0.0, 0.0)
        v1 = wp.vec3(1.0, 0.0, 0.0)
        v2 = wp.vec3(0.5, 1.0, 0.0)

        # collect triangle samples
        points = wp.array(dtype=wp.vec3, shape=(n_samples,))
        with wp.ScopedTimer("sample triangle"):
            wp.launch(
                kernel=visualize_triangle,
                dim=n_samples,
                inputs=[0, points, v0, v1, v2],
            )
        points = points.numpy()

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.3)

        # triangle outline for reference
        triangle_vertices = np.array(
            [
                [v0[0], v0[1], v0[2]],
                [v1[0], v1[1], v1[2]],
                [v2[0], v2[1], v2[2]],
                [v0[0], v0[1], v0[2]],  # close the triangle
            ]
        )
        ax.plot(
            triangle_vertices[:, 0],
            triangle_vertices[:, 1],
            triangle_vertices[:, 2],
            color="red",
            linewidth=2,
            alpha=0.8,
        )

        ax.set_title("Uniform Triangle Sampling (3D)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])

    else:
        print(f"Unknown target: {target}. Valid choices: 'triangle' and 'hemisphere'")
        exit(1)

    plt.show()
