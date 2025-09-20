import warp as wp
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Concentric disk mapping, Shirley & Chiu’s concentric map (1997)
# ---------------------------
@wp.func
def concentric_disk_sample(u1: float, u2: float) -> wp.vec2:
    # map [0,1] -> [-1,1]
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
def hemisphere_sample(u1: float, u2: float) -> wp.vec3:
    d = concentric_disk_sample(u1, u2)
    x = d[0]
    y = d[1]
    z = wp.sqrt(wp.max(0.0, 1.0 - x * x - y * y))

    return wp.normalize(wp.vec3(x, y, z))


@wp.func
def hemisphere_pdf(dir: wp.vec3) -> float:
    cos_theta = dir.z
    return cos_theta / wp.pi if cos_theta > 0.0 else 0.0


@wp.kernel
def visualize_hemisphere(rand_seed: int, dirs: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    state = wp.rand_init(rand_seed + tid)
    u1 = wp.randf(state)
    u2 = wp.randf(state)
    dirs[tid] = hemisphere_sample(u1, u2)


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
    args = parser.parse_known_args()[0]
    n_samples = args.num_samples

    # collect samples
    dirs = wp.array(dtype=wp.vec3, shape=(n_samples,))
    with wp.ScopedTimer("sample hemisphere"):
        wp.launch(
            kernel=visualize_hemisphere,
            dim=n_samples,
            inputs=[0, dirs],
        )
    dirs = dirs.numpy()

    # 3D scatter
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
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

    plt.show()
