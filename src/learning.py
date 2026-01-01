from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import warp as wp

from material import Material, GGX_MIN_ALPHA

if TYPE_CHECKING:
    from main import Renderer

logger = logging.getLogger(__name__)

__all__ = [
    "LearningSession",
]

# Optimizer hyperparameters / clamps
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

ALPHA_MIN = float(GGX_MIN_ALPHA)
ALPHA_MAX = 1.0

# Gradient clipping (global L2 norm) for stability under noisy/low-SPP gradients.
GRAD_CLIP_NORM = 10.0
GRAD_CLIP_EPS = 1e-8

HUBER_DELTA = 0.1

# Latent-space mapping safety.
# Keeping decode away from exact {0,1} reduces saturation (tiny sigmoid' gradients).
_LATENT_EPS = 1e-4


@wp.kernel
def _accumulate_radiance_kernel(
    # Read
    one_spp_radiances: wp.array(dtype=wp.vec3),
    # Write
    radiances_sum: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    radiances_sum[tid] += one_spp_radiances[tid]


@wp.kernel
def _compute_loss_kernel(
    # Read
    num_samples: int,
    num_pixels: int,
    radiance_sum: wp.array(dtype=wp.vec3),
    target_image: wp.array(dtype=wp.vec3),
    huber_delta: float,
    # Write
    loss: wp.array(dtype=wp.float32),
    per_pixel_loss: wp.array(dtype=wp.float32),
):
    """
    Calculates a per-pixel Huber loss between the current image and the target image.
    The loss is summed across RGB channels, then summed across pixels.
    """
    tid = wp.tid()

    predicted = radiance_sum[tid] / float(num_samples)
    target = target_image[tid]
    diff = predicted - target
    l = (
        pseudo_huber(diff[0], huber_delta)
        + pseudo_huber(diff[1], huber_delta)
        + pseudo_huber(diff[2], huber_delta)
    )
    per_pixel_loss[tid] = l
    wp.atomic_add(loss, 0, l)


@wp.kernel
def _gaussian_blur3x3(
    width: int,
    height: int,
    # Read
    src: wp.array(dtype=wp.vec3),
    # Write
    dst: wp.array(dtype=wp.vec3),
):
    """
    Lightweight 3x3 Gaussian blur with weights [[1,2,1],[2,4,2],[1,2,1]] / 16.
    """
    tid = wp.tid()
    x = tid % width
    y = tid // width

    acc = wp.vec3(0.0)
    w_sum = 0.0

    # y - 1
    if y > 0 and x > 0:
        w = 1.0
        acc += src[(y - 1) * width + (x - 1)] * w
        w_sum += w
    if y > 0:
        w = 2.0
        acc += src[(y - 1) * width + x] * w
        w_sum += w
    if y > 0 and x + 1 < width:
        w = 1.0
        acc += src[(y - 1) * width + (x + 1)] * w
        w_sum += w

    # y
    if x > 0:
        w = 2.0
        acc += src[y * width + (x - 1)] * w
        w_sum += w
    w = 4.0
    acc += src[tid] * w
    w_sum += w
    if x + 1 < width:
        w = 2.0
        acc += src[y * width + (x + 1)] * w
        w_sum += w

    # y + 1
    if y + 1 < height and x > 0:
        w = 1.0
        acc += src[(y + 1) * width + (x - 1)] * w
        w_sum += w
    if y + 1 < height:
        w = 2.0
        acc += src[(y + 1) * width + x] * w
        w_sum += w
    if y + 1 < height and x + 1 < width:
        w = 1.0
        acc += src[(y + 1) * width + (x + 1)] * w
        w_sum += w

    dst[tid] = acc / w_sum


@wp.func
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + wp.exp(-x))


@wp.func
def logit01(p: float, eps: float) -> float:
    # Clamp to avoid +/-inf at the boundaries.
    p = wp.min(wp.max(p, eps), 1.0 - eps)
    return wp.log(p) - wp.log(1.0 - p)


@wp.func
def bounded_sigmoid(x: float, eps: float) -> float:
    # Map R -> (eps, 1-eps)
    return eps + (1.0 - 2.0 * eps) * sigmoid(x)


@wp.func
def inv_bounded_sigmoid01(p: float, eps: float) -> float:
    # Invert bounded_sigmoid(), robust to mild out-of-range p.
    p = wp.min(wp.max(p, eps), 1.0 - eps)
    q = (p - eps) / (1.0 - 2.0 * eps)  # in [0,1]
    return logit01(q, eps)


@wp.func
def pseudo_huber(x: float, delta: float) -> float:
    """
    Smooth approximation to Huber / Smooth L1 (a.k.a. pseudo-Huber):

        delta^2 * (sqrt(1 + (x/delta)^2) - 1)

    - Quadratic near 0 (like L2)
    - Linear for large |x| (like L1)
    """
    # delta is expected > 0; treat as a hyperparameter.
    t = x / delta
    return delta * delta * (wp.sqrt(1.0 + t * t) - 1.0)


@wp.kernel
def encode_materials(
    # Read
    materials: wp.array(dtype=Material),
    # Write
    out_latent: wp.array(dtype=Material),
):
    i = wp.tid()
    m = materials[i]

    z = Material()

    # base_color: [0,1] -> logit
    z.base_color = wp.vec3(
        inv_bounded_sigmoid01(m.base_color[0], _LATENT_EPS),
        inv_bounded_sigmoid01(m.base_color[1], _LATENT_EPS),
        inv_bounded_sigmoid01(m.base_color[2], _LATENT_EPS),
    )

    # metallic: [0,1] -> logit
    z.metallic = inv_bounded_sigmoid01(m.metallic, _LATENT_EPS)

    # roughness: [roughness_min, roughness_max] -> logit of normalized
    # We optimize GGX alpha = roughness^2 (what the BSDF actually uses).
    # Clamp alpha to GGX_MIN_ALPHA to match the BSDF behavior and avoid non-identifiability.
    alpha = wp.max(m.roughness * m.roughness, float(ALPHA_MIN))
    inv_range = 1.0 / (ALPHA_MAX - ALPHA_MIN)
    an = (alpha - ALPHA_MIN) * inv_range
    z.roughness = logit01(an, _LATENT_EPS)

    out_latent[i] = z


@wp.kernel
def decode_materials(
    # Read
    latent_mats: wp.array(dtype=Material),
    static_mats: wp.array(dtype=Material),
    # Write
    out_mats: wp.array(dtype=Material),
):
    i = wp.tid()
    # Start from the "static" (non-learned) material so we preserve fields like
    # emissive_color/emissive_intensity/ior/opacity that affect rendering but
    # are not currently optimized in latent space.
    m = static_mats[i]

    # base_color: [0,1]
    m.base_color = wp.vec3(
        bounded_sigmoid(latent_mats[i].base_color[0], _LATENT_EPS),
        bounded_sigmoid(latent_mats[i].base_color[1], _LATENT_EPS),
        bounded_sigmoid(latent_mats[i].base_color[2], _LATENT_EPS),
    )

    # roughness via GGX alpha = roughness^2:
    # alpha in [ALPHA_MIN, ALPHA_MAX], then roughness = sqrt(alpha).
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * sigmoid(latent_mats[i].roughness)
    m.roughness = wp.sqrt(alpha)

    # metallic: [0,1]
    m.metallic = bounded_sigmoid(latent_mats[i].metallic, _LATENT_EPS)

    out_mats[i] = m


@wp.kernel
def decode_latent_grads(
    # Read
    latent_mats: wp.array(dtype=Material),
    decoded_grads: wp.array(dtype=Material),  # dL/d(decoded_materials)
    # Write
    out_latent_grads: wp.array(dtype=Material),  # dL/d(latent_materials)
):
    """
    Manual VJP for `decode_materials`.

    Warp currently does not propagate gradients through our struct-to-struct decode kernel
    (decoded_materials.grad is non-zero, but latent_materials.grad stays zero).
    This kernel explicitly maps dL/d(decoded_materials) -> dL/d(latent_materials).
    """
    i = wp.tid()
    z = latent_mats[i]
    g = decoded_grads[i]

    gz = Material()

    # base_color: bounded_sigmoid(z, eps)
    sx = sigmoid(z.base_color[0])
    sy = sigmoid(z.base_color[1])
    sz = sigmoid(z.base_color[2])
    scale = 1.0 - 2.0 * _LATENT_EPS
    gz.base_color = wp.vec3(
        g.base_color[0] * scale * sx * (1.0 - sx),
        g.base_color[1] * scale * sy * (1.0 - sy),
        g.base_color[2] * scale * sz * (1.0 - sz),
    )

    # metallic: bounded_sigmoid(z, eps)
    sm = sigmoid(z.metallic)
    gz.metallic = g.metallic * scale * sm * (1.0 - sm)

    # roughness via alpha = ALPHA_MIN + range * sigmoid(z), roughness = sqrt(alpha)
    # drough/dz = (0.5 / sqrt(alpha)) * range * sigmoid(z) * (1 - sigmoid(z))
    sr = sigmoid(z.roughness)
    a_range = ALPHA_MAX - ALPHA_MIN
    alpha = ALPHA_MIN + a_range * sr
    gz.roughness = g.roughness * (0.5 / wp.sqrt(alpha)) * a_range * sr * (1.0 - sr)

    out_latent_grads[i] = gz


@wp.kernel
def update_latent_materials_adam(
    # Read
    lr: float,
    t: float,
    grads: wp.array(dtype=Material),
    # Read/Write
    latent_materials: wp.array(dtype=Material),
    m_state: wp.array(dtype=Material),
    v_state: wp.array(dtype=Material),
):
    tid = wp.tid()
    z = latent_materials[tid]
    g = grads[tid]
    m = m_state[tid]
    v = v_state[tid]

    beta1 = ADAM_BETA1
    beta2 = ADAM_BETA2
    eps = ADAM_EPS

    # Bias correction denominators
    inv_bias1 = 1.0 / (1.0 - wp.pow(beta1, t))
    inv_bias2 = 1.0 / (1.0 - wp.pow(beta2, t))

    # --- base_color (vec3 latent; unconstrained) ---
    m.base_color = beta1 * m.base_color + (1.0 - beta1) * g.base_color
    v.base_color = beta2 * v.base_color + (1.0 - beta2) * wp.cw_mul(
        g.base_color, g.base_color
    )
    m_hat_c = m.base_color * inv_bias1
    v_hat_c = v.base_color * inv_bias2
    step_c = wp.cw_div(
        m_hat_c,
        wp.vec3(
            wp.sqrt(v_hat_c[0]) + eps,
            wp.sqrt(v_hat_c[1]) + eps,
            wp.sqrt(v_hat_c[2]) + eps,
        ),
    )
    z.base_color = z.base_color - lr * step_c

    # --- roughness (scalar latent; unconstrained) ---
    m.roughness = beta1 * m.roughness + (1.0 - beta1) * g.roughness
    v.roughness = beta2 * v.roughness + (1.0 - beta2) * (g.roughness * g.roughness)
    m_hat_r = m.roughness * inv_bias1
    v_hat_r = v.roughness * inv_bias2
    z.roughness = z.roughness - lr * m_hat_r / (wp.sqrt(v_hat_r) + eps)

    # --- metallic (scalar latent; unconstrained) ---
    m.metallic = beta1 * m.metallic + (1.0 - beta1) * g.metallic
    v.metallic = beta2 * v.metallic + (1.0 - beta2) * (g.metallic * g.metallic)
    m_hat_m = m.metallic * inv_bias1
    v_hat_m = v.metallic * inv_bias2
    z.metallic = z.metallic - lr * m_hat_m / (wp.sqrt(v_hat_m) + eps)

    # Not optimized in latent mode; keep Adam buffers inert.
    m.emissive_color = wp.vec3(0.0)
    v.emissive_color = wp.vec3(0.0)
    m.emissive_intensity = 0.0
    v.emissive_intensity = 0.0
    m.ior = 0.0
    v.ior = 0.0
    m.opacity = 0.0
    v.opacity = 0.0

    latent_materials[tid] = z
    m_state[tid] = m
    v_state[tid] = v


def compute_loss(
    num_samples: int,
    num_pixels: int,
    radiance_sum: wp.array(dtype=wp.vec3),
    target_image: wp.array(dtype=wp.vec3),
    huber_delta: float,
    per_pixel_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    wp.launch(
        kernel=_compute_loss_kernel,
        dim=num_pixels,
        inputs=[
            num_samples,
            num_pixels,
            radiance_sum,
            target_image,
            huber_delta,
        ],
        outputs=[
            loss,
            per_pixel_loss,
        ],
    )


class LearningSession:
    """
    Stateful differentiable rendering/learning loop.
    Mirrors the rendering API: call `step()` until it returns None.
    """

    def __init__(
        self,
        renderer: Renderer,
        *,
        target_image: np.ndarray,
        learning_rate: float,
        learning_rate_final: Optional[float] = None,
        max_epochs: int,
        rng_seed: int = 0,
        resample_interval: int = 0,
        min_spp: int = 1,
        max_spp: Optional[int] = None,
    ):
        self.renderer = renderer
        self._lr_start = learning_rate
        if self._lr_start <= 0.0:
            raise ValueError(f"learning_rate must be > 0; got {self._lr_start}")
        lr_final = learning_rate_final
        if lr_final is None:
            lr_final = 0.1 * self._lr_start
        if lr_final <= 0.0:
            raise ValueError(
                f"learning_rate_final must be > 0; got {lr_final} (before clamping)"
            )
        # Allow either direction (most cases decrease); scheduler interpolates between endpoints.
        self._lr_final = lr_final
        self.learning_rate = self._lr_start  # mutable, updated by scheduler

        # Samples-per-pixel schedule bounds.
        spp_cap = max_spp if max_spp is not None else renderer.max_iter
        if spp_cap <= 0:
            raise ValueError(f"max_spp must be > 0; got {spp_cap}")
        self._train_spp_max = spp_cap
        self._train_spp_min = max(1, min(min_spp, self._train_spp_max))
        self._current_spp = self._train_spp_min

        self._max_epochs = max_epochs
        self._epoch = 0
        self._rng_seed = rng_seed
        if self._rng_seed < 0:
            raise ValueError(f"rng_seed must be >= 0; got {self._rng_seed}")
        self._resample_interval = resample_interval
        if self._resample_interval < 0:
            raise ValueError(
                f"resample_interval must be >= 0 (0 disables reseeding); got {self._resample_interval}"
            )

        self._base_materials = renderer.materials

        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.per_pixel_loss = wp.zeros(
            renderer.width * renderer.height, dtype=wp.float32, requires_grad=True
        )

        if target_image.ndim != 1:
            raise ValueError(
                f"Unexpected target_image shape {target_image.shape}; expected a flat (N,) float array."
            )
        expected_len = renderer.width * renderer.height * 3
        if target_image.shape[0] != expected_len:
            raise ValueError(
                f"Unexpected target_image length {target_image.shape[0]}; expected {expected_len} (=W*H*3)."
            )
        self.target_image = wp.array(
            target_image.reshape((-1, 3)),
            dtype=wp.vec3,
        )
        self._blurred_target = wp.zeros_like(self.target_image)
        wp.launch(
            kernel=_gaussian_blur3x3,
            dim=renderer.width * renderer.height,
            inputs=[
                renderer.width,
                renderer.height,
                self.target_image,
            ],
            outputs=[self._blurred_target],
        )

        self.min_loss = float("inf")
        self.best_materials = wp.zeros_like(self._base_materials)
        self.initial_materials = wp.zeros_like(self._base_materials)
        self.current_loss: Optional[float] = None
        self._finalized = False
        wp.copy(self.initial_materials, self._base_materials)

        # Latent parameterization: optimize an unconstrained buffer, decode -> renderer.materials.
        self.latent_materials = wp.zeros_like(self._base_materials, requires_grad=True)
        wp.launch(
            kernel=encode_materials,
            dim=len(self._base_materials),
            inputs=[
                self._base_materials,
            ],
            outputs=[self.latent_materials],
        )
        self._opt_m = wp.zeros_like(self.latent_materials)
        self._opt_v = wp.zeros_like(self.latent_materials)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def current_lr(self) -> float:
        return float(self.learning_rate)

    @property
    def current_spp(self) -> int:
        return int(self._current_spp)

    def _schedule_epoch(self) -> tuple[float, int]:
        """
        Smoothly decay LR while ramping SPP:
        - start with higher LR and few SPP to chase low frequencies fast
        - end with lower LR and more SPP for fine details
        """
        # progress in [0,1]
        denom = max(1, self._max_epochs - 1)
        progress = min(1.0, self._epoch / denom)

        # Ease-out LR decay (quadratic) for stability near the end.
        lr = self._lr_final + (self._lr_start - self._lr_final) * (1.0 - progress) ** 2

        # Ease-in SPP ramp (quadratic) to spend more effort later.
        spp = int(
            round(
                self._train_spp_min
                + (self._train_spp_max - self._train_spp_min) * (progress**2)
            )
        )
        spp = max(self._train_spp_min, min(self._train_spp_max, spp))
        return lr, spp

    def step(self) -> Optional[float]:
        """
        Runs one learning epoch. Returns the loss as a float, or None if finished.
        An epoch here basically means one stochastic optimization step.
        """
        if self._epoch >= self._max_epochs:
            return None

        # Update LR/SPP schedule for this epoch.
        self.learning_rate, self._current_spp = self._schedule_epoch()
        self.renderer.max_iter = self._current_spp

        num_pixels = self.renderer.width * self.renderer.height

        decoded_materials_tape = wp.zeros_like(
            self._base_materials, requires_grad=True
        )
        decoded_materials_render = wp.zeros_like(self._base_materials)

        prev_renderer_materials = self.renderer.materials

        # Reset the loss and per-pixel loss gradients and values
        self.per_pixel_loss.zero_()
        self.loss.zero_()
        self.latent_materials.grad.zero_()

        radiance_sum = wp.zeros(num_pixels, dtype=wp.vec3, requires_grad=True)
        blurred_radiance_sum = wp.zeros_like(radiance_sum, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=decode_materials,
                dim=len(self._base_materials),
                inputs=[self.latent_materials, self._base_materials],
                outputs=[decoded_materials_tape],
            )

            # For some reason, Warp does not propagate adjoints through this struct->struct decode
            # so we implement JVP manually here.
            def _decode_backward():
                wp.launch(
                    kernel=decode_latent_grads,
                    dim=len(self.latent_materials),
                    inputs=[
                        self.latent_materials,
                        decoded_materials_tape.grad,
                    ],
                    outputs=[self.latent_materials.grad],
                )

            tape.record_func(
                _decode_backward, [self.latent_materials, decoded_materials_tape]
            )

        # For primal (path-recording) rendering, just copy tape decode to a render buffer.
        wp.copy(decoded_materials_render, decoded_materials_tape)

        # Seed schedule:
        # - Across epochs, optionally "resample" every `resample_interval` epochs to avoid overfitting a
        #   single fixed Monte Carlo sample set (which can cause plateaus).
        if self._resample_interval == 0:
            seed_epoch = self._rng_seed
        else:
            seed_epoch = self._rng_seed + (self._epoch // self._resample_interval)

        # Primal rendering uses internal accumulation buffers; reset at the start of the epoch.
        self.renderer.reset()
        self.renderer.rng_seed = seed_epoch

        for _ in range(self.renderer.max_iter):
            # Primal rendering
            self.renderer.materials = decoded_materials_render
            _, replay_data = self.renderer.render(
                record_path_replay_data=True, print_info=False
            )

            # Replay rendering
            self.renderer.materials = decoded_materials_tape
            with tape:
                one_spp_radiances = self.renderer.replay(replay_data)
                wp.launch(
                    kernel=_accumulate_radiance_kernel,
                    dim=num_pixels,
                    inputs=[
                        one_spp_radiances,
                    ],
                    outputs=[
                        radiance_sum,
                    ],
                )

        with tape:
            wp.launch(
                kernel=_gaussian_blur3x3,
                dim=num_pixels,
                inputs=[
                    self.renderer.width,
                    self.renderer.height,
                    radiance_sum,
                ],
                outputs=[blurred_radiance_sum],
            )
            compute_loss(
                self.renderer.max_iter,
                num_pixels,
                blurred_radiance_sum,
                self._blurred_target,
                HUBER_DELTA,
                self.per_pixel_loss,
                self.loss,
            )
            self.current_loss = float(self.loss.numpy()[0])

        # Track best params aligned with this loss (before optimizer step).
        if self.loss_value < self.min_loss:
            self.min_loss = self.loss_value
            wp.copy(self.best_materials, decoded_materials_render)

        tape.backward(self.loss)

        # --- Gradient clipping (global norm) ---
        if GRAD_CLIP_NORM > 0.0:
            g_np = self.latent_materials.grad.numpy()
            norm2 = 0.0
            for name in g_np.dtype.names:
                arr = np.asarray(g_np[name], dtype=np.float64)
                norm2 += float(np.sum(arr * arr))
            norm = float(np.sqrt(norm2))
            if norm > GRAD_CLIP_NORM:
                scale = float(GRAD_CLIP_NORM / (norm + GRAD_CLIP_EPS))
                for name in g_np.dtype.names:
                    g_np[name] *= scale
                wp.copy(self.latent_materials.grad, wp.array(g_np, dtype=Material))

        wp.launch(
            kernel=update_latent_materials_adam,
            dim=len(self.latent_materials),
            inputs=[
                self.learning_rate,
                float(self._epoch + 1),
                self.latent_materials.grad,
                self.latent_materials,
                self._opt_m,
                self._opt_v,
            ],
        )

        self.renderer.materials = prev_renderer_materials

        self._epoch += 1
        return self.loss_value

    @property
    def loss_value(self) -> float:
        if self.current_loss is None:
            raise RuntimeError("Loss not computed yet; call step() first.")
        return self.current_loss

    @property
    def per_pixel_loss_host(self) -> np.ndarray:
        return self.per_pixel_loss.numpy().reshape(
            (self.renderer.height, self.renderer.width, 1)
        )

    def finalize(self):
        """
        Applies the best-found materials and logs the delta relative to the initial state.
        """
        if self._finalized:
            return
        if self.min_loss < float("inf"):
            logger.info("Min loss: %f", self.min_loss)
        else:
            logger.info("No learning steps were executed.")

        best_np = self.best_materials.numpy()
        initial_np = self.initial_materials.numpy()

        def _fmt_val(v):
            return np.asarray(v).tolist()

        lines = ["Materials summary (before -> after | delta):"]
        for mat_id in range(len(best_np)):
            mesh_name = self.renderer.get_mesh_name_from_material_id(mat_id)
            lines.append(f"  [{mat_id}] mesh='{mesh_name}'")
            for field in best_np.dtype.names:
                before = _fmt_val(initial_np[field][mat_id])
                after = _fmt_val(best_np[field][mat_id])
                delta = _fmt_val(best_np[field][mat_id] - initial_np[field][mat_id])
                lines.append(f"    {field}: {before} -> {after} | Δ={delta}")
        logger.info("\n".join(lines))

        # Restore renderer SPP to the training maximum; caller may override for final renders.
        self.renderer.max_iter = self._train_spp_max
        # Apply best decoded materials back onto the renderer's base material buffer,
        # then restore the renderer to use that buffer for non-learning rendering.
        wp.copy(self._base_materials, self.best_materials)
        self.renderer.materials = self._base_materials
        self._finalized = True

    # Context manager helpers to ensure finalize is called.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        # propagate exceptions
        return False
