from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import warp as wp

from material import Material

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

# Material parameter constraints (for physical validity / numerical stability)
BASE_COLOR_MIN = 0.0
BASE_COLOR_MAX = 1.0
METALLIC_MIN = 0.0
METALLIC_MAX = 1.0
EMISSIVE_COLOR_MIN = 0.0
EMISSIVE_COLOR_MAX = 1.0
EMISSIVE_INTENSITY_MIN = 0.0
ROUGHNESS_MIN = 1e-3  # > 0 to avoid div-by-zero / bad NDF behavior
ROUGHNESS_MAX = 1.0


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
    # Write
    loss: wp.array(dtype=wp.float32),
    per_pixel_loss: wp.array(dtype=wp.float32),
):
    """
    Calculates the L1 loss between the current image and the target image.
    """
    tid = wp.tid()

    predicted = radiance_sum[tid] / float(num_samples)
    target = target_image[tid]
    diff = predicted - target
    abs_diff = wp.abs(diff[0]) + wp.abs(diff[1]) + wp.abs(diff[2])
    per_pixel_loss[tid] = abs_diff
    wp.atomic_add(loss, 0, abs_diff)


@wp.kernel
def update_materials_adam(
    # Read
    lr: float,
    t: float,
    grads: wp.array(dtype=Material),
    # Read/Write
    materials: wp.array(dtype=Material),
    m_state: wp.array(dtype=Material),
    v_state: wp.array(dtype=Material),
):
    tid = wp.tid()
    mat = materials[tid]
    g = grads[tid]
    m = m_state[tid]
    v = v_state[tid]

    beta1 = ADAM_BETA1
    beta2 = ADAM_BETA2
    eps = ADAM_EPS

    # Bias correction denominators
    inv_bias1 = 1.0 / (1.0 - wp.pow(beta1, t))
    inv_bias2 = 1.0 / (1.0 - wp.pow(beta2, t))

    # --- base_color (vec3) ---
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
    mat.base_color = wp.vec3(
        wp.clamp(mat.base_color[0] - lr * step_c[0], BASE_COLOR_MIN, BASE_COLOR_MAX),
        wp.clamp(mat.base_color[1] - lr * step_c[1], BASE_COLOR_MIN, BASE_COLOR_MAX),
        wp.clamp(mat.base_color[2] - lr * step_c[2], BASE_COLOR_MIN, BASE_COLOR_MAX),
    )

    # --- roughness (scalar) ---
    m.roughness = beta1 * m.roughness + (1.0 - beta1) * g.roughness
    v.roughness = beta2 * v.roughness + (1.0 - beta2) * (g.roughness * g.roughness)
    m_hat_r = m.roughness * inv_bias1
    v_hat_r = v.roughness * inv_bias2
    mat.roughness = wp.clamp(
        mat.roughness - lr * m_hat_r / (wp.sqrt(v_hat_r) + eps),
        ROUGHNESS_MIN,
        ROUGHNESS_MAX,
    )

    # --- metallic (scalar) ---
    m.metallic = beta1 * m.metallic + (1.0 - beta1) * g.metallic
    v.metallic = beta2 * v.metallic + (1.0 - beta2) * (g.metallic * g.metallic)
    m_hat_m = m.metallic * inv_bias1
    v_hat_m = v.metallic * inv_bias2
    mat.metallic = wp.clamp(
        mat.metallic - lr * m_hat_m / (wp.sqrt(v_hat_m) + eps),
        METALLIC_MIN,
        METALLIC_MAX,
    )

    # --- emissive_color (vec3) ---
    m.emissive_color = beta1 * m.emissive_color + (1.0 - beta1) * g.emissive_color
    v.emissive_color = beta2 * v.emissive_color + (1.0 - beta2) * wp.cw_mul(
        g.emissive_color, g.emissive_color
    )
    m_hat_e = m.emissive_color * inv_bias1
    v_hat_e = v.emissive_color * inv_bias2
    step_e = wp.cw_div(
        m_hat_e,
        wp.vec3(
            wp.sqrt(v_hat_e[0]) + eps,
            wp.sqrt(v_hat_e[1]) + eps,
            wp.sqrt(v_hat_e[2]) + eps,
        ),
    )
    mat.emissive_color = wp.vec3(
        wp.clamp(
            mat.emissive_color[0] - lr * step_e[0],
            EMISSIVE_COLOR_MIN,
            EMISSIVE_COLOR_MAX,
        ),
        wp.clamp(
            mat.emissive_color[1] - lr * step_e[1],
            EMISSIVE_COLOR_MIN,
            EMISSIVE_COLOR_MAX,
        ),
        wp.clamp(
            mat.emissive_color[2] - lr * step_e[2],
            EMISSIVE_COLOR_MIN,
            EMISSIVE_COLOR_MAX,
        ),
    )

    # --- emissive_intensity (scalar) ---
    m.emissive_intensity = (
        beta1 * m.emissive_intensity + (1.0 - beta1) * g.emissive_intensity
    )
    v.emissive_intensity = beta2 * v.emissive_intensity + (1.0 - beta2) * (
        g.emissive_intensity * g.emissive_intensity
    )
    m_hat_i = m.emissive_intensity * inv_bias1
    v_hat_i = v.emissive_intensity * inv_bias2
    mat.emissive_intensity = wp.max(
        mat.emissive_intensity - lr * m_hat_i / (wp.sqrt(v_hat_i) + eps),
        EMISSIVE_INTENSITY_MIN,
    )

    # Opacity + IOR are currently not optimized; keep state in sync (optional).
    m.ior = 0.0
    v.ior = 0.0
    m.opacity = 0.0
    v.opacity = 0.0

    # Write back
    materials[tid] = mat
    m_state[tid] = m
    v_state[tid] = v


def compute_loss(
    num_samples: int,
    num_pixels: int,
    radiance_sum: wp.array(dtype=wp.vec3),
    target_image: wp.array(dtype=wp.vec3),
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
        max_epochs: int,
    ):
        self.renderer = renderer
        self.learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._epoch = 0

        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.per_pixel_loss = wp.zeros(
            renderer.width * renderer.height, dtype=wp.float32, requires_grad=True
        )

        if target_image.ndim != 1:
            raise ValueError(
                f"Unexpected target_image shape {target_image.shape}, expected float."
            )
        self.target_image = wp.array(
            target_image.reshape((-1, 3)),
            dtype=wp.vec3,
        )

        self.min_loss = float("inf")
        self.best_materials = wp.zeros_like(renderer.materials)
        self.initial_materials = wp.zeros_like(renderer.materials)
        self.renderer.materials.requires_grad = True
        wp.copy(self.best_materials, renderer.materials)
        wp.copy(self.initial_materials, renderer.materials)
        self.current_loss: Optional[float] = None
        self._finalized = False

        # optimizer state (Adam): reuse Material struct for momentum (m) and velocity (v)
        self._opt_m = wp.zeros_like(renderer.materials)
        self._opt_v = wp.zeros_like(renderer.materials)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    def step(self) -> Optional[float]:
        """
        Runs one learning epoch. Returns the loss as a float, or None if finished.
        An epoch here basically means one stochastic optimization step.
        """
        if self._epoch >= self._max_epochs:
            return None

        num_pixels = self.renderer.width * self.renderer.height

        # Reset the loss and per-pixel loss gradients and values
        self.per_pixel_loss.zero_()
        self.loss.zero_()
        self.renderer.reset()
        self.renderer.materials.grad.zero_()

        radiance_sum = wp.zeros(num_pixels, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        for _ in range(self.renderer.max_iter):
            _, replay_data = self.renderer.render(record_path_replay_data=True)
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
            compute_loss(
                self.renderer.max_iter,
                num_pixels,
                radiance_sum,
                self.target_image,
                self.per_pixel_loss,
                self.loss,
            )
            self.current_loss = float(self.loss.numpy()[0])

        tape.backward(self.loss)

        wp.launch(
            kernel=update_materials_adam,
            dim=len(self.renderer.materials),
            inputs=[
                self.learning_rate,
                float(self._epoch + 1),
                self.renderer.materials.grad,
                self.renderer.materials,
                self._opt_m,
                self._opt_v,
            ],
        )

        if self.loss_value < self.min_loss:
            self.min_loss = self.loss_value
            wp.copy(self.best_materials, self.renderer.materials)

        self._epoch += 1
        return self.loss_value

    @property
    def loss_value(self) -> float:
        if self.current_loss is None:
            raise RuntimeError("Loss not computed yet; call step() first.")
        return self.current_loss

    @property
    def per_pixel_mse_host(self) -> np.ndarray:
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

        delta = {}
        best_np = self.best_materials.numpy()
        initial_np = self.initial_materials.numpy()
        for name in best_np.dtype.names:
            delta[name] = best_np[name] - initial_np[name]

        logger.info("Min loss materials delta: %s", delta)
        wp.copy(self.renderer.materials, self.best_materials)
        self._finalized = True

    # Context manager helpers to ensure finalize is called.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        # propagate exceptions
        return False
