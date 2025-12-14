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


@wp.kernel
def _compute_loss_kernel(
    # Read
    num_pixels: int,
    cur_image: wp.array(dtype=wp.vec3),
    target_image: wp.array(dtype=wp.vec3),
    # Write
    loss: wp.array(dtype=wp.float32),
    per_pixel_loss: wp.array(dtype=wp.float32),
):
    """
    Calculates the running average MSE loss between the current image and the target image.
    """
    tid = wp.tid()
    per_pixel_loss[tid] = wp.length_sq(cur_image[tid] - target_image[tid])
    wp.atomic_add(loss, 0, per_pixel_loss[tid] / float(num_pixels))


@wp.kernel
def update_materials(
    # Read
    lr: float,
    grads: wp.array(dtype=Material),
    # Read/Write
    materials: wp.array(dtype=Material),
):
    tid = wp.tid()
    material = materials[tid]
    grad = grads[tid]

    material.base_color -= lr * grad.base_color
    # TODO: make metallic differentiable
    # material.metallic -= lr * grad.metallic
    material.roughness -= lr * grad.roughness
    material.ior -= lr * grad.ior
    # material.emissive_color -= lr * grad.emissive_color
    # material.emissive_intensity -= lr * grad.emissive_intensity
    # material.opacity -= lr * grad.opacity
    materials[tid] = material


def compute_loss(
    cur_image: wp.array(dtype=wp.vec3),
    target_image: wp.array(dtype=wp.vec3),
    per_pixel_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    num_pixels: int,
):
    wp.launch(
        kernel=_compute_loss_kernel,
        dim=num_pixels,
        inputs=[
            num_pixels,
            cur_image,
            target_image,
        ],
        outputs=[per_pixel_loss, loss],
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
        batch_spp: int,
        max_epochs: int,
    ):
        self.renderer = renderer
        self.learning_rate = learning_rate
        self.batch_spp = batch_spp
        self._max_epochs = max_epochs
        self._epoch = 0

        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.per_pixel_loss = wp.zeros(
            renderer.width * renderer.height, dtype=wp.float32, requires_grad=True
        )

        if target_image.ndim == 1:
            reshaped = target_image.reshape((-1, 3))
        elif target_image.ndim == 3:
            reshaped = target_image.reshape((-1, 3))
        else:
            raise ValueError(
                f"Unexpected target_image shape {target_image.shape}, expected flat or HxWx3."
            )
        self.target_image = wp.array(
            reshaped,
            dtype=wp.vec3,
            requires_grad=True,
        )

        self.min_loss = float("inf")
        self.best_materials = wp.zeros_like(renderer.materials)
        self.initial_materials = wp.zeros_like(renderer.materials)
        wp.copy(self.best_materials, renderer.materials)
        wp.copy(self.initial_materials, renderer.materials)
        self.current_loss: Optional[float] = None
        self._finalized = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    def step(self) -> Optional[float]:
        """
        Runs one learning epoch. Returns the loss as a float, or None if finished.
        """
        if self._epoch >= self._max_epochs:
            return None

        num_pixels = self.renderer.width * self.renderer.height

        # Reset the loss and per-pixel loss gradients and values
        self.per_pixel_loss.zero_()
        self.loss.zero_()
        self.renderer.reset()

        for _ in range(self.batch_spp):
            self.renderer.render(record_path_replay_data=True)

        tape = wp.Tape()
        with tape:
            radiances = self.renderer.replay()
            compute_loss(
                radiances,
                self.target_image,
                self.per_pixel_loss,
                self.loss,
                num_pixels,
            )
            self.current_loss = float(self.loss.numpy()[0])

        tape.backward(self.loss)

        wp.launch(
            kernel=update_materials,
            dim=len(self.renderer.materials),
            inputs=[
                self.learning_rate,
                self.renderer.materials.grad,
                self.renderer.materials,
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
