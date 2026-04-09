"""IRIS world model adapter stub."""

from typing import Tuple

import numpy as np

from dreambench.adapters.base import WorldModelAdapter


class IRISAdapter(WorldModelAdapter):
    """Adapter for IRIS (Imagination with auto-Regression over an Inner Speech) model.

    This is a stub — implement load_checkpoint() and wire up the actual
    IRIS transformer for real benchmarking.
    """

    def __init__(self, checkpoint_path: str = "", latent_dim: int = 256):
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self._latent: np.ndarray | None = None
        self._obs_shape: tuple | None = None

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        raise NotImplementedError(
            f"IRIS checkpoint loading not yet implemented. "
            f"Subclass IRISAdapter and implement _load_checkpoint() "
            f"to load from: {path}"
        )

    def reset(self, initial_obs: np.ndarray) -> None:
        self._obs_shape = initial_obs.shape
        self._latent = np.zeros(self.latent_dim, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._latent is None:
            raise RuntimeError("Must call reset() before step()")
        obs = np.zeros(self._obs_shape, dtype=np.uint8)
        return obs, 0.0, False

    def get_latent(self) -> np.ndarray:
        if self._latent is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self._latent.copy()
