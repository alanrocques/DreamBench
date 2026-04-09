"""DreamerV3 world model adapter stub."""

from typing import Tuple

import numpy as np

from dreambench.adapters.base import WorldModelAdapter


class DreamerV3Adapter(WorldModelAdapter):
    """Adapter for DreamerV3 world model.

    This is a stub — implement load_checkpoint() and wire up the actual
    DreamerV3 RSSM for real benchmarking.
    """

    def __init__(self, checkpoint_path: str = "", latent_dim: int = 512):
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self._latent: np.ndarray | None = None
        self._obs_shape: tuple | None = None

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        """Load DreamerV3 checkpoint. Override with real implementation."""
        raise NotImplementedError(
            f"DreamerV3 checkpoint loading not yet implemented. "
            f"Subclass DreamerV3Adapter and implement _load_checkpoint() "
            f"to load from: {path}"
        )

    def reset(self, initial_obs: np.ndarray) -> None:
        self._obs_shape = initial_obs.shape
        # Stub: initialize latent as zeros
        self._latent = np.zeros(self.latent_dim, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._latent is None:
            raise RuntimeError("Must call reset() before step()")
        # Stub: return zeros
        obs = np.zeros(self._obs_shape, dtype=np.uint8)
        return obs, 0.0, False

    def get_latent(self) -> np.ndarray:
        if self._latent is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self._latent.copy()
