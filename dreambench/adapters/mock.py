"""Mock world model adapter for testing the benchmark pipeline."""

from typing import Tuple

import numpy as np

from dreambench.adapters.base import WorldModelAdapter


class MockAdapter(WorldModelAdapter):
    """Returns noisy copies of the initial observation. For pipeline testing only."""

    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
        self._state: np.ndarray | None = None
        self._step_count = 0

    def reset(self, initial_obs: np.ndarray) -> None:
        self._state = initial_obs.copy().astype(np.float32)
        self._step_count = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        self._step_count += 1
        noise = np.random.randn(*self._state.shape).astype(np.float32) * self.noise_std
        self._state = np.clip(self._state + noise, 0, 255)
        reward = float(np.random.random() * 0.1)  # small random reward
        done = False
        return self._state.astype(np.uint8), reward, done

    def get_latent(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self._state.flatten()
