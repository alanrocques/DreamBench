"""Abstract base class for world model adapters."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class WorldModelAdapter(ABC):
    """Interface that every world model must implement for benchmarking."""

    @abstractmethod
    def reset(self, initial_obs: np.ndarray) -> None:
        """Initialize the model's latent state from a real observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Predict next observation, reward, and done flag."""

    @abstractmethod
    def get_latent(self) -> np.ndarray:
        """Return current latent state (for probes that inspect it)."""
