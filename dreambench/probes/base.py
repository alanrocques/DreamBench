"""Abstract base class for diagnostic probes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class ProbeResult:
    """Result from running a probe on a single scenario."""

    probe_name: str
    scenario_name: str
    score: float  # Always in [0, 1], where 1 = perfect fidelity
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")


class BaseProbe(ABC):
    """Interface for diagnostic probes that detect specific failure modes."""

    name: str = "base"

    @abstractmethod
    def __call__(
        self,
        predicted_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        predicted_rewards: List[float],
        gt_rewards: List[float],
        metadata: Dict[str, Any],
    ) -> ProbeResult:
        """Compare world model rollout against ground truth.

        Args:
            predicted_frames: Observations from the world model.
            gt_frames: Observations from the real environment.
            predicted_rewards: Rewards predicted by the world model.
            gt_rewards: Rewards from the real environment.
            metadata: Scenario metadata (name, env_id, etc.).

        Returns:
            ProbeResult with a score in [0, 1] and diagnostic details.
        """
