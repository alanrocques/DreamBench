"""Abstract base class for environment wrappers and scenario definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


@dataclass
class Trajectory:
    """A recorded sequence of observations, rewards, and done flags."""

    observations: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]


@dataclass
class Scenario:
    """A diagnostic test case: initial state + action sequence + expected probe."""

    name: str
    env_id: str
    actions: List[int]
    probe: str
    description: str = ""
    initial_state_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        return cls(
            name=data["name"],
            env_id=data["env_id"],
            actions=data["actions"],
            probe=data["probe"],
            description=data.get("description", ""),
            initial_state_path=data.get("initial_state", None),
            metadata=data.get("metadata", {}),
        )


def load_scenarios(yaml_path: Path) -> List[Scenario]:
    """Load scenarios from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [Scenario.from_dict(s) for s in data["scenarios"]]


class BaseEnvWrapper(ABC):
    """Interface for environment wrappers that run ground-truth rollouts."""

    @abstractmethod
    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        """Execute a scenario in the real environment and return the trajectory.

        Args:
            scenario: The diagnostic scenario to run.

        Returns:
            Trajectory with observations, rewards, and done flags.
        """

    @abstractmethod
    def get_action_space_size(self) -> int:
        """Return the number of discrete actions available."""
