"""Mock world model adapter for testing the benchmark pipeline."""

from typing import List, Optional, Tuple

import numpy as np

from dreambench.adapters.base import WorldModelAdapter


class MockAdapter(WorldModelAdapter):
    """Mock adapter that replays ground-truth observations with optional noise.

    When noise_std=0 (default), this is a "perfect" world model that returns
    exact copies of ground-truth frames and rewards — useful for validating
    that all probes score ~1.0 on faithful replay.

    When noise_std>0, adds Gaussian noise to observations and jitter to rewards,
    simulating an imperfect world model.
    """

    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std
        self._state: np.ndarray | None = None
        self._step_count = 0
        self._gt_observations: Optional[List[np.ndarray]] = None
        self._gt_rewards: Optional[List[float]] = None

    def set_ground_truth(
        self, observations: List[np.ndarray], rewards: List[float]
    ) -> None:
        """Provide ground-truth trajectory for faithful replay.

        Called by BenchmarkRunner before running the model rollout.
        """
        self._gt_observations = observations
        self._gt_rewards = rewards

    def reset(self, initial_obs: np.ndarray) -> None:
        self._state = initial_obs.copy().astype(np.float32)
        self._step_count = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        self._step_count += 1

        # Replay ground-truth if available
        if (
            self._gt_observations is not None
            and self._step_count < len(self._gt_observations)
        ):
            obs = self._gt_observations[self._step_count].copy().astype(np.float32)
        else:
            obs = self._state.copy()

        # Add noise
        if self.noise_std > 0:
            noise = np.random.randn(*obs.shape).astype(np.float32) * self.noise_std
            obs = np.clip(obs + noise, 0, 255)

        self._state = obs

        # Replay ground-truth reward if available
        reward_idx = self._step_count - 1
        if self._gt_rewards is not None and reward_idx < len(self._gt_rewards):
            reward = self._gt_rewards[reward_idx]
            if self.noise_std > 0:
                reward += float(np.random.randn() * self.noise_std * 0.01)
        else:
            reward = 0.0

        done = False
        return obs.astype(np.uint8), reward, done

    def get_latent(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self._state.flatten()
