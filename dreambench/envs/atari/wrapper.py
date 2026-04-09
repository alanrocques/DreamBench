"""Atari environment wrapper using Gymnasium ALE."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory


class AtariEnvWrapper(BaseEnvWrapper):
    """Runs ground-truth rollouts in Atari environments via Gymnasium."""

    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "Atari support requires gymnasium. "
                "Install with: pip install dreambench[atari]"
            )

        env = gym.make(scenario.env_id, render_mode="rgb_array")
        obs, info = env.reset()

        # Restore saved state if provided
        if scenario.initial_state_path:
            state_path = Path(scenario.initial_state_path)
            if state_path.exists():
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                env.unwrapped.restore_state(state)
                obs = env.unwrapped._get_obs()

        observations = [obs]
        rewards = []
        dones = []

        for action in scenario.actions:
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(terminated or truncated)
            if terminated or truncated:
                break

        env.close()
        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def get_action_space_size(self) -> int:
        return 18  # Maximum Atari action space
