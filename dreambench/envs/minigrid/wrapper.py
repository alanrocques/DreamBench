"""MiniGrid environment wrapper using the minigrid library."""

import pickle
from pathlib import Path

import numpy as np

from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory


class MiniGridEnvWrapper(BaseEnvWrapper):
    """Runs ground-truth rollouts in MiniGrid environments.

    MiniGrid discrete actions:
        0 = turn left
        1 = turn right
        2 = move forward
        3 = pickup
        4 = drop
        5 = toggle (open door, interact)
        6 = done
    """

    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        try:
            import minigrid  # noqa: F401 — registers envs
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "MiniGrid support requires minigrid and gymnasium. "
                "Install with: pip install dreambench[minigrid]"
            )

        env = gym.make(scenario.env_id, render_mode="rgb_array")
        obs, info = env.reset()

        # Restore saved state if provided
        if scenario.initial_state_path:
            state_path = Path(scenario.initial_state_path)
            if state_path.exists():
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                env.unwrapped.load_state(state)
                obs = env.unwrapped.gen_obs()

        # MiniGrid obs is a dict with 'image', 'direction', 'mission'.
        # We capture the full RGB render as the observation for probes.
        initial_frame = env.render()
        observations = [initial_frame]
        rewards: list[float] = []
        dones: list[bool] = []

        for action in scenario.actions:
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            observations.append(frame)
            rewards.append(float(reward))
            dones.append(terminated or truncated)
            if terminated or truncated:
                break

        env.close()
        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def get_action_space_size(self) -> int:
        return 7  # MiniGrid has 7 discrete actions
