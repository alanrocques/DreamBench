"""Crafter environment wrapper."""

import numpy as np

from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory


class CrafterEnvWrapper(BaseEnvWrapper):
    """Runs ground-truth rollouts in the Crafter environment.

    Crafter has 17 discrete actions:
        0=noop, 1=move_left, 2=move_right, 3=move_up, 4=move_down,
        5=do, 6=sleep, 7=place_stone, 8=place_table, 9=place_furnace,
        10=place_plant, 11=make_wood_pickaxe, 12=make_stone_pickaxe,
        13=make_iron_pickaxe, 14=make_wood_sword, 15=make_stone_sword,
        16=make_iron_sword
    """

    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        try:
            import crafter
        except ImportError:
            raise ImportError(
                "Crafter support requires the crafter package. "
                "Install with: pip install dreambench[crafter]"
            )

        # Crafter uses its own env API (not gymnasium).
        # crafter.Env returns observations as RGB arrays directly.
        env = crafter.Env()
        obs = env.reset()

        observations = [obs]
        rewards: list[float] = []
        dones: list[bool] = []

        for action in scenario.actions:
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            if done:
                break

        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def get_action_space_size(self) -> int:
        return 17  # Crafter has 17 discrete actions
