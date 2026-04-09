"""Mock environment wrapper for testing the pipeline without real game installations."""

import numpy as np

from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory


class MockEnvWrapper(BaseEnvWrapper):
    """Generates synthetic trajectories for pipeline testing.

    Produces deterministic frame sequences with simple moving objects,
    allowing all probes to run without game dependencies.
    """

    def __init__(self, obs_shape: tuple = (84, 84, 3)):
        self.obs_shape = obs_shape

    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        rng = np.random.RandomState(hash(scenario.name) % 2**31)
        bg = rng.randint(0, 50, self.obs_shape, dtype=np.uint8)

        observations = [bg.copy()]
        rewards = []
        dones = []

        # Simulate a moving object (white square)
        obj_x, obj_y = 20, 20
        obj_size = 8

        for i, action in enumerate(scenario.actions):
            frame = bg.copy()

            # Move object based on action
            if action == 0:
                obj_y = max(0, obj_y - 2)
            elif action == 1:
                obj_y = min(self.obs_shape[0] - obj_size, obj_y + 2)
            elif action == 2:
                obj_x = max(0, obj_x - 2)
            elif action == 3:
                obj_x = min(self.obs_shape[1] - obj_size, obj_x + 2)

            # Draw object
            frame[obj_y:obj_y + obj_size, obj_x:obj_x + obj_size] = 255

            observations.append(frame)
            reward = 1.0 if (i + 1) % 5 == 0 else 0.0
            rewards.append(reward)
            dones.append(False)

        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def get_action_space_size(self) -> int:
        return 4
