"""Smoke test: verify each real env wrapper produces valid trajectories."""
import sys
import traceback
import numpy as np
from dreambench.envs.base import Scenario, Trajectory

WRAPPER_TESTS = {
    "minigrid": {
        "module": "dreambench.envs.minigrid.wrapper",
        "class": "MiniGridEnvWrapper",
        "scenario": Scenario(
            name="smoke_minigrid",
            env_id="MiniGrid-DoorKey-5x5-v0",
            actions=[2, 2, 1, 2, 2],
            probe="object_permanence",
        ),
    },
    "crafter": {
        "module": "dreambench.envs.crafter.wrapper",
        "class": "CrafterEnvWrapper",
        "scenario": Scenario(
            name="smoke_crafter",
            env_id="crafter",
            actions=[0, 1, 2, 3, 5],
            probe="reward_fidelity",
        ),
    },
    "atari": {
        "module": "dreambench.envs.atari.wrapper",
        "class": "AtariEnvWrapper",
        "scenario": Scenario(
            name="smoke_atari",
            env_id="ALE/Breakout-v5",
            actions=[0, 0, 1, 1, 0],
            probe="reward_fidelity",
        ),
    },
}


def test_wrapper(name, config):
    import importlib
    mod = importlib.import_module(config["module"])
    cls = getattr(mod, config["class"])
    wrapper = cls()

    # Test run_ground_truth
    traj = wrapper.run_ground_truth(config["scenario"])
    assert isinstance(traj, Trajectory), f"Expected Trajectory, got {type(traj)}"

    n_actions = len(config["scenario"].actions)
    assert len(traj.observations) >= 2, (
        f"Expected >= 2 observations, got {len(traj.observations)}"
    )
    # observations = initial + one per action (may be fewer if done early)
    assert len(traj.rewards) == len(traj.observations) - 1, (
        f"rewards length {len(traj.rewards)} != observations length "
        f"{len(traj.observations)} - 1"
    )
    assert len(traj.dones) == len(traj.rewards), (
        f"dones length {len(traj.dones)} != rewards length {len(traj.rewards)}"
    )

    # Check observation format
    obs = traj.observations[0]
    assert isinstance(obs, np.ndarray), f"obs is {type(obs)}, expected ndarray"
    assert obs.ndim == 3, f"Expected 3D array (H,W,C), got shape {obs.shape}"
    assert obs.shape[2] == 3, f"Expected 3 channels (RGB), got {obs.shape[2]}"
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")

    # All frames same shape
    for i, o in enumerate(traj.observations):
        assert o.shape == obs.shape, (
            f"Frame {i} shape {o.shape} != frame 0 shape {obs.shape}"
        )

    # Rewards are numeric
    for r in traj.rewards:
        assert isinstance(r, (int, float, np.floating)), (
            f"Reward is {type(r)}, expected numeric"
        )

    print(f"  trajectory: {len(traj.observations)} frames, "
          f"{len(traj.rewards)} rewards")
    print(f"  reward range: [{min(traj.rewards):.2f}, {max(traj.rewards):.2f}]")
    print(f"  action_space_size: {wrapper.get_action_space_size()}")
    print(f"  [PASS]")


if __name__ == "__main__":
    failed = []
    for name, config in WRAPPER_TESTS.items():
        print(f"\nTesting {name}...")
        try:
            test_wrapper(name, config)
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed.append(name)

    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    else:
        print("\nAll environment wrappers passed.")
