"""Validate all real-env scenarios by running them through their wrappers."""
import sys
import traceback
from dreambench.envs.base import load_scenarios
from dreambench.envs.registry import get_env_wrapper
from dreambench.runner import PROBE_REGISTRY
from pathlib import Path

ENVS = {
    "minigrid": "dreambench/envs/minigrid/scenarios.yaml",
    "crafter": "dreambench/envs/crafter/scenarios.yaml",
    "atari": "dreambench/envs/atari/scenarios.yaml",
}

valid_probes = set(PROBE_REGISTRY.keys())
failed = []

for env_name, yaml_path in ENVS.items():
    print(f"\n{'='*50}")
    print(f"Validating {env_name} ({yaml_path})")
    print(f"{'='*50}")

    try:
        scenarios = load_scenarios(Path(yaml_path))
    except Exception as e:
        print(f"  [FAIL] Could not load YAML: {e}")
        failed.append(f"{env_name}/yaml")
        continue

    wrapper = get_env_wrapper(env_name)

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario.name}")
        print(f"    env_id: {scenario.env_id}")
        print(f"    probe: {scenario.probe}")
        print(f"    actions: {len(scenario.actions)} steps")

        # Check probe name is valid
        if scenario.probe not in valid_probes:
            print(f"    [FAIL] Unknown probe '{scenario.probe}'. "
                  f"Valid: {valid_probes}")
            failed.append(f"{env_name}/{scenario.name}")
            continue

        # Actually run it
        try:
            traj = wrapper.run_ground_truth(scenario)
            print(f"    trajectory: {len(traj.observations)} frames")
            if any(traj.dones):
                done_frame = traj.dones.index(True)
                print(f"    episode ended at frame {done_frame}")
            print(f"    [PASS]")
        except Exception as e:
            print(f"    [FAIL] {e}")
            traceback.print_exc()
            failed.append(f"{env_name}/{scenario.name}")

if failed:
    print(f"\n\nFailed scenarios: {failed}")
    sys.exit(1)
else:
    print(f"\n\nAll scenarios validated successfully.")
