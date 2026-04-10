"""Quick debug script — check if tracker can find objects in env observations."""
import sys
import numpy as np
from pathlib import Path
from dreambench.envs.base import load_scenarios
from dreambench.envs.registry import get_env_wrapper
from dreambench.probes.utils.tracking import track_objects, detect_objects

env_name = sys.argv[1] if len(sys.argv) > 1 else "minigrid"
wrapper = get_env_wrapper(env_name)

yaml_path = Path(f"dreambench/envs/{env_name}/scenarios.yaml")
scenarios = load_scenarios(yaml_path)
scenario = scenarios[0]

print(f"Environment: {env_name}")
print(f"Scenario: {scenario.name}")

traj = wrapper.run_ground_truth(scenario)
print(f"Obs shape: {traj.observations[0].shape}")
print(f"Obs dtype: {traj.observations[0].dtype}")
print(f"Obs range: [{traj.observations[0].min()}, {traj.observations[0].max()}]")

bg = traj.observations[0]
for i, frame in enumerate(traj.observations[1:4]):
    dets = detect_objects(frame, bg_frame=bg, threshold=30)
    print(f"Frame {i+1}: {len(dets)} objects detected")
    for d in dets:
        print(f"  centroid=({d[0]:.1f}, {d[1]:.1f}), area={d[2]:.0f}")

# Also test full tracking
tracks = track_objects(traj.observations, threshold=30)
print(f"\nFull tracking: {len(tracks)} tracks found")
for i, t in enumerate(tracks[:5]):
    print(f"  Track {i}: {len(t)} points, "
          f"frames {t[0][0]}-{t[-1][0]}")
