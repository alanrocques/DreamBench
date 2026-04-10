# DreamBench: Real Environment Integration — Implementation Plan
# (Corrected after codebase review)

## Context

DreamBench benchmarks hallucination in learned world models. The pipeline
works end-to-end with mock env + mock adapter (all probes ~1.0 at noise_std=0).
Now we need to verify the three real environment wrappers (MiniGrid, Crafter,
Atari) produce valid ground-truth trajectories and that the mock adapter can
replay them faithfully through all probes.

**Goal:** `python scripts/run_benchmark.py model=mock env=minigrid model.noise_std=0`
(and crafter, atari) → all probes >= 0.95.

## How the pipeline works (read this first)

Understanding the flow prevents wasted effort:

1. `run_benchmark.py` loads adapter class from `cfg.model.adapter` and env
   wrapper class from `cfg.env.wrapper` via Hydra configs
2. `BenchmarkRunner.evaluate_scenario(scenario)` calls
   `self.env_wrapper.run_ground_truth(scenario)` to get a `Trajectory`
   (observations, rewards, dones) from the real environment
3. `BenchmarkRunner.run_model_rollout(scenario, gt_trajectory)` calls
   `self.adapter.set_ground_truth(gt_observations, gt_rewards)` then
   replays the action sequence through `self.adapter.step(action)`
4. The `MockAdapter` stores the GT data and replays it frame-by-frame
   (adding Gaussian noise if `noise_std > 0`)
5. Probes receive `(predicted_frames, gt_frames, predicted_rewards,
   gt_rewards, metadata)` and return a score in [0, 1]

**Key implication:** The MockAdapter does NOT need its own env instance.
It receives GT data from the runner via `set_ground_truth()` and replays it.
This means switching from mock env to real env requires ZERO adapter changes.
The only things that need to work are the env wrappers and their scenarios.

## What already exists and works

- `configs/env/minigrid.yaml`, `crafter.yaml`, `atari.yaml` — all present
  with correct `wrapper` dotted paths and `scenarios_path` values
- `dreambench/envs/registry.py` — auto-registers all four wrappers on import
- `dreambench/adapters/mock.py` — `MockAdapter` with `set_ground_truth()`
  support, works with any env wrapper automatically
- `dreambench/envs/minigrid/scenarios.yaml` — 4 scenarios
- `dreambench/envs/crafter/scenarios.yaml` — 4 scenarios
- `dreambench/envs/atari/scenarios.yaml` — 11 scenarios
- All 5 probes with consistent call signature

## What does NOT need to change

- `adapters/mock.py` — works as-is with any env via `set_ground_truth()`
- Hydra env configs — already created and correct
- `runner.py` — orchestration is env-agnostic
- `run_benchmark.py` — CLI already supports `env=minigrid` etc.

## Important details

- **noise_std:** `configs/model/mock.yaml` sets `noise_std: 0.1` by default.
  For calibration runs, override to 0: `model.noise_std=0` on the CLI.
  With noise_std=0.1, scores will be below 1.0 — that's expected, not a bug.
- **Non-determinism:** Scenarios don't specify seeds. MiniGrid and Atari
  wrappers don't pass a seed to `env.reset()`. This means real-env runs may
  produce slightly different trajectories each time (different random
  level layouts). This is fine for now — the mock adapter replays whatever
  GT the env produces, so probes should still score ~1.0. But scores may
  vary slightly between runs.
- **Crafter API:** Crafter uses its own API, not Gymnasium. `env.step()`
  returns `(obs, reward, done, info)` — 4 values, not 5. The wrapper
  already handles this correctly.
- **Observation format:** All three real env wrappers produce uint8 RGB
  numpy arrays. MiniGrid uses `env.render()` (rgb_array mode). Atari
  returns obs directly from `env.step()`. Crafter returns obs from
  `env.step()`. The MockAdapter casts to float32 internally then back
  to uint8, which can cause tiny rounding diffs even at noise_std=0
  (this is why entity_integrity scored 0.997 on mock env).

---

## Task sequence

### Task 1: Install environment dependencies

```bash
pip install minigrid gymnasium "gymnasium[atari]" ale-py crafter
pip install autorom
autorom --accept-license
```

Validate each import:
```bash
python -c "import minigrid; print('minigrid OK:', minigrid.__version__)"
python -c "import crafter; print('crafter OK')"
python -c "import ale_py; print('ale-py OK:', ale_py.__version__)"
python -c "import gymnasium; print('gymnasium OK:', gymnasium.__version__)"
```

If `autorom --accept-license` fails or ROMs aren't found, try:
```bash
pip install autorom[accept-rom-license]
```

Fix all import errors before proceeding.

---

### Task 2: Smoke test each env wrapper in isolation

Create `scripts/smoke_test_envs.py`. This must test wrappers through
`run_ground_truth(scenario)` — NOT through reset()/step(), because
the wrappers don't expose those methods. They only implement the
`BaseEnvWrapper` ABC which has `run_ground_truth()` and
`get_action_space_size()`.

```python
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
```

Run: `python scripts/smoke_test_envs.py`

**Common failures and fixes:**

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'minigrid'` | Package not installed | Go back to Task 1 |
| `gymnasium.error.NamespaceNotFound: ALE/Breakout-v5` | ROMs not installed | Run `autorom --accept-license` |
| `traj.rewards` length mismatch | Wrapper returns wrong number of rewards when episode ends early | Check the `if terminated or truncated: break` logic — rewards/dones must still be appended for that final step BEFORE breaking |
| `obs.ndim != 3` for MiniGrid | MiniGrid returning dict obs instead of RGB render | Verify `render_mode="rgb_array"` is passed to `gym.make()` and `env.render()` is called (the wrapper already does this — if it fails, check minigrid version) |
| `AttributeError` on Crafter | Crafter API changed between versions | Check `crafter.Env()` constructor and `step()` return values |

Fix wrappers one at a time: MiniGrid → Crafter → Atari.

---

### Task 3: Validate scenario YAMLs against real environments

Ensure every scenario in the real env YAML files can actually run. This
catches invalid env_ids, action values out of range, and bad
initial_state_path references.

Create `scripts/validate_scenarios.py`:

```python
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
```

Run: `python scripts/validate_scenarios.py`

**This will take a minute or two** because it runs all 19 real-env scenarios.
Atari scenarios are the slowest (ROM loading).

**What to watch for:**
- If a scenario's episode terminates before all actions are played, that's
  OK — the wrapper handles early termination. But if it terminates on
  frame 1 or 2, the action sequence is probably wrong for that env_id
  (e.g. needs a "start" action first).
  Fix the scenario YAML, not the wrapper.
- If MiniGrid scenarios produce very short trajectories (2-3 frames),
  the agent might be hitting a wall. The action sequences may need
  adjustment for the random level layout (non-determinism).
- Atari games like Breakout and Pong need action=1 (FIRE) to start
  the ball. Check that scenarios begin with a FIRE action.

---

### Task 4: Run the full benchmark — MiniGrid first

```bash
python scripts/run_benchmark.py model=mock env=minigrid model.noise_std=0
```

**Expected:** All probes >= 0.95.

**If a probe scores 0 or near 0:**

The problem is almost certainly in observation format, NOT probe logic.
Debug with this script:

```python
# Quick debug script — save as scripts/debug_probe.py
import numpy as np
from pathlib import Path
from dreambench.envs.base import load_scenarios
from dreambench.envs.registry import get_env_wrapper
from dreambench.probes.utils.tracking import track_objects, detect_objects

wrapper = get_env_wrapper("minigrid")
scenarios = load_scenarios(Path("dreambench/envs/minigrid/scenarios.yaml"))
scenario = scenarios[0]

traj = wrapper.run_ground_truth(scenario)
print(f"Obs shape: {traj.observations[0].shape}")
print(f"Obs dtype: {traj.observations[0].dtype}")
print(f"Obs range: [{traj.observations[0].min()}, {traj.observations[0].max()}]")

# Test if tracker can find anything
bg = traj.observations[0]
for i, frame in enumerate(traj.observations[1:4]):
    dets = detect_objects(frame, bg_frame=bg, threshold=30)
    print(f"Frame {i+1}: {len(dets)} objects detected")
    for d in dets:
        print(f"  centroid=({d[0]:.1f}, {d[1]:.1f}), area={d[2]:.0f}")
```

Run: `python scripts/debug_probe.py`

If 0 objects are detected, the `threshold` may be too high for MiniGrid's
rendered frames. Try lowering to 15 or 10. Do NOT change probe code — adjust
the scenario action sequence or add env-specific tracker config.

---

### Task 5: Run the full benchmark — Crafter

```bash
python scripts/run_benchmark.py model=mock env=crafter model.noise_std=0
```

**Expected:** All probes >= 0.95.

Crafter-specific notes:
- Crafter observations are 64x64 RGB by default. The tracker's
  `match_distance` (40px) and `teleport_distance` (50px) thresholds
  may be too large relative to the 64x64 frame — nearly the full image.
  This means the tracker might match objects that shouldn't match.
- Crafter is non-deterministic (procedural world generation). Each
  `run_ground_truth()` call produces a different world. The mock adapter
  replays the GT from that specific run, so probes should still score
  ~1.0.
- If a scenario terminates early (player dies), that's fine.

---

### Task 6: Run the full benchmark — Atari

```bash
python scripts/run_benchmark.py model=mock env=atari model.noise_std=0
```

**Expected:** All probes >= 0.95.

Atari-specific notes:
- **Slowest run** — 11 scenarios, each loading a ROM.
- Atari observations are typically 210x160 RGB.
- `ALE/Breakout-v5` and `ALE/Pong-v5` need action 1 (FIRE) to start.
  Check that scenarios begin with this action.
- `ALE/MontezumaRevenge-v5` is hard — the agent may die immediately
  with the given actions. If the scenario produces only 2-3 frames,
  the action sequence needs adjustment.
- The Atari wrapper has a `_get_obs()` fallback when loading saved
  states (`obs = env.unwrapped._get_obs()`). This line only executes
  when `initial_state_path` is set. No current scenarios use it, so
  it won't be hit. Don't worry about it.

---

### Task 7: Generate reports

```bash
python scripts/generate_report.py --results results/minigrid --output reports/minigrid.html
python scripts/generate_report.py --results results/crafter --output reports/crafter.html
python scripts/generate_report.py --results results/atari --output reports/atari.html
```

Open each HTML file and verify:
- Radar chart renders with all probe categories
- All scenario names are listed
- Scores are displayed correctly

---

### Task 8 (optional): Convenience script to run all envs

Create `scripts/run_all_envs.sh`:
```bash
#!/bin/bash
set -e

for env in minigrid crafter atari; do
    echo "========== $env =========="
    python scripts/run_benchmark.py model=mock env=$env model.noise_std=0
    python scripts/generate_report.py \
        --results "results/$env" \
        --output "reports/${env}_mock.html"
    echo ""
done

echo "All environments passed with mock adapter (noise_std=0)."
```

---

## Success criteria

All of these must be true:

1. `python scripts/smoke_test_envs.py` — all three wrappers pass
2. `python scripts/validate_scenarios.py` — all 19 scenarios run
3. `python scripts/run_benchmark.py model=mock env=minigrid model.noise_std=0` → overall >= 0.95
4. `python scripts/run_benchmark.py model=mock env=crafter model.noise_std=0` → overall >= 0.95
5. `python scripts/run_benchmark.py model=mock env=atari model.noise_std=0` → overall >= 0.95
6. HTML reports generate for all three environments without errors
7. No changes to probe code or runner.py — if probes score low, fix
   the env wrapper observation format or scenario action sequences

## Critical notes for Claude Code

- **Read `runner.py` first.** Understand that `run_model_rollout()` calls
  `adapter.set_ground_truth()` — this is why the mock adapter works with
  any env without modification.
- **Do NOT modify `adapters/mock.py`.** It already works with real envs.
- **Do NOT modify probe code.** If probes score 0, the issue is observation
  format — fix the wrapper or scenario, not the probe.
- **Always use `model.noise_std=0`** for calibration runs. The default
  config has `noise_std: 0.1` which adds noise and lowers scores.
- **Fix one env at a time.** Order: MiniGrid → Crafter → Atari.
- **MiniGrid render:** The wrapper calls `env.render()` to get RGB arrays.
  This requires `render_mode="rgb_array"` in `gym.make()`. The wrapper
  already does this. If it breaks, check minigrid version compatibility.
- **Crafter is NOT gymnasium.** It uses `crafter.Env()` which has a
  4-return `step()`: `(obs, reward, done, info)`. The wrapper handles
  this. Do not wrap it in gymnasium.
- **Atari ROMs:** Must be installed via `autorom --accept-license`.
  Without ROMs, Atari envs fail with a confusing file-not-found error.
- The Hydra `_self_` warning is cosmetic. Fix it by adding `_self_` to
  the end of the defaults list in `dreambench/configs/config.yaml` if
  convenient, but it's not blocking.