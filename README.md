# DreamBench

A diagnostic benchmark for learned world models. Instead of a single reconstruction or reward score, DreamBench runs a suite of targeted probes that measure *where and how* a world model's imagination breaks.

## Why

Learned world models (DreamerV3, IRIS, DIAMOND, Δ-IRIS, etc.) are usually evaluated on downstream reward or pixel metrics (FID, LPIPS, SSIM). Those numbers hide the failures that actually matter for planning and RL:

- Low FID while routinely duplicating entities
- Plausible frames that violate object permanence
- Average-accurate reward prediction that hallucinates spurious rewards
- Physics that looks right until a specific interaction breaks it

DreamBench decomposes fidelity into five independently-scored failure categories and produces a per-probe breakdown for each model.

## Failure taxonomy

| Probe | What it tests |
|---|---|
| `object_permanence` | Entities persist correctly under occlusion or off-screen |
| `physics_consistency` | Gravity, collision, and momentum are preserved |
| `entity_integrity` | Objects keep their identity, count, and appearance over time |
| `reward_fidelity` | Predicted reward matches ground-truth game logic |
| `temporal_coherence` | Causal state (doors opened, items collected) survives long rollouts |

## Supported environments

- **Atari** via Gymnasium + ALE — Pong, Breakout, Space Invaders, Montezuma's Revenge, Freeway
- **MiniGrid** — DoorKey, MultiRoom, KeyCorridor, LavaCrossing
- **Crafter** — full survival environment
- **Mock** — toy environment for testing the harness without heavy deps

Each environment ships with hand-designed diagnostic scenarios (initial state + action sequence) in `dreambench/envs/<env>/scenarios.yaml`.

## Supported adapters

Adapters live in `dreambench/adapters/`:

- `mock.MockAdapter` — returns the ground-truth trajectory with configurable noise; use for sanity-checking probes
- `dreamerv3.DreamerV3Adapter`
- `iris.IRISAdapter`
- `delta_iris.DeltaIRISAdapter`
- `diamond.DIAMONDAdapter`

## Install

```bash
git clone <repo-url> dreambench
cd dreambench
pip install -e ".[all]"        # or pick extras: atari, minigrid, crafter, viz, cv
```

Python 3.9+.

## Quick start

Run the mock model against the mock env to verify the install:

```bash
python scripts/run_benchmark.py          # defaults: model=mock env=mock
```

Run DreamerV3 on Atari and write results under `results/atari/`:

```bash
python scripts/run_benchmark.py model=dreamerv3 env=atari
```

Run a single probe:

```bash
python scripts/run_benchmark.py model=dreamerv3 env=atari probes=[object_permanence]
```

Configuration is Hydra-based — see `dreambench/configs/` for model and env YAMLs. `scripts/run_all_envs.sh` runs a full sweep.

Generate an HTML report from a results directory:

```bash
python scripts/generate_report.py --results results/atari --output reports/atari.html
```

## Output

`run_benchmark.py` writes `results.json` with the overall score, per-probe means, per-scenario scores, and raw probe details. It also prints a summary:

```
============================================================
DreamBench Results: dreamerv3
============================================================
Overall Score: 0.742

Per-Probe Breakdown:
  object_permanence: 0.810 (std=0.064, n=6)
    ball_behind_bricks: 0.733
    ...
```

## Adding a new model

1. Subclass `WorldModelAdapter` in `dreambench/adapters/base.py` and implement `reset`, `step`, and `get_latent`.
2. Add a config at `dreambench/configs/model/<your_model>.yaml` pointing `adapter:` at your class.
3. Run `python scripts/run_benchmark.py model=<your_model> env=<env>`.

All existing probes and environments work automatically — the adapter is the only integration point.

## Adding scenarios

Append to the relevant `scenarios.yaml`:

```yaml
- name: ball_behind_bricks
  env: BreakoutNoFrameskip-v4
  initial_state: checkpoints/breakout_corner_pocket.pkl
  actions: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0]
  probe: object_permanence
  description: >
    Ball enters a gap in the brick wall. Tests whether the model
    maintains the ball's existence and trajectory while occluded.
```

Validate with `python scripts/validate_scenarios.py`.

## Repo layout

```
dreambench/
├── adapters/     world-model interfaces (mock, dreamerv3, iris, delta_iris, diamond)
├── envs/         env wrappers + scenarios.yaml per environment
├── probes/       the five failure detectors + shared utils
├── metrics/      aggregation and summarization
├── configs/      Hydra configs (model/, env/, config.yaml)
├── templates/    HTML report templates
└── runner.py     BenchmarkRunner: runs scenarios, dispatches probes
scripts/          run_benchmark.py, generate_report.py, validators, smoke tests
tests/            pytest suite
```

## Tests

```bash
pytest
```

## License

MIT
