# DreamBench 🧪

**A systematic benchmark for measuring hallucination in learned world models for game environments.**

World models are increasingly used to train RL agents in "imagination" — but how faithful is that imagination? DreamBench provides a standardized suite of diagnostic tests that measure exactly *where and how* world models break, across architectures and environments.

---

## Motivation

Learned world models (DreamerV3, IRIS, DIAMOND, Genie, GameNGen, etc.) are evaluated almost exclusively on **downstream task reward** or **pixel-level reconstruction metrics** (FID, LPIPS, SSIM). These aggregate scores hide critical failure modes:

- A model can achieve low FID while routinely **duplicating entities**
- A model can produce plausible-looking frames while violating **object permanence**
- Reward prediction can be accurate on average while **hallucinating spurious rewards**
- Physics can look right most of the time while failing on **edge-case interactions**

DreamBench decomposes world model fidelity into **interpretable, independently measurable failure categories** — giving researchers a diagnostic report, not just a number.

---

## Failure Taxonomy

DreamBench organizes world model failures into five categories:

| Category | What it tests | Example failure |
|---|---|---|
| **Object permanence** | Do entities persist correctly when occluded or off-screen? | An enemy disappears when behind a wall, then reappears as a different entity |
| **Physics consistency** | Are physical rules (gravity, collision, momentum) maintained? | A thrown object changes trajectory mid-flight with no interaction |
| **Entity integrity** | Do objects maintain identity, count, and appearance over time? | Two coins merge into one, or one enemy splits into two |
| **Reward fidelity** | Does the predicted reward match ground-truth game logic? | Model predicts reward for collecting a coin that was already collected |
| **Temporal coherence** | Are causal sequences maintained over long rollouts? | A door that was opened reverts to closed 20 frames later |

---

## Supported Environments

DreamBench ships with test suites for:

- **Atari 2600** (via ALE) — Pong, Breakout, Space Invaders, Montezuma's Revenge, Freeway
- **MiniGrid** — DoorKey, MultiRoom, KeyCorridor, LavaCrossing
- **Crafter** — Full survival environment with crafting, combat, and resource management
- **Procgen** (stretch goal) — Coinrun, Starpilot, Maze

Each environment has hand-designed **diagnostic scenarios**: specific initial states + action sequences known to trigger failures.

---

## Architecture

```
dreambench/
├── envs/                     # Environment wrappers & scenario definitions
│   ├── atari/
│   │   ├── scenarios.yaml    # Diagnostic state-action pairs
│   │   └── wrapper.py
│   ├── minigrid/
│   └── crafter/
├── probes/                   # Failure detection modules
│   ├── object_permanence.py
│   ├── physics_consistency.py
│   ├── entity_integrity.py
│   ├── reward_fidelity.py
│   └── temporal_coherence.py
├── adapters/                 # World model interfaces
│   ├── base.py               # Abstract adapter class
│   ├── dreamerv3.py
│   ├── iris.py
│   ├── diamond.py
│   └── your_model.py         # Easy to add your own
├── metrics/                  # Scoring & aggregation
│   ├── per_probe.py
│   ├── composite.py
│   └── visualization.py
├── configs/                  # Hydra configs for experiments
├── scripts/
│   ├── run_benchmark.py
│   └── generate_report.py
└── paper/                    # LaTeX writeup
```

---

## How It Works

### 1. Adapter interface

Every world model implements a minimal adapter:

```python
class WorldModelAdapter(ABC):
    @abstractmethod
    def reset(self, initial_obs: np.ndarray) -> None:
        """Initialize the model's latent state from a real observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Predict next observation, reward, and done flag."""

    @abstractmethod
    def get_latent(self) -> np.ndarray:
        """Return current latent state (for probes that inspect it)."""
```

### 2. Diagnostic scenarios

Each scenario is a `(env, initial_state, action_sequence, ground_truth_trajectory)` tuple. We run the same action sequence through both the **real environment** and the **world model**, then compare.

```yaml
# Example: Atari Breakout object permanence test
- name: ball_behind_bricks
  env: BreakoutNoFrameskip-v4
  initial_state: checkpoints/breakout_corner_pocket.pkl
  actions: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0]
  probe: object_permanence
  description: >
    Ball enters a gap in the brick wall. Tests whether the model
    maintains the ball's existence and trajectory while occluded.
```

### 3. Probes

Probes are lightweight detectors that compare world-model rollouts against ground truth:

```python
class ObjectPermanenceProbe(BaseProbe):
    """Detects when entities vanish or teleport unexpectedly."""

    def __call__(self, predicted_frames, ground_truth_frames, metadata):
        # Track objects across frames using lightweight detection
        pred_tracks = self.tracker.track(predicted_frames)
        gt_tracks = self.tracker.track(ground_truth_frames)

        # Compare: did any GT object disappear in prediction?
        # Did any predicted object appear from nowhere?
        return PermanenceReport(
            vanished=self.find_vanished(pred_tracks, gt_tracks),
            teleported=self.find_teleported(pred_tracks, gt_tracks),
            duplicated=self.find_duplicated(pred_tracks),
            score=self.compute_score(pred_tracks, gt_tracks),
        )
```

### 4. Composite scoring

Each probe returns a score in [0, 1]. DreamBench aggregates these into a **radar chart** per model showing strengths and weaknesses, plus a single composite score for leaderboard ranking.

---

## Quick Start

```bash
# Install
git clone https://github.com/yourname/dreambench.git
cd dreambench
pip install -e ".[all]"

# Run the full benchmark on DreamerV3
python scripts/run_benchmark.py \
    model=dreamerv3 \
    envs=[atari,minigrid,crafter] \
    output_dir=results/dreamerv3

# Generate a diagnostic report
python scripts/generate_report.py \
    --results results/dreamerv3 \
    --output reports/dreamerv3.html
```

---

## Adding Your Own Model

1. Subclass `WorldModelAdapter` in `adapters/your_model.py`
2. Register it in `configs/model/your_model.yaml`
3. Run `python scripts/run_benchmark.py model=your_model`

That's it — all probes and environments work automatically.

---

## Planned Features

- [ ] **Automated scenario generation** — use RL to find action sequences that maximize model failures
- [ ] **Latent-space probes** — detect hallucination directly in latent representations
- [ ] **Video diff viewer** — side-by-side ground truth vs. model rollout with failure annotations
- [ ] **Online leaderboard** — submit results via CLI, auto-publish to a hosted leaderboard
- [ ] **Procgen support**
- [ ] **3D environment support** (MiniWorld, Habitat)

---

## Citation

```bibtex
@software{dreambench2026,
  title={DreamBench: Benchmarking Hallucination in Learned World Models},
  author={Your Name},
  year={2026},
  url={https://github.com/yourname/dreambench}
}
```

---

## License

MIT