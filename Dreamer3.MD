# DreamBench: DreamerV3 Integration — Implementation Plan

## Context

DreamBench is fully working with mock + real envs (MiniGrid, Crafter, Atari).
Now we integrate the first real world model: DreamerV3. The stub exists at
`dreambench/adapters/dreamerv3.py` and needs to be wired to a real
DreamerV3 implementation that can:

1. Load a trained checkpoint
2. Encode a real observation into the RSSM latent state
3. Given an action, predict the next observation, reward, and done flag

## Implementation choice: which DreamerV3 codebase

There are two viable options:

- **danijar/dreamerv3** (official, JAX/Ninjax) — canonical, but JAX
  dependency is heavy and the checkpoint format uses custom Ninjax
  serialization. No pretrained checkpoints are published.
- **NM512/dreamerv3-torch** (PyTorch) — well-tested reproduction,
  PyTorch is more widely installed, checkpoint is a standard
  `torch.save()` dict with `agent_state_dict`. Also no published
  pretrained checkpoints, but PyTorch training is faster to set up.

**Recommendation: Use NM512/dreamerv3-torch.** PyTorch is a more
accessible dependency for users, the checkpoint format is standard,
and the codebase is cleaner for extraction. The adapter should import
only the world model components (encoder, RSSM, decoder, reward head)
— NOT the full training loop or actor/critic.

**However:** Neither repo publishes pretrained weights. The plan must
include a training step OR a "bring your own checkpoint" approach.
We'll do both: provide a training script AND document how to point
at an existing checkpoint.

## How DreamerV3's RSSM works (adapter author must understand this)

DreamerV3's world model has these components:

```
Encoder:    obs (image) → embedding (vector)
RSSM:       (h_prev, z_prev, action) → h_next (deterministic)
Prior:      h_next → z_prior (stochastic, 32 categorical distributions)
Posterior:  (h_next, embedding) → z_posterior (stochastic)
Decoder:    (h, z) → reconstructed observation
Reward:     (h, z) → predicted reward
Continue:   (h, z) → predicted continuation probability
```

For benchmarking (open-loop imagination without real observations):
- `reset(obs)`: encode obs → embedding, run posterior to get (h₀, z₀)
- `step(action)`: run RSSM dynamics to get h₁, sample z₁ from PRIOR
  (not posterior — we don't have real obs), decode (h₁, z₁) → predicted
  obs, predict reward from (h₁, z₁)

This is "imagination" mode — the model predicts forward without seeing
real frames. This is exactly what DreamBench tests.

## How the runner uses the adapter

From `runner.py`, the flow for a real model (NOT mock) is:

```python
def run_model_rollout(self, scenario, gt_trajectory):
    # MockAdapter uses set_ground_truth — real models ignore it
    if hasattr(self.adapter, "set_ground_truth"):
        self.adapter.set_ground_truth(...)

    initial_obs = gt_trajectory.observations[0]
    self.adapter.reset(initial_obs)  # encode first real frame

    observations = [initial_obs]
    for action in scenario.actions:
        obs, reward, done = self.adapter.step(action)  # imagine forward
        observations.append(obs)
        ...
```

The adapter receives ONE real frame (the first), then predicts
everything else. This is open-loop rollout — the hardest test for
a world model.

---

## Task sequence

### Task 1: Install dreamerv3-torch as a dependency

Clone NM512/dreamerv3-torch into a location accessible to DreamBench.
Do NOT install it as a pip package — it's not packaged that way. Instead,
clone it and add it to PYTHONPATH, or copy the needed modules.

**Recommended approach:** Clone it as a git submodule or into a `third_party`
directory, then do a selective import of just the world model components.

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/NM512/dreamerv3-torch.git
cd ..
```

Then install its dependencies:
```bash
pip install torch torchvision
```

**Important:** dreamerv3-torch's requirements.txt pulls in many env
dependencies we don't need. Only install torch — the model code itself
needs only torch and numpy.

Validate:
```bash
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import sys; sys.path.insert(0, 'third_party/dreamerv3-torch'); from models import WorldModel; print('WorldModel import OK')"
```

The second import will likely fail because of internal import paths.
Read `third_party/dreamerv3-torch/models.py` and trace what it needs.
The key classes are in `models.py` (WorldModel, RSSM, ConvEncoder,
ConvDecoder, DenseHead). You may need to also copy or symlink
`networks.py`, `tools.py`, and any other dependencies.

If the import structure is too tangled, the alternative is to extract
just the forward-pass code into a standalone file. The RSSM forward
pass is ~100 lines — it's feasible to extract it.

---

### Task 2: Understand the dreamerv3-torch checkpoint format

Read `third_party/dreamerv3-torch/dreamer.py` to understand how
checkpoints are saved and loaded:

```python
# From dreamer.py — checkpoint loading
checkpoint = torch.load(logdir / "latest.pt")
agent.load_state_dict(checkpoint["agent_state_dict"])
```

The checkpoint contains `agent_state_dict` which is the full agent
(world model + actor + critic). We only need the world model weights.

Find out:
1. What keys in `agent_state_dict` correspond to the world model
   (encoder, RSSM, decoder, reward head, continue head)?
2. What is the expected observation shape? (Atari: typically 64x64 or
   96x96 grayscale. Crafter: 64x64 RGB. MiniGrid: variable.)
3. What is the action space representation? (One-hot for discrete envs.)

Write a small script to load a checkpoint and print the state dict keys:
```python
import torch
ckpt = torch.load("path/to/latest.pt", map_location="cpu")
for k in sorted(ckpt["agent_state_dict"].keys()):
    v = ckpt["agent_state_dict"][k]
    print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
```

This tells you the exact architecture dimensions and helps verify
the checkpoint is compatible.

---

### Task 3: Train a small checkpoint (or provide instructions)

Since no pretrained DreamerV3 checkpoints are published, we need to
train one. Start with the FASTEST option:

**Option A: Train on Breakout for ~500K steps (fastest useful checkpoint)**

```bash
cd third_party/dreamerv3-torch
python dreamer.py --configs atari \
    --task atari_breakout \
    --logdir ../../checkpoints/dreamerv3_breakout \
    --steps 500000 \
    --jax.platform cpu  # if no GPU
```

This takes ~2-4 hours on a GPU, longer on CPU. The model won't be
good at the game but WILL have learned basic visual dynamics — enough
for DreamBench to produce meaningful failure patterns.

**Option B: Train on Crafter (matches our scenario suite)**

```bash
python dreamer.py --configs crafter \
    --logdir ../../checkpoints/dreamerv3_crafter \
    --steps 500000
```

**Option C: Bring your own checkpoint**

If you already have a trained DreamerV3 checkpoint from any prior work,
skip training. Just note the config it was trained with (obs size,
action space, model size).

**Start with ONE environment.** Train a Breakout checkpoint first.
Multi-env support comes later.

---

### Task 4: Implement the DreamerV3Adapter

Replace the stub in `dreambench/adapters/dreamerv3.py`. The adapter
must do these things:

1. **`__init__(checkpoint_path, obs_size, action_size, device)`**
   - Load the dreamerv3-torch WorldModel from checkpoint
   - Move to device (CPU or GPU)
   - Set to eval mode

2. **`reset(initial_obs)`**
   - Preprocess obs: resize to model's expected input size, normalize
   - Encode through the CNN encoder to get embedding
   - Run the posterior model to get initial (h, z) latent state
   - Store (h, z) as self._state

3. **`step(action)`**
   - One-hot encode the action
   - Run RSSM dynamics: (h, z, action) → h_next
   - Sample z_next from the PRIOR (not posterior — no real obs available)
   - Decode (h_next, z_next) → predicted obs
   - Predict reward from (h_next, z_next)
   - Predict continue probability → done = continue < 0.5
   - Update self._state = (h_next, z_next)
   - Postprocess predicted obs back to uint8 RGB at original resolution
   - Return (obs, reward, done)

4. **`get_latent()`**
   - Return concatenation of (h, z) as a flat numpy array

Here is a skeleton (adjust imports based on what Task 1 reveals):

```python
"""DreamerV3 world model adapter using NM512/dreamerv3-torch."""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dreambench.adapters.base import WorldModelAdapter

# Add dreamerv3-torch to path
_D3_PATH = Path(__file__).parent.parent.parent / "third_party" / "dreamerv3-torch"
if str(_D3_PATH) not in sys.path:
    sys.path.insert(0, str(_D3_PATH))


class DreamerV3Adapter(WorldModelAdapter):
    """Adapter for DreamerV3 using NM512/dreamerv3-torch."""

    def __init__(
        self,
        checkpoint_path: str,
        obs_size: int = 64,       # model's expected input resolution
        action_size: int = 18,    # number of discrete actions
        device: str = "cpu",
        model_size: str = "small", # matches dreamerv3-torch config
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = torch.device(device)
        self._original_obs_shape = None

        # Import and build the world model
        # NOTE: exact imports depend on dreamerv3-torch's module structure.
        # Read models.py and adjust these imports.
        from models import WorldModel

        # Build model with correct config
        # You'll need to reconstruct the config dict that WorldModel expects.
        # Read the dreamerv3-torch code to find the right config keys.
        config = self._build_config(model_size)
        self.world_model = WorldModel(config).to(self.device)
        self.world_model.eval()

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        # Latent state
        self._h = None  # deterministic state (GRU hidden)
        self._z = None  # stochastic state (categorical)

    def _build_config(self, model_size):
        """Build config dict matching dreamerv3-torch's expectations."""
        # TODO: Read dreamerv3-torch/configs.yaml and replicate the
        # relevant world model config here. Key fields:
        # - encoder/decoder architecture
        # - RSSM dimensions (deter, stoch, classes)
        # - reward/continue head dimensions
        raise NotImplementedError("Build config from dreamerv3-torch")

    def _load_checkpoint(self, path: str):
        """Load world model weights from a dreamerv3-torch checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["agent_state_dict"]

        # Filter to only world model keys
        # The prefix depends on how the agent nests the world model.
        # Run Task 2's key-printing script to find the correct prefix.
        wm_keys = {
            k.replace("world_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("world_model.")
        }
        self.world_model.load_state_dict(wm_keys, strict=False)

    def _preprocess(self, obs: np.ndarray) -> torch.Tensor:
        """Convert uint8 RGB observation to model input format."""
        # dreamerv3-torch expects: (B, C, H, W) float32 in [0, 1]
        # or [-0.5, 0.5] depending on the codebase — CHECK THIS
        img = torch.from_numpy(obs).float() / 255.0
        if img.ndim == 3:
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        # Resize to model's expected resolution
        if img.shape[-2:] != (self.obs_size, self.obs_size):
            img = F.interpolate(
                img, size=(self.obs_size, self.obs_size),
                mode="bilinear", align_corners=False,
            )
        return img.to(self.device)

    def _postprocess(self, predicted: torch.Tensor) -> np.ndarray:
        """Convert model output back to uint8 RGB at original resolution."""
        # (1, C, H, W) float → (H, W, C) uint8
        img = predicted.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        if self._original_obs_shape is not None:
            h, w = self._original_obs_shape[:2]
            img = F.interpolate(
                img.permute(2, 0, 1).unsqueeze(0),
                size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        img = (img.clamp(0, 1) * 255).byte()
        return img.cpu().numpy()

    @torch.no_grad()
    def reset(self, initial_obs: np.ndarray) -> None:
        self._original_obs_shape = initial_obs.shape
        img = self._preprocess(initial_obs)

        # Encode observation
        embedding = self.world_model.encoder(img)

        # Initialize RSSM state
        # The exact API depends on dreamerv3-torch's RSSM implementation.
        # Typical pattern:
        self._h = torch.zeros(1, self.world_model.rssm.deter_size).to(self.device)
        action = torch.zeros(1, self.action_size).to(self.device)

        # Run posterior to get initial stochastic state
        # posterior uses both h and the real embedding
        self._h, self._z = self.world_model.rssm.observe(
            embedding, action, self._h
        )
        # NOTE: the exact method name (observe, obs_step, posterior)
        # varies by implementation. Read the RSSM class to find it.

    @torch.no_grad()
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._h is None:
            raise RuntimeError("Must call reset() before step()")

        # One-hot encode action
        action_oh = torch.zeros(1, self.action_size).to(self.device)
        action_oh[0, action] = 1.0

        # RSSM forward: predict next state using PRIOR (no real obs)
        self._h, self._z = self.world_model.rssm.imagine(
            action_oh, self._h, self._z
        )
        # NOTE: method might be called imagine_step, img_step, etc.

        # Decode predicted observation
        latent = torch.cat([self._h, self._z], dim=-1)
        predicted_img = self.world_model.decoder(latent)

        # Predict reward
        reward_pred = self.world_model.reward_head(latent)
        reward = float(reward_pred.squeeze().cpu())

        # Predict continuation
        if hasattr(self.world_model, "cont_head"):
            cont = self.world_model.cont_head(latent)
            done = float(cont.squeeze().cpu()) < 0.5
        else:
            done = False

        obs = self._postprocess(predicted_img)
        return obs, reward, done

    def get_latent(self) -> np.ndarray:
        if self._h is None:
            raise RuntimeError("Must call reset() before get_latent()")
        latent = torch.cat([self._h, self._z], dim=-1)
        return latent.squeeze(0).cpu().numpy()
```

**CRITICAL:** The skeleton above has many TODO markers and approximate
API calls. The exact method names, tensor shapes, and data formats
MUST be verified by reading the actual dreamerv3-torch source code.
Do not guess — read `models.py`, `networks.py`, and the RSSM class.

---

### Task 5: Observation preprocessing compatibility

DreamBench env wrappers return full-resolution RGB observations:
- MiniGrid: variable, typically ~(40, 40, 3) to ~(320, 320, 3)
  via `env.render()` 
- Crafter: (64, 64, 3) 
- Atari: (210, 160, 3)

DreamerV3 expects a fixed resolution (typically 64x64 or 96x96),
often grayscale for Atari. The adapter's `_preprocess` must handle:

1. Resize to model's expected resolution
2. Convert to grayscale if the model was trained on grayscale
3. Normalize to the model's expected range ([0,1] or [-0.5, 0.5])

And `_postprocess` must reverse this back to the original resolution
and format so probes can compare against ground truth frames.

**The resize introduces information loss** — predicted frames will be
blurrier than GT at full resolution. This WILL affect probe scores.
Document this as a known limitation. Probes compare at the upscaled
resolution, not the model's native resolution.

To mitigate: also compute probe scores at the model's native resolution
by downscaling GT frames to match. This is a better comparison but
requires a config flag. Implement native-resolution comparison as the
default, with an optional `--full_resolution` flag.

---

### Task 6: Update Hydra config

Update `dreambench/configs/model/dreamerv3.yaml`:

```yaml
name: dreamerv3
adapter: dreambench.adapters.dreamerv3.DreamerV3Adapter
checkpoint_path: checkpoints/dreamerv3_breakout/latest.pt
obs_size: 64
action_size: 18  # Atari max; override per env
device: cpu  # or cuda
model_size: small
```

The `action_size` needs to match what the model was trained with.
This creates a constraint: **a DreamerV3 checkpoint trained on
Breakout can only be benchmarked against Breakout scenarios** (or
other games with the same action space). Cross-env benchmarking
requires separate checkpoints per env.

Add env-specific overrides by creating compound configs:

```yaml
# configs/model/dreamerv3_breakout.yaml
name: dreamerv3_breakout
adapter: dreambench.adapters.dreamerv3.DreamerV3Adapter
checkpoint_path: checkpoints/dreamerv3_breakout/latest.pt
obs_size: 64
action_size: 4  # Breakout has 4 actions
device: cpu
model_size: small
```

---

### Task 7: Test with a single scenario first

Don't try to run the full benchmark immediately. Test the adapter
in isolation:

```python
# scripts/test_dreamerv3_adapter.py
import numpy as np
from dreambench.adapters.dreamerv3 import DreamerV3Adapter

adapter = DreamerV3Adapter(
    checkpoint_path="checkpoints/dreamerv3_breakout/latest.pt",
    obs_size=64,
    action_size=4,
    device="cpu",
)

# Fake observation
obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
adapter.reset(obs)
print(f"Latent shape: {adapter.get_latent().shape}")

# Step
for i in range(5):
    pred_obs, reward, done = adapter.step(1)
    print(f"Step {i}: obs shape={pred_obs.shape}, "
          f"reward={reward:.3f}, done={done}")
```

Expected behavior:
- `reset()` runs without error
- `get_latent()` returns a 1D array
- `step()` returns `(ndarray, float, bool)` with obs matching the
  original shape
- Predicted obs should look vaguely like a game frame, not random noise
  (even with a weakly-trained model)

---

### Task 8: Run the benchmark on Atari scenarios

Once the adapter works in isolation:

```bash
python scripts/run_benchmark.py \
    model=dreamerv3_breakout \
    env=atari \
    model.device=cpu
```

**Expected:** Scores will be LOW. This is correct — a real world model
hallucinating forward without real observations will make mistakes.
The interesting part is WHICH probes score low and which score high.

Typical DreamerV3 failure patterns:
- `reward_fidelity`: moderate (DreamerV3 has a dedicated reward head)
- `object_permanence`: low (objects tend to blur and vanish in long
  rollouts)
- `physics_consistency`: moderate (basic trajectories are OK, edge
  cases fail)
- `entity_integrity`: low (entities duplicate or merge in decoded frames)
- `temporal_coherence`: varies (short rollouts OK, degrades with length)

Generate the report and compare against the mock baseline:

```bash
python scripts/generate_report.py \
    --results results/dreamerv3_breakout \
    --output reports/dreamerv3_breakout.html
```

---

### Task 9: Add training documentation

Create `docs/training_dreamerv3.md` or add a section to README:

```markdown
## Training DreamerV3 for Benchmarking

DreamBench requires a trained DreamerV3 checkpoint. No pretrained
weights are published, so you must train your own.

### Quick start (Breakout, ~2-4 hours on GPU)

```bash
cd third_party/dreamerv3-torch
python dreamer.py --configs atari \
    --task atari_breakout \
    --logdir ../../checkpoints/dreamerv3_breakout \
    --steps 500000
```

### Run the benchmark

```bash
python scripts/run_benchmark.py \
    model=dreamerv3_breakout \
    env=atari
```

### Bring your own checkpoint

If you have an existing dreamerv3-torch checkpoint:

1. Place it at `checkpoints/your_model/latest.pt`
2. Create `configs/model/your_dreamerv3.yaml` with the correct
   obs_size and action_size for your training environment
3. Run: `python scripts/run_benchmark.py model=your_dreamerv3 env=atari`
```

---

## Success criteria

1. `DreamerV3Adapter` loads a real checkpoint without errors
2. `adapter.reset(obs)` encodes a real observation into latent state
3. `adapter.step(action)` produces plausible-looking predicted frames
4. Full benchmark runs on Atari scenarios without crashes
5. Report generates with meaningful (non-zero, non-one) probe scores
6. Radar chart shows differentiated scores across probe categories

## Critical notes for Claude Code

- **Read the dreamerv3-torch source code before writing adapter code.**
  The skeleton in Task 4 is approximate — exact class names, method
  signatures, and tensor formats MUST come from reading `models.py`,
  `networks.py`, and the RSSM implementation in the actual repo.
- **A checkpoint trained on Breakout can only benchmark Breakout.**
  The action space is baked into the model. Don't try to run a
  Breakout checkpoint on MiniGrid — it will crash or produce garbage.
- **Start with CPU.** Get it working on CPU first, then add GPU support.
  The benchmark is not training — inference is fast enough on CPU for
  15-frame rollouts.
- **The adapter must NOT import the full dreamerv3-torch training loop.**
  Only import the world model components (encoder, RSSM, decoder,
  reward/continue heads). The training infrastructure (replay buffer,
  actor, critic, optimizer) is irrelevant.
- **Observation preprocessing is the #1 source of bugs.** If the model
  was trained on 64x64 grayscale Atari frames and you feed it 210x160
  RGB, predictions will be garbage. Match the training preprocessing
  EXACTLY.
- **If you can't get dreamerv3-torch imports working** due to tangled
  internal dependencies, the fallback is to copy the ~300 lines of
  model code (encoder, RSSM, decoder, heads) into a standalone file
  inside `dreambench/adapters/`. This is ugly but pragmatic.
- **Do NOT modify probes, runner, or env wrappers.** The adapter is
  the only new code.