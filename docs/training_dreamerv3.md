# Training DreamerV3 for Benchmarking

DreamBench requires a trained DreamerV3 checkpoint. No pretrained weights are
published, so you must train your own.

## Prerequisites

```bash
# Clone dreamerv3-torch (if not already done)
mkdir -p third_party
cd third_party
git clone https://github.com/NM512/dreamerv3-torch.git
cd ..

# Install PyTorch
pip install torch torchvision
```

## Quick start: Breakout (~2-4 hours on GPU)

```bash
./scripts/train_dreamerv3.sh breakout 500000
```

This trains for 500K steps. The model won't master the game but will learn
basic visual dynamics — enough for DreamBench to produce meaningful diagnostics.

## Train on Crafter

```bash
./scripts/train_dreamerv3.sh crafter 500000
```

## Run the benchmark

```bash
# Atari (with a Breakout checkpoint)
python scripts/run_benchmark.py model=dreamerv3_breakout env=atari

# Crafter (with a Crafter checkpoint)
python scripts/run_benchmark.py model=dreamerv3_crafter env=crafter

# Generate report
python scripts/generate_report.py \
    --results results/dreamerv3_breakout \
    --output reports/dreamerv3_breakout.html
```

## Bring your own checkpoint

If you have an existing dreamerv3-torch checkpoint:

1. Place it at `checkpoints/your_model/latest.pt`
2. Create a Hydra config at `dreambench/configs/model/your_model.yaml`:

```yaml
name: your_model
adapter: dreambench.adapters.dreamerv3.DreamerV3Adapter
checkpoint_path: checkpoints/your_model/latest.pt
obs_size: 64          # must match training config
action_size: 4        # must match training env
device: cpu           # or cuda
```

3. Run: `python scripts/run_benchmark.py model=your_model env=atari`

## Architecture overrides

If your checkpoint was trained with non-default architecture, add these
to the Hydra config:

```yaml
dyn_deter: 512        # deterministic state size
dyn_stoch: 32         # stochastic state size
dyn_hidden: 512       # GRU hidden size
dyn_discrete: 32      # categorical classes
cnn_depth: 32         # conv filter depth
units: 512            # MLP units
```

Crafter defaults use larger dimensions (`dyn_deter: 4096`, `dyn_hidden: 1024`,
`units: 1024`). Use the `dreamerv3_crafter` config for Crafter checkpoints.

## Expected results

A 500K-step DreamerV3 will produce **low but differentiated** probe scores:

| Probe | Expected range | Why |
|---|---|---|
| reward_fidelity | 0.3 - 0.6 | Dedicated reward head, moderate accuracy |
| object_permanence | 0.1 - 0.3 | Objects blur and vanish in long rollouts |
| physics_consistency | 0.2 - 0.5 | Basic trajectories OK, edge cases fail |
| entity_integrity | 0.1 - 0.3 | Entities duplicate/merge in decoded frames |
| temporal_coherence | 0.2 - 0.6 | Short rollouts OK, degrades with length |

The interesting part is the **relative scores across probes** — this is the
diagnostic fingerprint that DreamBench provides.
