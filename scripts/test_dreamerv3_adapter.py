"""Test the DreamerV3 adapter in isolation (no checkpoint needed)."""

import numpy as np
import sys

# Test 1: Import and construct without checkpoint
print("Test 1: Construct adapter without checkpoint...")
from dreambench.adapters.dreamerv3 import DreamerV3Adapter

adapter = DreamerV3Adapter(
    checkpoint_path="",  # no checkpoint — uses random weights
    obs_size=64,
    action_size=4,
    device="cpu",
)
print("  OK — adapter constructed")

# Test 2: Reset with a fake observation
print("\nTest 2: Reset with fake observation...")
obs = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
adapter.reset(obs)
print(f"  OK — latent shape: {adapter.get_latent().shape}")

# Test 3: Step forward
print("\nTest 3: Step forward 5 times...")
for i in range(5):
    pred_obs, reward, done = adapter.step(1)
    print(f"  Step {i}: obs shape={pred_obs.shape}, "
          f"reward={reward:.4f}, done={done}")

# Test 4: Verify output shapes match input
print(f"\nTest 4: Output shape matches input...")
assert pred_obs.shape == obs.shape, (
    f"Shape mismatch: {pred_obs.shape} != {obs.shape}"
)
print(f"  OK — {pred_obs.shape} == {obs.shape}")

# Test 5: With a Crafter-sized observation (64x64)
print("\nTest 5: 64x64 observation (no resize needed)...")
adapter2 = DreamerV3Adapter(
    checkpoint_path="", obs_size=64, action_size=17, device="cpu"
)
obs64 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
adapter2.reset(obs64)
pred, r, d = adapter2.step(0)
print(f"  OK — pred shape: {pred.shape}")
assert pred.shape == (64, 64, 3)

print("\nAll tests passed. Adapter works (with random weights).")
print("To use with a real model, train a checkpoint first:")
print("  See docs/training_dreamerv3.md")
