"""Probe that measures whether predicted rewards match ground-truth game logic."""

from typing import Any, Dict, List

import numpy as np

from dreambench.probes.base import BaseProbe, ProbeResult


class RewardFidelityProbe(BaseProbe):
    """Compares predicted reward sequence against ground truth.

    Score is the fraction of timesteps where reward matches (within tolerance).
    """

    name = "reward_fidelity"

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def __call__(
        self,
        predicted_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        predicted_rewards: List[float],
        gt_rewards: List[float],
        metadata: Dict[str, Any],
    ) -> ProbeResult:
        if not gt_rewards:
            return ProbeResult(
                probe_name=self.name,
                scenario_name=metadata.get("name", ""),
                score=1.0,
                details={"reason": "no rewards to compare"},
            )

        min_len = min(len(predicted_rewards), len(gt_rewards))
        pred = np.array(predicted_rewards[:min_len])
        gt = np.array(gt_rewards[:min_len])

        matches = np.abs(pred - gt) <= self.tolerance
        match_rate = float(np.mean(matches))

        # Track specific mismatches for diagnostics
        mismatches = []
        for i in range(min_len):
            if not matches[i]:
                mismatches.append({
                    "timestep": i,
                    "predicted": float(pred[i]),
                    "ground_truth": float(gt[i]),
                    "delta": float(pred[i] - gt[i]),
                })

        # Penalize length mismatch
        length_penalty = 1.0
        if len(predicted_rewards) != len(gt_rewards):
            length_penalty = min_len / max(len(predicted_rewards), len(gt_rewards))

        score = match_rate * length_penalty

        return ProbeResult(
            probe_name=self.name,
            scenario_name=metadata.get("name", ""),
            score=score,
            details={
                "match_rate": match_rate,
                "length_penalty": length_penalty,
                "num_mismatches": len(mismatches),
                "mismatches": mismatches[:10],  # cap for readability
                "predicted_length": len(predicted_rewards),
                "gt_length": len(gt_rewards),
            },
        )
