"""Probe that detects temporal state reversions in predicted rollouts."""

from typing import Any, Dict, List

import numpy as np

from dreambench.probes.base import BaseProbe, ProbeResult


class TemporalCoherenceProbe(BaseProbe):
    """Detects state reversions where predicted frames rewind to earlier states.

    Compares each predicted frame to past predicted frames using pixel-level
    difference. If a later frame is suspiciously similar to a much earlier
    frame (but different from its immediate predecessor), this indicates
    a temporal reversion -- the world model "forgot" intervening state changes.

    Score = 1 - (reversion_count / total_frames), clamped to [0, 1].
    """

    name = "temporal_coherence"

    def __init__(
        self,
        similarity_threshold: float = 5.0,
        min_frame_gap: int = 3,
        dissimilarity_threshold: float = 15.0,
    ):
        """
        Args:
            similarity_threshold: Max mean pixel difference to consider
                two frames "similar" (potential reversion).
            min_frame_gap: Minimum number of frames apart for a similarity
                to count as a reversion (not just a slow scene).
            dissimilarity_threshold: Minimum mean pixel difference between
                a frame and its immediate predecessor to confirm the scene
                was actually changing (avoids false positives in static scenes).
        """
        self.similarity_threshold = similarity_threshold
        self.min_frame_gap = min_frame_gap
        self.dissimilarity_threshold = dissimilarity_threshold

    def _frame_difference(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute mean absolute pixel difference between two frames."""
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        # Handle shape mismatches gracefully
        if a_f.shape != b_f.shape:
            # Flatten to 1D for comparison if shapes differ
            min_size = min(a_f.size, b_f.size)
            a_f = a_f.ravel()[:min_size]
            b_f = b_f.ravel()[:min_size]
        return float(np.mean(np.abs(a_f - b_f)))

    def __call__(
        self,
        predicted_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        predicted_rewards: List[float],
        gt_rewards: List[float],
        metadata: Dict[str, Any],
    ) -> ProbeResult:
        scenario_name = metadata.get("name", "")
        n = len(predicted_frames)

        if n < self.min_frame_gap + 1:
            return ProbeResult(
                probe_name=self.name,
                scenario_name=scenario_name,
                score=1.0,
                details={"reason": "not enough frames for reversion detection"},
            )

        # Precompute differences between consecutive frames to identify
        # which parts of the sequence are actually changing
        consecutive_diffs = []
        for i in range(1, n):
            consecutive_diffs.append(
                self._frame_difference(predicted_frames[i], predicted_frames[i - 1])
            )

        # Detect reversions: for each frame, check if it's very similar
        # to a much earlier frame while being different from its neighbor
        reversions: List[Dict[str, Any]] = []
        reversion_frames = set()

        for i in range(self.min_frame_gap, n):
            # Check if this frame is changing relative to its predecessor
            if consecutive_diffs[i - 1] < self.dissimilarity_threshold:
                continue  # scene is static here, skip

            # Compare to earlier frames
            for j in range(0, i - self.min_frame_gap + 1):
                diff = self._frame_difference(
                    predicted_frames[i], predicted_frames[j]
                )
                if diff < self.similarity_threshold:
                    # Also verify the GT doesn't have the same reversion
                    gt_reversion = False
                    if i < len(gt_frames) and j < len(gt_frames):
                        gt_diff = self._frame_difference(gt_frames[i], gt_frames[j])
                        if gt_diff < self.similarity_threshold:
                            gt_reversion = True

                    if not gt_reversion:
                        reversions.append({
                            "current_frame": i,
                            "reverted_to_frame": j,
                            "similarity": diff,
                            "consecutive_diff": consecutive_diffs[i - 1],
                        })
                        reversion_frames.add(i)
                    break  # only report first match per frame

        reversion_count = len(reversion_frames)
        score = max(0.0, min(1.0, 1.0 - reversion_count / max(n, 1)))

        return ProbeResult(
            probe_name=self.name,
            scenario_name=scenario_name,
            score=score,
            details={
                "reversion_count": reversion_count,
                "total_frames": n,
                "reversions": reversions[:10],
                "avg_consecutive_diff": float(np.mean(consecutive_diffs))
                if consecutive_diffs
                else 0.0,
            },
        )
