"""Probe that checks entity count and appearance integrity over time."""

from typing import Any, Dict, List

import numpy as np

from dreambench.probes.base import BaseProbe, ProbeResult
from dreambench.probes.utils.tracking import detect_objects, track_objects, match_tracks


class EntityIntegrityProbe(BaseProbe):
    """Tracks entity count consistency between predicted and GT rollouts.

    Detects:
    - Count mismatches: frames where predicted entity count differs from GT
      (indicates merging or splitting of entities)
    - Appearance changes: large pixel differences in tracked entity patches

    Score is based on entity count accuracy over time.
    """

    name = "entity_integrity"

    def __init__(
        self,
        threshold: int = 30,
        patch_size: int = 16,
        appearance_threshold: float = 50.0,
        match_distance: float = 20.0,
    ):
        self.threshold = threshold
        self.patch_size = patch_size
        self.appearance_threshold = appearance_threshold
        self.match_distance = match_distance

    def _extract_patch(
        self, frame: np.ndarray, cx: float, cy: float
    ) -> np.ndarray:
        """Extract a square patch around a centroid."""
        h, w = frame.shape[:2]
        half = self.patch_size // 2
        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + half)
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + half)
        patch = frame[y0:y1, x0:x1]
        return patch.astype(np.float64)

    def __call__(
        self,
        predicted_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        predicted_rewards: List[float],
        gt_rewards: List[float],
        metadata: Dict[str, Any],
    ) -> ProbeResult:
        scenario_name = metadata.get("name", "")

        if len(gt_frames) < 2:
            return ProbeResult(
                probe_name=self.name,
                scenario_name=scenario_name,
                score=1.0,
                details={"reason": "not enough frames"},
            )

        # Compare entity counts frame by frame
        min_len = min(len(gt_frames), len(predicted_frames))
        bg_gt = gt_frames[0]
        bg_pred = predicted_frames[0]

        count_matches = 0
        count_mismatches: List[Dict[str, Any]] = []

        for i in range(min_len):
            gt_dets = detect_objects(gt_frames[i], bg_frame=bg_gt, threshold=self.threshold)
            pred_dets = detect_objects(
                predicted_frames[i], bg_frame=bg_pred, threshold=self.threshold
            )

            if len(gt_dets) == len(pred_dets):
                count_matches += 1
            else:
                count_mismatches.append({
                    "frame": i,
                    "gt_count": len(gt_dets),
                    "pred_count": len(pred_dets),
                    "delta": len(pred_dets) - len(gt_dets),
                })

        count_accuracy = count_matches / max(min_len, 1)

        # Check appearance consistency for matched tracks
        gt_tracks = track_objects(gt_frames, threshold=self.threshold)
        pred_tracks = track_objects(predicted_frames, threshold=self.threshold)
        matched, _, _ = match_tracks(
            gt_tracks, pred_tracks, distance_threshold=self.match_distance
        )

        appearance_changes: List[Dict[str, Any]] = []
        appearance_scores: List[float] = []

        for gt_idx, pred_idx in matched:
            gt_track = gt_tracks[gt_idx]
            pred_track = pred_tracks[pred_idx]

            # Compare patches at overlapping frame indices
            gt_frames_map = {pt[0]: (pt[1], pt[2]) for pt in gt_track}
            pred_frames_map = {pt[0]: (pt[1], pt[2]) for pt in pred_track}

            common_frames = set(gt_frames_map.keys()) & set(pred_frames_map.keys())
            for fidx in sorted(common_frames):
                if fidx >= len(gt_frames) or fidx >= len(predicted_frames):
                    continue
                gt_cx, gt_cy = gt_frames_map[fidx]
                pred_cx, pred_cy = pred_frames_map[fidx]

                gt_patch = self._extract_patch(gt_frames[fidx], gt_cx, gt_cy)
                pred_patch = self._extract_patch(
                    predicted_frames[fidx], pred_cx, pred_cy
                )

                # Compare patches -- handle size mismatches
                min_h = min(gt_patch.shape[0], pred_patch.shape[0])
                min_w = min(gt_patch.shape[1], pred_patch.shape[1])
                if min_h == 0 or min_w == 0:
                    continue

                gt_p = gt_patch[:min_h, :min_w]
                pred_p = pred_patch[:min_h, :min_w]

                # Handle color vs grayscale
                if gt_p.ndim != pred_p.ndim:
                    if gt_p.ndim == 3:
                        gt_p = np.mean(gt_p, axis=2)
                    if pred_p.ndim == 3:
                        pred_p = np.mean(pred_p, axis=2)

                diff = float(np.mean(np.abs(gt_p - pred_p)))
                if diff > self.appearance_threshold:
                    appearance_changes.append({
                        "gt_track": gt_idx,
                        "pred_track": pred_idx,
                        "frame": fidx,
                        "mean_diff": diff,
                    })
                # Normalize appearance score: 0 diff -> 1.0, threshold+ -> 0.0
                app_score = max(0.0, 1.0 - diff / max(self.appearance_threshold, 1.0))
                appearance_scores.append(app_score)

        # Combine scores: weight count accuracy more heavily
        avg_appearance = float(np.mean(appearance_scores)) if appearance_scores else 1.0
        score = 0.7 * count_accuracy + 0.3 * avg_appearance
        score = max(0.0, min(1.0, score))

        return ProbeResult(
            probe_name=self.name,
            scenario_name=scenario_name,
            score=score,
            details={
                "count_accuracy": count_accuracy,
                "count_mismatches": count_mismatches[:10],
                "avg_appearance_score": avg_appearance,
                "appearance_changes": appearance_changes[:10],
                "frames_analyzed": min_len,
                "matched_tracks": len(matched),
            },
        )
