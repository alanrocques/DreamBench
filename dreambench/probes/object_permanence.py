"""Probe that detects object permanence violations in world model predictions."""

from typing import Any, Dict, List

import numpy as np

from dreambench.probes.base import BaseProbe, ProbeResult
from dreambench.probes.utils.tracking import track_objects, match_tracks, Track


class ObjectPermanenceProbe(BaseProbe):
    """Detects objects that vanish, teleport, or duplicate in predicted frames.

    Compares object tracks in predicted vs ground-truth rollouts to find:
    - Vanished objects: present in GT but missing in prediction
    - Teleported objects: sudden large position jumps not present in GT
    - Duplicated objects: extra objects appearing in prediction with no GT match

    Score = 1 - (failure_count / total_tracked_objects), clamped to [0, 1].
    """

    name = "object_permanence"

    def __init__(
        self,
        threshold: int = 30,
        teleport_distance: float = 50.0,
        match_distance: float = 20.0,
    ):
        self.threshold = threshold
        self.teleport_distance = teleport_distance
        self.match_distance = match_distance

    def _detect_teleports(self, track: Track) -> List[Dict[str, Any]]:
        """Find sudden large jumps within a single track."""
        teleports = []
        for i in range(1, len(track)):
            prev = track[i - 1]
            curr = track[i]
            dx = curr[1] - prev[1]
            dy = curr[2] - prev[2]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > self.teleport_distance:
                teleports.append({
                    "frame": curr[0],
                    "from": (prev[1], prev[2]),
                    "to": (curr[1], curr[2]),
                    "distance": float(dist),
                })
        return teleports

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
                details={"reason": "not enough frames to track"},
            )

        # Track objects in both rollouts
        gt_tracks = track_objects(gt_frames, threshold=self.threshold)
        pred_tracks = track_objects(predicted_frames, threshold=self.threshold)

        # Match tracks between GT and predicted
        matched, unmatched_gt, unmatched_pred = match_tracks(
            gt_tracks, pred_tracks, distance_threshold=self.match_distance
        )

        # Count failures
        vanished_events: List[Dict[str, Any]] = []
        teleported_events: List[Dict[str, Any]] = []
        duplicated_events: List[Dict[str, Any]] = []

        # Vanished: GT tracks with no match in predicted
        for gt_idx in unmatched_gt:
            t = gt_tracks[gt_idx]
            vanished_events.append({
                "gt_track_idx": gt_idx,
                "track_length": len(t),
                "avg_position": (
                    float(np.mean([p[1] for p in t])),
                    float(np.mean([p[2] for p in t])),
                ),
            })

        # Duplicated: predicted tracks with no match in GT
        for pred_idx in unmatched_pred:
            t = pred_tracks[pred_idx]
            # Only count as duplicated if the track is non-trivial
            if len(t) >= 2:
                duplicated_events.append({
                    "pred_track_idx": pred_idx,
                    "track_length": len(t),
                    "avg_position": (
                        float(np.mean([p[1] for p in t])),
                        float(np.mean([p[2] for p in t])),
                    ),
                })

        # Teleported: check predicted tracks for large jumps not in GT
        for gt_idx, pred_idx in matched:
            gt_teleports = self._detect_teleports(gt_tracks[gt_idx])
            pred_teleports = self._detect_teleports(pred_tracks[pred_idx])
            # Only flag teleports in predicted that are NOT in GT
            excess = len(pred_teleports) - len(gt_teleports)
            if excess > 0:
                for tp in pred_teleports[-excess:]:
                    teleported_events.append({
                        "pred_track_idx": pred_idx,
                        **tp,
                    })

        # Also check unmatched predicted tracks for teleports
        for pred_idx in unmatched_pred:
            teleports = self._detect_teleports(pred_tracks[pred_idx])
            for tp in teleports:
                teleported_events.append({
                    "pred_track_idx": pred_idx,
                    **tp,
                })

        failure_count = (
            len(vanished_events) + len(teleported_events) + len(duplicated_events)
        )
        total_objects = max(len(gt_tracks) + len(pred_tracks), 1)
        score = max(0.0, min(1.0, 1.0 - failure_count / total_objects))

        return ProbeResult(
            probe_name=self.name,
            scenario_name=scenario_name,
            score=score,
            details={
                "gt_track_count": len(gt_tracks),
                "pred_track_count": len(pred_tracks),
                "matched_count": len(matched),
                "vanished": vanished_events[:10],
                "teleported": teleported_events[:10],
                "duplicated": duplicated_events[:10],
                "failure_count": failure_count,
                "total_objects": total_objects,
            },
        )
