"""Probe that measures physics consistency between predicted and GT trajectories."""

from typing import Any, Dict, List, Tuple

import numpy as np

from dreambench.probes.base import BaseProbe, ProbeResult
from dreambench.probes.utils.tracking import track_objects, match_tracks, Track


def _compute_velocities(track: Track) -> List[Tuple[float, float]]:
    """Compute frame-to-frame velocity (displacement) for a track.

    Returns list of (vx, vy) tuples, one per consecutive pair of track points.
    """
    velocities = []
    for i in range(1, len(track)):
        dt = track[i][0] - track[i - 1][0]
        if dt == 0:
            dt = 1
        vx = (track[i][1] - track[i - 1][1]) / dt
        vy = (track[i][2] - track[i - 1][2]) / dt
        velocities.append((vx, vy))
    return velocities


def _compute_accelerations(
    velocities: List[Tuple[float, float]],
) -> List[float]:
    """Compute acceleration magnitude from velocity sequence."""
    accels = []
    for i in range(1, len(velocities)):
        ax = velocities[i][0] - velocities[i - 1][0]
        ay = velocities[i][1] - velocities[i - 1][1]
        accels.append(float(np.sqrt(ax ** 2 + ay ** 2)))
    return accels


def _velocity_correlation(
    vel_a: List[Tuple[float, float]],
    vel_b: List[Tuple[float, float]],
) -> float:
    """Compute correlation between two velocity profiles.

    Uses the cosine similarity of the speed sequences, returning a value
    in [0, 1] where 1 means perfectly correlated.
    """
    min_len = min(len(vel_a), len(vel_b))
    if min_len == 0:
        return 1.0  # no data to compare

    speeds_a = np.array(
        [np.sqrt(v[0] ** 2 + v[1] ** 2) for v in vel_a[:min_len]]
    )
    speeds_b = np.array(
        [np.sqrt(v[0] ** 2 + v[1] ** 2) for v in vel_b[:min_len]]
    )

    norm_a = np.linalg.norm(speeds_a)
    norm_b = np.linalg.norm(speeds_b)

    if norm_a < 1e-8 and norm_b < 1e-8:
        return 1.0  # both stationary
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0  # one stationary, other not

    # Pearson-like correlation mapped to [0, 1]
    correlation = float(np.dot(speeds_a, speeds_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, correlation))


class PhysicsConsistencyProbe(BaseProbe):
    """Compares velocity/acceleration profiles between predicted and GT.

    Measures how well the predicted world model preserves physical dynamics
    by comparing object motion patterns against ground truth.

    Score is the average velocity correlation across matched tracks,
    penalized by acceleration anomalies.
    """

    name = "physics_consistency"

    def __init__(
        self,
        threshold: int = 30,
        accel_spike_factor: float = 3.0,
        match_distance: float = 20.0,
    ):
        self.threshold = threshold
        self.accel_spike_factor = accel_spike_factor
        self.match_distance = match_distance

    def __call__(
        self,
        predicted_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        predicted_rewards: List[float],
        gt_rewards: List[float],
        metadata: Dict[str, Any],
    ) -> ProbeResult:
        scenario_name = metadata.get("name", "")

        if len(gt_frames) < 3:
            return ProbeResult(
                probe_name=self.name,
                scenario_name=scenario_name,
                score=1.0,
                details={"reason": "not enough frames for velocity analysis"},
            )

        gt_tracks = track_objects(gt_frames, threshold=self.threshold)
        pred_tracks = track_objects(predicted_frames, threshold=self.threshold)

        matched, _, _ = match_tracks(
            gt_tracks, pred_tracks, distance_threshold=self.match_distance
        )

        if not matched:
            # No matched tracks -- if both have no tracks, that's fine
            if not gt_tracks and not pred_tracks:
                return ProbeResult(
                    probe_name=self.name,
                    scenario_name=scenario_name,
                    score=1.0,
                    details={"reason": "no objects detected in either rollout"},
                )
            return ProbeResult(
                probe_name=self.name,
                scenario_name=scenario_name,
                score=0.0,
                details={"reason": "no tracks could be matched between rollouts"},
            )

        correlations = []
        accel_spike_events: List[Dict[str, Any]] = []

        for gt_idx, pred_idx in matched:
            gt_vel = _compute_velocities(gt_tracks[gt_idx])
            pred_vel = _compute_velocities(pred_tracks[pred_idx])

            corr = _velocity_correlation(gt_vel, pred_vel)
            correlations.append(corr)

            # Detect acceleration spikes in predicted not in GT
            gt_accel = _compute_accelerations(gt_vel)
            pred_accel = _compute_accelerations(pred_vel)

            if gt_accel:
                gt_mean_accel = float(np.mean(gt_accel))
                gt_std_accel = float(np.std(gt_accel)) if len(gt_accel) > 1 else gt_mean_accel
                spike_threshold = gt_mean_accel + self.accel_spike_factor * max(gt_std_accel, 1.0)
            else:
                spike_threshold = self.accel_spike_factor

            for i, a in enumerate(pred_accel):
                if a > spike_threshold:
                    accel_spike_events.append({
                        "pred_track_idx": pred_idx,
                        "velocity_step": i,
                        "acceleration": a,
                        "threshold": spike_threshold,
                    })

        avg_correlation = float(np.mean(correlations))

        # Penalize for acceleration spikes
        spike_penalty = len(accel_spike_events) / max(
            sum(len(_compute_velocities(pred_tracks[pi])) for _, pi in matched), 1
        )
        spike_penalty = min(spike_penalty, 1.0)

        score = max(0.0, min(1.0, avg_correlation * (1.0 - 0.5 * spike_penalty)))

        return ProbeResult(
            probe_name=self.name,
            scenario_name=scenario_name,
            score=score,
            details={
                "avg_velocity_correlation": avg_correlation,
                "matched_track_count": len(matched),
                "per_track_correlations": correlations[:10],
                "accel_spike_count": len(accel_spike_events),
                "accel_spikes": accel_spike_events[:10],
                "spike_penalty": spike_penalty,
            },
        )
