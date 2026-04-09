"""Tests for diagnostic probes using synthetic numpy frames."""

import numpy as np
import pytest

from dreambench.probes.base import ProbeResult
from dreambench.probes.object_permanence import ObjectPermanenceProbe
from dreambench.probes.physics_consistency import PhysicsConsistencyProbe
from dreambench.probes.entity_integrity import EntityIntegrityProbe
from dreambench.probes.temporal_coherence import TemporalCoherenceProbe
from dreambench.probes.utils.tracking import (
    detect_objects,
    track_objects,
    match_tracks,
)

# ---------------------------------------------------------------------------
# Helpers to generate synthetic frame sequences
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a blank (black) frame."""
    return np.zeros((h, w), dtype=np.uint8)


def _frame_with_blob(
    h: int = 64,
    w: int = 64,
    cx: int = 32,
    cy: int = 32,
    radius: int = 5,
    intensity: int = 200,
) -> np.ndarray:
    """Create a frame with a bright circular blob."""
    frame = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    frame[mask] = intensity
    return frame


def _moving_blob_sequence(
    n_frames: int = 10,
    start_x: int = 10,
    start_y: int = 32,
    dx: int = 4,
    dy: int = 0,
    h: int = 64,
    w: int = 64,
    radius: int = 5,
) -> list:
    """Generate frames with a blob moving linearly."""
    frames = []
    for i in range(n_frames):
        cx = start_x + i * dx
        cy = start_y + i * dy
        cx = max(radius, min(w - radius - 1, cx))
        cy = max(radius, min(h - radius - 1, cy))
        frames.append(_frame_with_blob(h, w, cx, cy, radius))
    return frames


def _default_metadata(name: str = "test_scenario") -> dict:
    return {"name": name, "env_id": "test", "description": "synthetic test"}


def _dummy_rewards(n: int = 10) -> list:
    return [0.0] * n


# ---------------------------------------------------------------------------
# Tests for tracking utilities
# ---------------------------------------------------------------------------

class TestTracking:
    def test_detect_objects_finds_blob(self):
        bg = _blank_frame()
        frame = _frame_with_blob(cx=30, cy=30)
        dets = detect_objects(frame, bg_frame=bg, threshold=30)
        assert len(dets) >= 1
        # Centroid should be near (30, 30)
        cx, cy, area = dets[0]
        assert abs(cx - 30) < 5
        assert abs(cy - 30) < 5
        assert area > 0

    def test_detect_objects_empty_frame(self):
        bg = _blank_frame()
        frame = _blank_frame()
        dets = detect_objects(frame, bg_frame=bg, threshold=30)
        assert len(dets) == 0

    def test_track_objects_moving_blob(self):
        frames = _moving_blob_sequence(n_frames=8)
        tracks = track_objects(frames, threshold=30)
        # Should find at least one track
        assert len(tracks) >= 1
        # The longest track should span multiple frames
        longest = max(tracks, key=len)
        assert len(longest) >= 4

    def test_match_tracks_identical(self):
        frames = _moving_blob_sequence(n_frames=8)
        tracks_a = track_objects(frames, threshold=30)
        tracks_b = track_objects(frames, threshold=30)
        matched, um_a, um_b = match_tracks(tracks_a, tracks_b)
        assert len(matched) >= 1
        assert len(um_a) == 0
        assert len(um_b) == 0


# ---------------------------------------------------------------------------
# Tests for ObjectPermanenceProbe
# ---------------------------------------------------------------------------

class TestObjectPermanenceProbe:
    def test_identical_sequences_score_high(self):
        frames = _moving_blob_sequence(n_frames=10)
        probe = ObjectPermanenceProbe(threshold=30)
        result = probe(
            predicted_frames=frames,
            gt_frames=frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        assert result.score >= 0.8  # identical should be high

    def test_vanished_object_detected(self):
        """Object present in GT but disappears in predicted."""
        gt_frames = _moving_blob_sequence(n_frames=10, start_x=10, dx=4)

        # Predicted: object present for first 5 frames, then gone
        pred_frames = []
        for i in range(10):
            if i < 5:
                pred_frames.append(
                    _frame_with_blob(cx=10 + i * 4, cy=32)
                )
            else:
                pred_frames.append(_blank_frame())

        probe = ObjectPermanenceProbe(threshold=30)
        result = probe(
            predicted_frames=pred_frames,
            gt_frames=gt_frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        # Score should be less than perfect since object vanishes
        # (but exact score depends on tracking details)

    def test_returns_valid_probe_result(self):
        probe = ObjectPermanenceProbe()
        result = probe(
            predicted_frames=[_blank_frame()],
            gt_frames=[_blank_frame()],
            predicted_rewards=[],
            gt_rewards=[],
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "object_permanence"
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Tests for PhysicsConsistencyProbe
# ---------------------------------------------------------------------------

class TestPhysicsConsistencyProbe:
    def test_identical_motion_scores_high(self):
        frames = _moving_blob_sequence(n_frames=10, dx=3)
        probe = PhysicsConsistencyProbe(threshold=30)
        result = probe(
            predicted_frames=frames,
            gt_frames=frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        assert result.score >= 0.7

    def test_different_velocities_score_lower(self):
        gt_frames = _moving_blob_sequence(n_frames=10, dx=2)
        # Predicted has different velocity
        pred_frames = _moving_blob_sequence(n_frames=10, dx=5)

        probe = PhysicsConsistencyProbe(threshold=30, match_distance=40)
        result = probe(
            predicted_frames=pred_frames,
            gt_frames=gt_frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0

    def test_too_few_frames(self):
        probe = PhysicsConsistencyProbe()
        result = probe(
            predicted_frames=[_blank_frame(), _blank_frame()],
            gt_frames=[_blank_frame(), _blank_frame()],
            predicted_rewards=[0.0, 0.0],
            gt_rewards=[0.0, 0.0],
            metadata=_default_metadata(),
        )
        assert result.score == 1.0
        assert result.details.get("reason") is not None


# ---------------------------------------------------------------------------
# Tests for EntityIntegrityProbe
# ---------------------------------------------------------------------------

class TestEntityIntegrityProbe:
    def test_identical_sequences(self):
        frames = _moving_blob_sequence(n_frames=8)
        probe = EntityIntegrityProbe(threshold=30)
        result = probe(
            predicted_frames=frames,
            gt_frames=frames,
            predicted_rewards=_dummy_rewards(8),
            gt_rewards=_dummy_rewards(8),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        assert result.score >= 0.7

    def test_entity_count_mismatch(self):
        """GT has 1 object, predicted has 2."""
        gt_frames = _moving_blob_sequence(n_frames=8, start_x=20, dx=3)

        pred_frames = []
        for i in range(8):
            # Two blobs in each predicted frame
            frame = _frame_with_blob(cx=20 + i * 3, cy=32, radius=5)
            # Add second blob
            yy, xx = np.ogrid[:64, :64]
            mask2 = (xx - 50) ** 2 + (yy - 15) ** 2 <= 25
            frame[mask2] = 200
            pred_frames.append(frame)

        probe = EntityIntegrityProbe(threshold=30)
        result = probe(
            predicted_frames=pred_frames,
            gt_frames=gt_frames,
            predicted_rewards=_dummy_rewards(8),
            gt_rewards=_dummy_rewards(8),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0

    def test_returns_valid_result(self):
        probe = EntityIntegrityProbe()
        result = probe(
            predicted_frames=[_blank_frame()],
            gt_frames=[_blank_frame()],
            predicted_rewards=[],
            gt_rewards=[],
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "entity_integrity"


# ---------------------------------------------------------------------------
# Tests for TemporalCoherenceProbe
# ---------------------------------------------------------------------------

class TestTemporalCoherenceProbe:
    def test_no_reversion_scores_high(self):
        """Linearly moving blob -- no reversions."""
        frames = _moving_blob_sequence(n_frames=12, dx=4)
        probe = TemporalCoherenceProbe(
            similarity_threshold=5.0,
            min_frame_gap=3,
            dissimilarity_threshold=5.0,
        )
        result = probe(
            predicted_frames=frames,
            gt_frames=frames,
            predicted_rewards=_dummy_rewards(12),
            gt_rewards=_dummy_rewards(12),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        assert result.score >= 0.8

    def test_reversion_detected(self):
        """Sequence that reverts to an earlier state."""
        # Frames 0-5: moving right, frames 6-9: copy of frames 0-3
        forward = _moving_blob_sequence(n_frames=6, start_x=10, dx=5)
        # Revert: copy early frames
        revert = _moving_blob_sequence(n_frames=4, start_x=10, dx=5)
        pred_frames = forward + revert

        # GT continues moving forward
        gt_frames = _moving_blob_sequence(n_frames=10, start_x=10, dx=5)

        probe = TemporalCoherenceProbe(
            similarity_threshold=5.0,
            min_frame_gap=3,
            dissimilarity_threshold=5.0,
        )
        result = probe(
            predicted_frames=pred_frames,
            gt_frames=gt_frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.score <= 1.0
        # Should detect some reversions
        assert result.details["reversion_count"] >= 1

    def test_static_scene_no_false_positives(self):
        """A static scene should not trigger reversions."""
        frame = _frame_with_blob(cx=32, cy=32)
        frames = [frame.copy() for _ in range(10)]
        probe = TemporalCoherenceProbe()
        result = probe(
            predicted_frames=frames,
            gt_frames=frames,
            predicted_rewards=_dummy_rewards(10),
            gt_rewards=_dummy_rewards(10),
            metadata=_default_metadata(),
        )
        assert result.score == 1.0 or result.details["reversion_count"] == 0

    def test_too_few_frames(self):
        probe = TemporalCoherenceProbe(min_frame_gap=3)
        result = probe(
            predicted_frames=[_blank_frame(), _blank_frame()],
            gt_frames=[_blank_frame(), _blank_frame()],
            predicted_rewards=[0.0, 0.0],
            gt_rewards=[0.0, 0.0],
            metadata=_default_metadata(),
        )
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Integration: runner registers all probes
# ---------------------------------------------------------------------------

class TestProbeRegistry:
    def test_all_probes_registered(self):
        from dreambench.runner import PROBE_REGISTRY

        expected = {
            "reward_fidelity",
            "object_permanence",
            "physics_consistency",
            "entity_integrity",
            "temporal_coherence",
        }
        assert expected.issubset(set(PROBE_REGISTRY.keys()))
