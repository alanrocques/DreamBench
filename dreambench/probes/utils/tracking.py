"""Lightweight object tracking using classical CV (numpy-based, opencv optional)."""

from typing import List, Optional, Tuple

import numpy as np

# Type aliases
Detection = Tuple[float, float, float]  # (centroid_x, centroid_y, area)
TrackPoint = Tuple[int, float, float, float]  # (frame_idx, cx, cy, area)
Track = List[TrackPoint]
MatchedPair = Tuple[int, int]  # (track_idx_a, track_idx_b)

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _connected_components_numpy(
    binary: np.ndarray,
) -> List[Detection]:
    """Simple flood-fill connected components using numpy only.

    This is a fallback when opencv is not available. It uses a basic
    iterative flood-fill approach.
    """
    labeled = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    h, w = binary.shape
    detections: List[Detection] = []

    for y in range(h):
        for x in range(w):
            if binary[y, x] > 0 and labeled[y, x] == 0:
                current_label += 1
                # BFS flood fill
                stack = [(y, x)]
                pixels: List[Tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if binary[cy, cx] == 0 or labeled[cy, cx] != 0:
                        continue
                    labeled[cy, cx] = current_label
                    pixels.append((cy, cx))
                    stack.extend(
                        [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]
                    )

                if len(pixels) >= 4:  # minimum area filter
                    ys = np.array([p[0] for p in pixels], dtype=np.float64)
                    xs = np.array([p[1] for p in pixels], dtype=np.float64)
                    centroid_x = float(np.mean(xs))
                    centroid_y = float(np.mean(ys))
                    area = float(len(pixels))
                    detections.append((centroid_x, centroid_y, area))

    return detections


def detect_objects(
    frame: np.ndarray,
    bg_frame: Optional[np.ndarray] = None,
    threshold: int = 30,
) -> List[Detection]:
    """Find objects using frame differencing + connected components.

    Args:
        frame: Current frame as a numpy array (H, W) or (H, W, C).
        bg_frame: Background reference frame. If None, assumes frame is
            already a difference image or uses a zero background.
        threshold: Pixel intensity threshold for foreground detection.

    Returns:
        List of (centroid_x, centroid_y, area) tuples for detected objects.
    """
    # Convert to grayscale if needed
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2).astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)

    if bg_frame is not None:
        if bg_frame.ndim == 3:
            bg_gray = np.mean(bg_frame, axis=2).astype(np.uint8)
        else:
            bg_gray = bg_frame.astype(np.uint8)
        diff = np.abs(gray.astype(np.int16) - bg_gray.astype(np.int16)).astype(
            np.uint8
        )
    else:
        diff = gray

    binary = (diff > threshold).astype(np.uint8)

    if _HAS_CV2:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        detections: List[Detection] = []
        for i in range(1, num_labels):  # skip background label 0
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area >= 4:  # minimum area filter
                cx = float(centroids[i, 0])
                cy = float(centroids[i, 1])
                detections.append((cx, cy, area))
        return detections
    else:
        return _connected_components_numpy(binary)


def track_objects(
    frames: List[np.ndarray],
    bg_frame: Optional[np.ndarray] = None,
    threshold: int = 30,
    match_distance: float = 40.0,
) -> List[Track]:
    """Track objects across a list of frames.

    Uses nearest-neighbor matching between consecutive frames.

    Args:
        frames: List of video frames as numpy arrays.
        bg_frame: Background frame for differencing. If None, uses the
            first frame as background.
        threshold: Pixel threshold for object detection.
        match_distance: Maximum distance to match objects between frames.

    Returns:
        List of tracks. Each track is a list of
        (frame_idx, centroid_x, centroid_y, area) tuples.
    """
    if not frames:
        return []

    if bg_frame is None:
        bg_frame = frames[0]

    tracks: List[Track] = []
    # Active track indices -> last known detection
    active: List[int] = []  # indices into tracks list

    for frame_idx, frame in enumerate(frames):
        detections = detect_objects(frame, bg_frame=bg_frame, threshold=threshold)

        if not active:
            # Initialize tracks from first detections
            for det in detections:
                tracks.append([(frame_idx, det[0], det[1], det[2])])
                active.append(len(tracks) - 1)
            continue

        # Match detections to active tracks using nearest neighbor
        if not detections:
            # No detections this frame -- active tracks get no new point
            continue

        # Get last known positions of active tracks
        last_positions = []
        for tidx in active:
            last_pt = tracks[tidx][-1]
            last_positions.append((last_pt[1], last_pt[2]))

        det_positions = [(d[0], d[1]) for d in detections]

        # Compute distance matrix
        n_tracks = len(last_positions)
        n_dets = len(det_positions)
        dist = np.zeros((n_tracks, n_dets))
        for i, (tx, ty) in enumerate(last_positions):
            for j, (dx, dy) in enumerate(det_positions):
                dist[i, j] = np.sqrt((tx - dx) ** 2 + (ty - dy) ** 2)

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        new_active: List[int] = []

        # Sort all pairs by distance
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                pairs.append((dist[i, j], i, j))
        pairs.sort()

        for d, i, j in pairs:
            if d > match_distance:
                break
            if i in matched_tracks or j in matched_dets:
                continue
            tidx = active[i]
            tracks[tidx].append(
                (frame_idx, detections[j][0], detections[j][1], detections[j][2])
            )
            matched_tracks.add(i)
            matched_dets.add(j)
            new_active.append(tidx)

        # Unmatched active tracks persist but don't update
        for i in range(n_tracks):
            if i not in matched_tracks:
                new_active.append(active[i])

        # Unmatched detections start new tracks
        for j in range(n_dets):
            if j not in matched_dets:
                tracks.append(
                    [
                        (
                            frame_idx,
                            detections[j][0],
                            detections[j][1],
                            detections[j][2],
                        )
                    ]
                )
                new_active.append(len(tracks) - 1)

        active = new_active

    return tracks


def match_tracks(
    tracks_a: List[Track],
    tracks_b: List[Track],
    distance_threshold: float = 20.0,
) -> Tuple[List[MatchedPair], List[int], List[int]]:
    """Match tracks between two rollouts using average centroid distance.

    Args:
        tracks_a: Tracks from rollout A.
        tracks_b: Tracks from rollout B.
        distance_threshold: Maximum average centroid distance for a match.

    Returns:
        Tuple of (matched_pairs, unmatched_a_indices, unmatched_b_indices).
    """
    if not tracks_a or not tracks_b:
        return (
            [],
            list(range(len(tracks_a))),
            list(range(len(tracks_b))),
        )

    def _track_avg_pos(track: Track) -> Tuple[float, float]:
        xs = [pt[1] for pt in track]
        ys = [pt[2] for pt in track]
        return (float(np.mean(xs)), float(np.mean(ys)))

    avg_a = [_track_avg_pos(t) for t in tracks_a]
    avg_b = [_track_avg_pos(t) for t in tracks_b]

    # Distance matrix
    n_a, n_b = len(avg_a), len(avg_b)
    dist = np.zeros((n_a, n_b))
    for i, (ax, ay) in enumerate(avg_a):
        for j, (bx, by) in enumerate(avg_b):
            dist[i, j] = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    # Greedy matching
    matched: List[MatchedPair] = []
    used_a = set()
    used_b = set()

    pairs = []
    for i in range(n_a):
        for j in range(n_b):
            pairs.append((dist[i, j], i, j))
    pairs.sort()

    for d, i, j in pairs:
        if d > distance_threshold:
            break
        if i in used_a or j in used_b:
            continue
        matched.append((i, j))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(n_a) if i not in used_a]
    unmatched_b = [j for j in range(n_b) if j not in used_b]

    return matched, unmatched_a, unmatched_b
