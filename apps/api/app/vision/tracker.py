"""ByteTrack configuration and result parsing.

Ultralytics runs detection and association in a single ``model.track`` call, so
this module owns two narrow jobs: materialising the ByteTrack YAML the tracker
reads, and translating an Ultralytics ``Results`` object into our API schemas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core.models import Detection, Track

logger = logging.getLogger(__name__)

#: ByteTrack hyper-parameters. These are the Ultralytics 8.3.x keys -- there is no
#: ``max_age`` / ``min_hits`` in this implementation, and no standalone ByteTrack
#: package is involved.
BYTETRACK_CONFIG: dict[str, Any] = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def ensure_tracker_config(path: Path) -> Path:
    """Write the ByteTrack YAML if absent and return its path.

    Written from :data:`BYTETRACK_CONFIG` rather than shipped as a static file so
    the documented values and the values actually used cannot drift apart.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generated from apps/api/app/tracker.py:BYTETRACK_CONFIG - do not edit by hand."]
    for key, value in BYTETRACK_CONFIG.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote tracker config to %s", path)
    return path


def parse_results(result: Any) -> tuple[list[Detection], list[Track]]:
    """Split an Ultralytics ``Results`` into detections and identified tracks.

    Returns:
        ``(detections, tracks)`` where *detections* holds every box in the frame
        and *tracks* holds only the subset ByteTrack has assigned a persistent id.
        A box without an id is still detected -- it just has not survived enough
        frames to be confirmed -- so dropping it from *tracks* is correct, while
        inventing an id for it would corrupt identity metrics downstream.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []

    names: dict[int, str] = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    detections: list[Detection] = []
    tracks: list[Track] = []
    for index in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[index])
        class_id = int(classes[index])
        payload = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": float(confs[index]),
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
        }
        detections.append(Detection(**payload))
        if ids is not None:
            tracks.append(Track(**payload, track_id=int(ids[index])))
    return detections, tracks
