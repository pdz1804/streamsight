"""Server-side frame annotation.

Boxes are burned into the JPEG rather than drawn client-side so that the pixels
and the overlay can never desynchronise while frames are in flight. The client
still receives the structured tracks for the legend and inspection panel.

Colours are derived from the track id via a fixed palette, so an object keeps its
colour for as long as ByteTrack keeps its identity -- that visual continuity is
what makes tracking legible to a viewer.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.models import Track

#: Perceptually distinct hues, BGR. Chosen to stay readable on both bright and
#: dark footage and to remain distinguishable for viewers with common CVD.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 176, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (148, 255, 181),
    (255, 108, 145),
    (116, 200, 255),
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(track_id: int | None) -> tuple[int, int, int]:
    """Stable BGR colour for a track id; unidentified boxes get a neutral grey."""
    if track_id is None:
        return (160, 160, 160)
    return PALETTE[track_id % len(PALETTE)]


def draw_tracks(
    frame: np.ndarray, tracks: list[Track], *, show_confidence: bool = True
) -> np.ndarray:
    """Return a copy of *frame* with boxes and id labels drawn on it."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(width, height) / 480))
    font_scale = max(0.4, min(width, height) / 1400)

    for track in tracks:
        color = color_for(track.track_id)
        x1 = int(max(0, min(track.x1, width - 1)))
        y1 = int(max(0, min(track.y1, height - 1)))
        x2 = int(max(0, min(track.x2, width - 1)))
        y2 = int(max(0, min(track.y2, height - 1)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        label = (
            track.class_name if track.track_id is None else f"#{track.track_id} {track.class_name}"
        )
        if show_confidence:
            label = f"{label} {track.confidence:.2f}"
        _draw_label(canvas, label, x1, y1, color, font_scale, thickness)
    return canvas


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw a filled chip above the box, flipping inside when it would clip."""
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = max(2, thickness * 2)
    chip_h = text_h + baseline + pad
    top = y - chip_h
    if top < 0:  # box hugs the top edge - put the chip inside it instead
        top = y
    bottom = top + chip_h
    right = min(canvas.shape[1], x + text_w + pad * 2)
    cv2.rectangle(canvas, (x, top), (right, bottom), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        text,
        (x + pad, bottom - baseline - pad // 2),
        _FONT,
        font_scale,
        _contrast_text_color(color),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _contrast_text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text for legibility against the chip colour."""
    blue, green, red = bgr
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)
