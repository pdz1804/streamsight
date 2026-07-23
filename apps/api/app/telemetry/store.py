"""SQLite persistence for frame stats and track lifecycles.

Writes happen on a dedicated thread behind a bounded queue so disk I/O can never
stall the inference loop; if the queue fills, telemetry is dropped rather than
back-pressuring the stream. Losing a log row is acceptable, dropping frames is not.

Only frame *summaries* and track *lifecycles* are stored -- not one row per box
per frame, which at 30 FPS would write millions of rows an hour for no analytical
gain.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.models import Track

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    session_id   TEXT    NOT NULL,
    frame_id     INTEGER NOT NULL,
    precision    TEXT    NOT NULL,
    imgsz        INTEGER NOT NULL,
    latency_ms   REAL    NOT NULL,
    detections   INTEGER NOT NULL,
    tracks       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);

CREATE TABLE IF NOT EXISTS track_events (
    session_id   TEXT    NOT NULL,
    track_id     INTEGER NOT NULL,
    class_name   TEXT    NOT NULL,
    first_seen   REAL    NOT NULL,
    last_seen    REAL    NOT NULL,
    frame_count  INTEGER NOT NULL,
    max_conf     REAL    NOT NULL,
    PRIMARY KEY (session_id, track_id)
);
"""

_QUEUE_MAXSIZE = 2048


@dataclass(frozen=True)
class FrameRecord:
    """One frame's summary, queued for persistence."""

    session_id: str
    frame_id: int
    precision: str
    imgsz: int
    latency_ms: float
    detections: int
    tracks: tuple[tuple[int, str, float], ...]
    ts: float


class DetectionStore:
    """Async SQLite writer for frame and track telemetry."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._queue: queue.Queue[FrameRecord | None] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._dropped = 0

    def start(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._thread = threading.Thread(target=self._drain, name="detection-store", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._thread = None
        if self._dropped:
            logger.warning("dropped %d telemetry records under load", self._dropped)

    def record(
        self,
        *,
        session_id: str,
        frame_id: int,
        precision: str,
        imgsz: int,
        latency_ms: float,
        detections: int,
        tracks: list[Track],
    ) -> None:
        """Queue one frame's telemetry. Never blocks."""
        payload = FrameRecord(
            session_id=session_id,
            frame_id=frame_id,
            precision=precision,
            imgsz=imgsz,
            latency_ms=latency_ms,
            detections=detections,
            tracks=tuple(
                (t.track_id, t.class_name, t.confidence) for t in tracks if t.track_id is not None
            ),
            ts=time.time(),
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped += 1

    @property
    def dropped_records(self) -> int:
        return self._dropped

    # -------------------------------------------------------------- internals

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _drain(self) -> None:
        conn = self._connect()
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                batch = [record]
                # Opportunistically coalesce so one commit covers many frames.
                while len(batch) < 64:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._queue.put(None)
                        break
                    batch.append(nxt)
                try:
                    self._write_batch(conn, batch)
                except Exception as exc:  # noqa: BLE001
                    # Catching only sqlite3.Error would let anything else kill
                    # this thread silently, after which every subsequent record()
                    # fills the queue and telemetry stops with no indication why.
                    logger.warning("telemetry write failed: %s", exc)
        finally:
            conn.close()

    @staticmethod
    def _write_batch(conn: sqlite3.Connection, batch: list[FrameRecord]) -> None:
        conn.executemany(
            "INSERT INTO frames (ts, session_id, frame_id, precision, imgsz, latency_ms,"
            " detections, tracks) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    r.ts,
                    r.session_id,
                    r.frame_id,
                    r.precision,
                    r.imgsz,
                    r.latency_ms,
                    r.detections,
                    len(r.tracks),
                )
                for r in batch
            ],
        )
        conn.executemany(
            "INSERT INTO track_events (session_id, track_id, class_name, first_seen, last_seen,"
            " frame_count, max_conf) VALUES (?,?,?,?,?,1,?)"
            " ON CONFLICT(session_id, track_id) DO UPDATE SET"
            "   last_seen = excluded.last_seen,"
            "   frame_count = track_events.frame_count + 1,"
            "   max_conf = MAX(track_events.max_conf, excluded.max_conf)",
            [
                (r.session_id, track_id, class_name, r.ts, r.ts, conf)
                for r in batch
                for track_id, class_name, conf in r.tracks
            ],
        )
        conn.commit()
