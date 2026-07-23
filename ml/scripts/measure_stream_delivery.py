"""Measure what the stream actually delivers, per transport.

Reports frames/second and bytes/frame for a WebSocket client held open against
the live API, once per requested encoding, so the two transports can be compared
on the same server in the same conditions.

What this measures and what it does not
---------------------------------------
This is the **delivery** rate: how fast finished frames leave the server and
arrive at a client. It is the ceiling on what a browser can paint, and it is the
number the server controls. It is *not* the browser's paint rate, which adds
JPEG decode and a canvas draw -- measure that with the Playwright check, which
counts real paints in a real renderer.

Reading the comparison
----------------------
``binary`` and ``base64`` differ only in transport, so the gap between them is
what removing base64 bought. Comparing either against a figure recorded before
the pump was split into producer/consumer tasks shows what the overlap bought.
Attributing the whole difference to one change would be wrong.

Usage:
    python ml/scripts/measure_stream_delivery.py --seconds 60
    python ml/scripts/measure_stream_delivery.py --encodings binary --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"

sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.streaming.wire import decode_stream_frame  # noqa: E402

#: Frames in this opening window are discarded. The first seconds cover source
#: open, model warmup and TCP ramp, none of which represent steady state.
WARMUP_S = 5.0


async def measure(base: str, source: str, encoding: str, seconds: float) -> dict[str, Any]:
    """Hold one stream open and count what arrives."""
    import websockets

    url = (
        f"{base.replace('http', 'ws')}/detect/stream"
        f"?source={source}&loop=true&encoding={encoding}"
    )

    frames = 0
    total_bytes = 0
    server_fps: list[float] = []
    started = time.perf_counter()
    steady_started: float | None = None
    steady_frames = 0
    steady_bytes = 0

    async with websockets.connect(url, max_size=None, ping_interval=20) as socket:
        while time.perf_counter() - started < seconds + WARMUP_S:
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=30)
            except TimeoutError:
                break

            size = len(message)
            if isinstance(message, bytes | bytearray):
                header, _ = decode_stream_frame(bytes(message))
                is_frame = header.get("kind") == "frame"
                reported = header.get("fps")
            else:
                payload = json.loads(message)
                is_frame = payload.get("kind") == "frame"
                reported = payload.get("fps")
            if not is_frame:
                continue

            frames += 1
            total_bytes += size
            elapsed = time.perf_counter() - started
            if elapsed >= WARMUP_S:
                if steady_started is None:
                    steady_started = time.perf_counter()
                    continue  # start the window at this frame, do not count it twice
                steady_frames += 1
                steady_bytes += size
                if isinstance(reported, int | float):
                    server_fps.append(float(reported))

        await socket.send(json.dumps({"action": "stop"}))

    window = (time.perf_counter() - steady_started) if steady_started else 0.0
    return {
        "encoding": encoding,
        "window_s": round(window, 2),
        "frames": steady_frames,
        "delivered_fps": round(steady_frames / window, 2) if window > 0 else 0.0,
        "bytes_per_frame": round(steady_bytes / steady_frames) if steady_frames else 0,
        "megabits_per_s": round(steady_bytes * 8 / window / 1e6, 2) if window > 0 else 0.0,
        "server_reported_fps": round(statistics.fmean(server_fps), 2) if server_fps else 0.0,
        "frames_including_warmup": frames,
        "bytes_including_warmup": total_bytes,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.api.rstrip("/")
    results = []
    for encoding in args.encodings:
        print(f"measuring {encoding} for {args.seconds:.0f}s (+{WARMUP_S:.0f}s warmup)")
        result = await measure(base, args.source, encoding, args.seconds)
        print(
            f"  {encoding:<7} {result['delivered_fps']:>6.2f} fps  "
            f"{result['bytes_per_frame'] / 1024:>7.1f} KiB/frame  "
            f"{result['megabits_per_s']:>6.2f} Mb/s"
        )
        results.append(result)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "source": args.source,
        "seconds_per_encoding": args.seconds,
        "results": results,
    }

    by_encoding = {r["encoding"]: r for r in results}
    if "binary" in by_encoding and "base64" in by_encoding:
        binary, legacy = by_encoding["binary"], by_encoding["base64"]
        if legacy["delivered_fps"] > 0:
            summary["transport_speedup"] = round(
                binary["delivered_fps"] / legacy["delivered_fps"], 3
            )
        if legacy["bytes_per_frame"] > 0:
            summary["bytes_ratio"] = round(binary["bytes_per_frame"] / legacy["bytes_per_frame"], 3)
        print(
            f"\nbinary vs base64: {summary.get('transport_speedup', 0):.2f}x fps, "
            f"{summary.get('bytes_ratio', 0):.2f}x bytes per frame"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seconds", type=float, default=60, help="steady-state window, per encoding"
    )
    parser.add_argument("--api", default="http://127.0.0.1:8100")
    parser.add_argument("--source", default="sample")
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["binary", "base64"],
        choices=["binary", "base64"],
    )
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args(argv)

    summary = asyncio.run(run(args))

    target = Path(args.out) if args.out else REPORTS_DIR / "stream_delivery.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
