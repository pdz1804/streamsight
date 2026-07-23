"""Long-running stability soak against the live API (NFR-6).

Drives the real WebSocket stream rather than a standalone inference loop, because
the things that leak are the things a loop does not exercise: the capture ring
buffer, the tracker's identity table, the metrics collector, the telemetry queue,
and the per-frame JPEG allocations.

Samples ``/metrics`` on an interval and reports drift in GPU memory and process
RSS between the first steady-state sample and the last. A short warmup window is
excluded so one-off allocations are not counted as a leak.

Usage:
    python ml/scripts/soak_stream.py --duration 14400
    python ml/scripts/soak_stream.py --duration 600 --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"

sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.wire import decode_stream_frame  # noqa: E402

#: Samples inside this window after start are excluded from drift analysis.
WARMUP_S = 120.0
#: NFR-6 threshold.
VRAM_DRIFT_LIMIT_MB = 200


def fetch_metrics(base: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as response:  # noqa: S310
            return json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - a failed probe is a data point, not a crash
        print(f"  metrics probe failed: {exc}")
        return None


async def consume_stream(ws_url: str, stop: asyncio.Event, counter: dict[str, int]) -> None:
    """Hold the stream open, reconnecting if the server drops it."""
    import websockets

    while not stop.is_set():
        try:
            async with websockets.connect(ws_url, max_size=None, ping_interval=20) as socket:
                counter["connections"] += 1
                while not stop.is_set():
                    message = await asyncio.wait_for(socket.recv(), timeout=30)
                    # Frames arrive as binary (header + JPEG) on the default
                    # transport; status messages are always text. Soaking the
                    # default is the point -- opting into base64 here would
                    # exercise a path no browser uses.
                    if isinstance(message, bytes | bytearray):
                        decode_stream_frame(bytes(message))
                        counter["frames"] += 1
                        continue
                    payload = json.loads(message)
                    if payload.get("kind") == "frame":
                        counter["frames"] += 1
                    elif payload.get("phase") == "error":
                        counter["errors"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconnecting on anything is the point
            counter["reconnects"] += 1
            if stop.is_set():
                return
            print(f"  stream dropped ({type(exc).__name__}), reconnecting")
            await asyncio.sleep(2.0)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.api.rstrip("/")
    ws_url = f"{base.replace('http', 'ws')}/detect/stream?source={args.source}&loop=true"

    if fetch_metrics(base) is None:
        raise SystemExit(f"API not reachable at {base}; start uvicorn first")

    stop = asyncio.Event()
    counter = {"frames": 0, "connections": 0, "reconnects": 0, "errors": 0}
    consumer = asyncio.create_task(consume_stream(ws_url, stop, counter))

    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    print(f"soaking {args.duration:.0f}s against {base}, sampling every {args.interval:.0f}s")

    try:
        while time.perf_counter() - started < args.duration:
            await asyncio.sleep(args.interval)
            elapsed = time.perf_counter() - started
            metrics = fetch_metrics(base)
            if metrics is None:
                counter["errors"] += 1
                continue
            sample = {
                "elapsed_s": round(elapsed, 1),
                "gpu_used_mb": metrics["gpu"]["used_mb"],
                "process_ram_mb": metrics["process_ram_mb"],
                "fps": metrics["fps"],
                "frames_processed": metrics["frames_processed"],
                "unique_tracks": metrics["unique_tracks"],
                "degraded_mode": metrics["degraded_mode"],
                "frames_received": counter["frames"],
            }
            samples.append(sample)
            print(
                f"  t={sample['elapsed_s']:>7.0f}s  gpu={sample['gpu_used_mb']:>5} MiB  "
                f"rss={sample['process_ram_mb']:>5} MiB  fps={sample['fps']:>5.1f}  "
                f"frames={sample['frames_processed']}"
            )
    finally:
        stop.set()
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer

    return summarize(samples, counter, args)


def summarize(
    samples: list[dict[str, Any]], counter: dict[str, int], args: argparse.Namespace
) -> dict[str, Any]:
    steady = [s for s in samples if s["elapsed_s"] >= WARMUP_S] or samples
    if not steady:
        raise SystemExit("no samples collected")

    gpu = [s["gpu_used_mb"] for s in steady]
    rss = [s["process_ram_mb"] for s in steady]
    fps = [s["fps"] for s in steady if s["fps"] > 0]

    gpu_drift = gpu[-1] - gpu[0]
    rss_drift = rss[-1] - rss[0]
    degraded = any(s["degraded_mode"] for s in samples)

    result = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "duration_s": args.duration,
        "interval_s": args.interval,
        "samples": len(samples),
        "steady_state_samples": len(steady),
        "warmup_excluded_s": WARMUP_S,
        "frames_processed": steady[-1]["frames_processed"],
        "frames_received": counter["frames"],
        "reconnects": counter["reconnects"],
        "errors": counter["errors"],
        "degraded_at_any_point": degraded,
        "gpu_first_mb": gpu[0],
        "gpu_last_mb": gpu[-1],
        "gpu_peak_mb": max(gpu),
        "gpu_drift_mb": gpu_drift,
        "rss_first_mb": rss[0],
        "rss_last_mb": rss[-1],
        "rss_peak_mb": max(rss),
        "rss_drift_mb": rss_drift,
        "fps_mean": round(statistics.fmean(fps), 2) if fps else 0.0,
        "fps_min": min(fps) if fps else 0.0,
        "vram_drift_limit_mb": VRAM_DRIFT_LIMIT_MB,
        "passed": gpu_drift < VRAM_DRIFT_LIMIT_MB and not degraded and counter["errors"] == 0,
        "series": samples,
    }

    print("\n=== soak summary ===")
    print(f"duration          {result['duration_s']:.0f}s over {result['samples']} samples")
    print(f"frames processed  {result['frames_processed']}")
    print(
        f"GPU  {result['gpu_first_mb']} -> {result['gpu_last_mb']} MiB "
        f"(drift {gpu_drift:+d} MiB, peak {result['gpu_peak_mb']})"
    )
    print(
        f"RSS  {result['rss_first_mb']} -> {result['rss_last_mb']} MiB "
        f"(drift {rss_drift:+d} MiB)"
    )
    print(f"FPS  mean {result['fps_mean']}, min {result['fps_min']}")
    print(f"reconnects {result['reconnects']}, errors {result['errors']}, degraded {degraded}")
    print(
        f"RESULT: {'PASS' if result['passed'] else 'FAIL'} "
        f"(limit {VRAM_DRIFT_LIMIT_MB} MiB VRAM drift)"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=14400, help="seconds (default 4 hours)")
    parser.add_argument("--interval", type=float, default=60, help="sampling interval, seconds")
    parser.add_argument("--api", default="http://127.0.0.1:8100")
    parser.add_argument("--source", default="sample")
    parser.add_argument("--out", help="JSON output path")
    args = parser.parse_args(argv)

    result = asyncio.run(run(args))

    target = Path(args.out) if args.out else REPORTS_DIR / "soak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {target}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
