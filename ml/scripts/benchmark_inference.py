"""Throughput and VRAM harness for the full detect + track pipeline.

This is the measurement contract every later phase builds on. It reports the
*full-pipeline* rate (decode, inference, tracking), not detector-only
throughput, because that is the number the streaming viewer actually delivers.

Usage:
    python ml/scripts/benchmark_inference.py --engine int8_trt --imgsz 640 --frames 300
    python ml/scripts/benchmark_inference.py --engine fp32_cpu --duration 900
    python ml/scripts/benchmark_inference.py --engine fp16_onnx --frames 200 --json out.json

Duration control is mutually exclusive: pass either --frames or --duration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

import cv2  # noqa: E402
from app.core.config import get_settings, probe_gpu  # noqa: E402
from app.inference.backends import BACKENDS, availability, get_backend  # noqa: E402
from app.inference.detector import Detector  # noqa: E402

#: Frames discarded before timing starts. The first inferences pay one-off
#: allocation and kernel-selection costs that would skew a short run.
WARMUP_FRAMES = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--engine",
        "--precision",
        dest="engine",
        default="fp32_gpu",
        choices=sorted(BACKENDS),
        help="inference backend to measure",
    )
    parser.add_argument(
        "--imgsz",
        "--resolution",
        dest="imgsz",
        type=int,
        default=640,
        help="inference resolution in pixels",
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--frames", type=int, help="stop after this many timed frames")
    duration.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--source", help="video path (defaults to the bundled sample clip)")
    parser.add_argument("--json", dest="json_out", help="also write the result to this JSON file")
    parser.add_argument(
        "--vram-poll-frames",
        type=int,
        default=20,
        help="how often to sample GPU memory",
    )
    args = parser.parse_args(argv)
    if args.frames is None and args.duration is None:
        args.frames = 300
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    gpu = probe_gpu()
    backend = get_backend(args.engine)

    runnable, reason = availability(backend, settings, gpu.available)
    if not runnable:
        raise SystemExit(f"backend '{args.engine}' cannot run here: {reason}")

    source = Path(args.source) if args.source else settings.assets_dir / "sample.mp4"
    if not source.exists():
        raise SystemExit(f"video source not found: {source} (run ml/scripts/fetch_assets.py)")

    detector = Detector(backend, args.imgsz, settings)
    load_started = time.perf_counter()
    detector.load()
    load_s = time.perf_counter() - load_started
    after_load = probe_gpu()

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {source}")

    latencies: list[float] = []
    peak_vram_mb = after_load.used_mb
    detections_total = 0
    unique_tracks: set[int] = set()
    frame_index = 0
    timed = 0
    started = 0.0
    deadline_reached = False

    print(f"backend={backend.key} imgsz={args.imgsz} source={source.name}")
    print(f"model load {load_s:.2f}s, VRAM after load {after_load.used_mb} MiB")

    while True:
        ok, frame = capture.read()
        if not ok:
            # Loop the clip so a long soak does not end when the file does.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if not ok:
                break

        frame_index += 1
        if frame_index <= WARMUP_FRAMES:
            detector.track(frame)
            if frame_index == WARMUP_FRAMES:
                detector.reset_tracker()
                started = time.perf_counter()
            continue

        tick = time.perf_counter()
        _, tracks, _ = detector.track(frame)
        latencies.append((time.perf_counter() - tick) * 1000.0)
        timed += 1
        detections_total += len(tracks)
        unique_tracks.update(t.track_id for t in tracks if t.track_id is not None)

        if timed % args.vram_poll_frames == 0 and gpu.available:
            peak_vram_mb = max(peak_vram_mb, probe_gpu().used_mb)

        if args.frames is not None and timed >= args.frames:
            break
        if args.duration is not None and time.perf_counter() - started >= args.duration:
            deadline_reached = True
            break

    elapsed = time.perf_counter() - started
    capture.release()
    detector.close()

    ordered = sorted(latencies)
    result: dict[str, Any] = {
        "engine": backend.key,
        "label": backend.label,
        "device": backend.device,
        "imgsz": args.imgsz,
        "source": source.name,
        "frames": timed,
        "elapsed_s": round(elapsed, 2),
        "fps_mean": round(timed / elapsed, 2) if elapsed > 0 else 0.0,
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_p50_ms": round(_percentile(ordered, 0.50), 2),
        "latency_p95_ms": round(_percentile(ordered, 0.95), 2),
        "model_load_s": round(load_s, 2),
        "gpu_available": gpu.available,
        "gpu_name": gpu.name,
        "vram_after_load_mb": after_load.used_mb,
        "vram_peak_mb": peak_vram_mb,
        "detections_per_frame": round(detections_total / timed, 2) if timed else 0.0,
        "unique_tracks": len(unique_tracks),
        "stopped_on": "duration" if deadline_reached else "frames",
    }

    print(
        f"\n{result['fps_mean']:.2f} FPS mean over {timed} frames "
        f"({result['latency_mean_ms']:.1f} ms mean, p95 {result['latency_p95_ms']:.1f} ms)"
    )
    if gpu.available:
        print(f"peak VRAM {peak_vram_mb} MiB on {gpu.name}")
    else:
        print("no NVIDIA GPU: CPU path, VRAM not applicable")
    return result


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
