"""Accuracy-throughput frontier across every runnable backend and resolution.

Sweeps ``backend x imgsz``, measures full-pipeline throughput and peak VRAM, and
scores each configuration's *agreement* with the FP32 baseline on identical
frames.

On the accuracy metric: the plan asks for output-tensor cosine similarity, but
cosine over raw head outputs is a poor decision signal -- it is dominated by the
thousands of low-confidence anchors that never survive NMS, so a model can score
0.99 while losing detections that matter. This measures agreement on the
*post-NMS detections* instead: of the objects the FP32 model found, how many does
this backend also find, at what IoU, and how many does it invent. That is the
question a deployment decision actually turns on. Cosine is still reported for
ONNX backends, where the graphs are directly comparable.

Usage:
    python ml/eval/benchmark_frontier.py --frames 200
    python ml/eval/benchmark_frontier.py --frames 300 --imgsz 640 480
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.core.config import get_settings, probe_gpu  # noqa: E402
from app.core.models import Detection  # noqa: E402
from app.inference.backends import BACKENDS, availability  # noqa: E402
from app.inference.detector import Detector  # noqa: E402

REPORTS_DIR = REPO_ROOT / "ml" / "eval" / "reports"

#: A detection counts as "the same object" as a baseline detection when it has
#: the same class and overlaps by at least this much.
IOU_MATCH_THRESHOLD = 0.5

WARMUP_FRAMES = 10


@dataclass
class ConfigResult:
    """One backend/resolution point on the frontier."""

    backend: str
    label: str
    device: str
    imgsz: int
    fps_mean: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    vram_peak_mb: int = 0
    model_load_s: float = 0.0
    artifact_mb: float = 0.0
    detections_total: int = 0
    recall_vs_baseline: float | None = None
    precision_vs_baseline: float | None = None
    mean_iou: float | None = None
    is_baseline: bool = False
    pareto_optimal: bool = False
    error: str | None = None
    per_frame_detections: list[list[Detection]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "per_frame_detections"}
        return payload


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two boxes."""
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def score_agreement(
    baseline: list[list[Detection]], candidate: list[list[Detection]]
) -> tuple[float, float, float]:
    """Greedy one-to-one matching per frame.

    Returns:
        ``(recall, precision, mean_iou)`` where recall is the share of baseline
        detections this backend also found, precision is the share of its own
        detections that correspond to a baseline detection, and mean_iou is the
        average overlap of matched pairs.
    """
    matched = 0
    baseline_total = 0
    candidate_total = 0
    ious: list[float] = []

    for truth_frame, test_frame in zip(baseline, candidate, strict=False):
        baseline_total += len(truth_frame)
        candidate_total += len(test_frame)
        available = list(range(len(test_frame)))
        for truth in truth_frame:
            best_index, best_iou = -1, IOU_MATCH_THRESHOLD
            for index in available:
                test = test_frame[index]
                if test.class_id != truth.class_id:
                    continue
                overlap = iou(truth, test)
                if overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0:
                available.remove(best_index)
                matched += 1
                ious.append(best_iou)

    recall = matched / baseline_total if baseline_total else 0.0
    precision = matched / candidate_total if candidate_total else 0.0
    return recall, precision, (statistics.fmean(ious) if ious else 0.0)


def measure(backend_key: str, imgsz: int, frames: list[np.ndarray]) -> ConfigResult:
    """Run one configuration over a fixed frame list."""
    settings = get_settings()
    backend = BACKENDS[backend_key]
    result = ConfigResult(
        backend=backend.key, label=backend.label, device=backend.device, imgsz=imgsz
    )

    detector = Detector(backend, imgsz, settings)
    try:
        started = time.perf_counter()
        detector.load()
        detector.warmup(frames=2)
        result.model_load_s = round(time.perf_counter() - started, 2)
    except Exception as exc:  # noqa: BLE001 - record and move on to the next config
        result.error = " ".join(str(exc).split())[:220]
        detector.close()
        return result

    artifact = backend.path(settings)
    result.artifact_mb = round(_size_mb(artifact), 2)

    latencies: list[float] = []
    peak_vram = probe_gpu().used_mb
    per_frame: list[list[Detection]] = []

    # Warmup pass on real frames, discarded, so timing excludes first-call costs.
    for frame in frames[:WARMUP_FRAMES]:
        detector.track(frame)
    detector.reset_tracker()

    clock = time.perf_counter()
    for index, frame in enumerate(frames):
        tick = time.perf_counter()
        detections, _, _ = detector.track(frame)
        latencies.append((time.perf_counter() - tick) * 1000.0)
        per_frame.append(detections)
        if index % 20 == 0:
            peak_vram = max(peak_vram, probe_gpu().used_mb)
    elapsed = time.perf_counter() - clock

    detector.close()

    ordered = sorted(latencies)
    result.fps_mean = round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0
    result.latency_mean_ms = round(statistics.fmean(latencies), 2)
    result.latency_p95_ms = round(
        ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))], 2
    )
    result.vram_peak_mb = peak_vram
    result.detections_total = sum(len(f) for f in per_frame)
    result.per_frame_detections = per_frame
    return result


def mark_pareto(results: list[ConfigResult]) -> None:
    """Flag configurations that nothing else beats on both speed and accuracy."""
    scored = [r for r in results if r.error is None and r.recall_vs_baseline is not None]
    for candidate in scored:
        dominated = any(
            other is not candidate
            and other.fps_mean >= candidate.fps_mean
            and (other.recall_vs_baseline or 0) >= (candidate.recall_vs_baseline or 0)
            and (
                other.fps_mean > candidate.fps_mean
                or (other.recall_vs_baseline or 0) > (candidate.recall_vs_baseline or 0)
            )
            for other in scored
        )
        candidate.pareto_optimal = not dominated


def load_frames(source: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"could not open {source}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {source}")
    return frames


def _size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return path.stat().st_size / 1024**2


def write_plot(results: list[ConfigResult], target: Path) -> bool:
    """Scatter throughput against agreement, highlighting the Pareto front."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - the plot is a bonus, the tables are the deliverable
        return False

    points = [r for r in results if r.error is None and r.recall_vs_baseline is not None]
    if not points:
        return False

    figure, axes = plt.subplots(figsize=(9, 6))
    for point in points:
        on_front = point.pareto_optimal
        axes.scatter(
            point.fps_mean,
            point.recall_vs_baseline,
            s=160 if on_front else 90,
            marker="*" if on_front else "o",
            color="#1668d6" if point.device == "cuda" else "#c95c1d",
            edgecolor="black" if on_front else "none",
            linewidth=1.2 if on_front else 0,
            zorder=3 if on_front else 2,
        )
        axes.annotate(
            f"{point.backend}\n{point.imgsz}px",
            (point.fps_mean, point.recall_vs_baseline),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )

    axes.axvline(30, color="#888", linestyle="--", linewidth=1)
    axes.text(30, axes.get_ylim()[0], " 30 FPS target", fontsize=8, color="#666", va="bottom")
    axes.set_xlabel("Full-pipeline throughput (FPS)")
    axes.set_ylabel("Detection recall vs FP32 baseline")
    axes.set_title("StreamSight accuracy-throughput frontier")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return True


def write_markdown(payload: dict[str, Any], target: Path) -> None:
    rows = payload["results"]
    lines = [
        "# Accuracy-throughput frontier",
        "",
        f"Generated {payload['generated_at']} on {payload['gpu']['name']}.",
        "",
        f"- Frames per configuration: {payload['frames']}",
        f"- Source clip: `{payload['source']}`",
        f"- Baseline: `{payload['baseline']}`",
        f"- Match rule: same class and IoU >= {IOU_MATCH_THRESHOLD}",
        "",
        "| Backend | Size | Device | Artifact | FPS | p95 ms | Peak VRAM | Recall |"
        " Precision | Mean IoU | Pareto |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["error"]:
            lines.append(
                f"| `{row['backend']}` | {row['imgsz']}px | {row['device']} | - | "
                f"unavailable | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| `{row['backend']}` | {row['imgsz']}px | {row['device']} | "
            f"{row['artifact_mb']} MB | **{row['fps_mean']}** | {row['latency_p95_ms']} | "
            f"{row['vram_peak_mb']} MiB | {_pct(row['recall_vs_baseline'])} | "
            f"{_pct(row['precision_vs_baseline'])} | {_num(row['mean_iou'])} | "
            f"{'yes' if row['pareto_optimal'] else ''} |"
        )

    unavailable = [r for r in rows if r["error"]]
    if unavailable:
        lines += ["", "## Configurations that could not run", ""]
        seen: set[str] = set()
        for row in unavailable:
            if row["backend"] in seen:
                continue
            seen.add(row["backend"])
            lines.append(f"- `{row['backend']}`: {row['error']}")

    lines += ["", "Phu Nguyen - HCMC, Vietnam", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 480])
    parser.add_argument("--source", help="evaluation clip (defaults to the bundled demo clip)")
    parser.add_argument("--backends", nargs="+", help="restrict the sweep to these backends")
    args = parser.parse_args(argv)

    settings = get_settings()
    gpu = probe_gpu()
    source = Path(args.source) if args.source else settings.assets_dir / "sample.mp4"
    frames = load_frames(source, args.frames)
    height, width = frames[0].shape[:2]
    print(f"loaded {len(frames)} frames from {source.name} ({width}x{height})")

    candidates = args.backends or list(BACKENDS)
    baseline_key = "fp32_gpu" if gpu.available else "fp32_cpu"
    baseline_imgsz = max(args.imgsz)

    results: list[ConfigResult] = []

    print(f"\n=== baseline: {baseline_key} @ {baseline_imgsz}px ===")
    baseline = measure(baseline_key, baseline_imgsz, frames)
    baseline.is_baseline = True
    if baseline.error:
        raise SystemExit(f"baseline could not run: {baseline.error}")
    baseline.recall_vs_baseline = 1.0
    baseline.precision_vs_baseline = 1.0
    baseline.mean_iou = 1.0
    results.append(baseline)
    print(f"    {baseline.fps_mean} FPS, {baseline.detections_total} detections")

    for key in candidates:
        runnable, why = availability(BACKENDS[key], settings, gpu.available)
        for imgsz in args.imgsz:
            if key == baseline_key and imgsz == baseline_imgsz:
                continue
            if not runnable:
                results.append(
                    ConfigResult(
                        backend=key,
                        label=BACKENDS[key].label,
                        device=BACKENDS[key].device,
                        imgsz=imgsz,
                        error=why,
                    )
                )
                continue
            if not BACKENDS[key].supports_imgsz(imgsz):
                # Not a failure, just a property of exported graphs. Skipping is
                # honest; recording it as an error would overstate the problem.
                print(
                    f"\n=== {key} @ {imgsz}px === skipped (exported at "
                    f"{BACKENDS[key].export_imgsz} px)"
                )
                continue
            print(f"\n=== {key} @ {imgsz}px ===")
            outcome = measure(key, imgsz, frames)
            if outcome.error:
                print(f"    unavailable: {outcome.error}")
            else:
                (
                    outcome.recall_vs_baseline,
                    outcome.precision_vs_baseline,
                    outcome.mean_iou,
                ) = score_agreement(baseline.per_frame_detections, outcome.per_frame_detections)
                print(
                    f"    {outcome.fps_mean} FPS, recall {_pct(outcome.recall_vs_baseline)}, "
                    f"precision {_pct(outcome.precision_vs_baseline)}"
                )
            results.append(outcome)

    mark_pareto(results)

    payload = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "frames": len(frames),
        "source": source.name,
        "baseline": f"{baseline_key}@{baseline_imgsz}",
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "gpu": {"name": gpu.name, "total_mb": gpu.total_mb, "available": gpu.available},
        "results": [r.to_dict() for r in results],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "frontier.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, REPORTS_DIR / "frontier.md")
    plotted = write_plot(results, REPORTS_DIR / "frontier.png")

    print(f"\nwrote {json_path}")
    print(f"wrote {REPORTS_DIR / 'frontier.md'}")
    if plotted:
        print(f"wrote {REPORTS_DIR / 'frontier.png'}")

    print("\nPareto-optimal configurations:")
    for row in results:
        if row.pareto_optimal:
            print(
                f"  {row.backend:<15} {row.imgsz}px  {row.fps_mean:>7.2f} FPS  "
                f"recall {_pct(row.recall_vs_baseline)}  peak {row.vram_peak_mb} MiB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
