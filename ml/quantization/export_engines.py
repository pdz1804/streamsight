"""Export YOLO11n to every deployment format the runtime knows about.

One script, one canonical route per format. All INT8 calibration goes through
Ultralytics' ``model.export(..., int8=True, data=<yaml>)``: calibration images are
supplied via the data yaml's split, never an ad-hoc image directory, because no
such Ultralytics export interface exists.

Formats:
    fp16_onnx      ONNX, half precision, GPU via onnxruntime
    int8_onnx_cpu  ONNX, 8-bit, CPU deployment
    openvino_cpu   OpenVINO IR, CPU deployment
    fp16_trt       TensorRT engine, half precision
    int8_trt       TensorRT engine, 8-bit, calibrated

Usage:
    python ml/quantization/export_engines.py --formats fp16_onnx openvino_cpu
    python ml/quantization/export_engines.py --all --data coco8.yaml

Calibration note: ``coco8.yaml`` (8 images) is a *bootstrap only* and its
accuracy is provisional. Phase 3's 500-image ``calib.yaml`` supersedes it; do not
quote accuracy numbers from a coco8-calibrated engine.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling: calibrate.py

from app.backends import BACKENDS  # noqa: E402
from app.config import get_settings  # noqa: E402

#: format key -> Ultralytics export kwargs. `imgsz` and `data` are filled in later.
#:
#: `fp16_onnx` passes `device=0` deliberately. Ultralytics silently drops
#: `half=True` when exporting ONNX from CPU, which yields an FP32 graph wearing
#: an FP16 filename. Verified by inspecting initializer dtypes after export.
EXPORT_SPECS: dict[str, dict[str, Any]] = {
    "fp16_onnx": {"format": "onnx", "half": True, "device": 0},
    "openvino_cpu": {"format": "openvino", "half": True},
    "fp16_trt": {"format": "engine", "half": True, "device": 0},
    "int8_trt": {"format": "engine", "half": False, "int8": True, "device": 0},
}

#: Formats whose export requires a CUDA device to be present.
GPU_ONLY = {"fp16_onnx", "fp16_trt", "int8_trt"}

#: Formats that consume calibration data.
NEEDS_CALIBRATION = {"int8_trt"}

#: Formats produced by ONNX Runtime rather than by Ultralytics.
ORT_QUANTIZED = {"int8_onnx_cpu"}

#: Every exportable format, in a stable order for the CLI and the summary table.
ALL_FORMATS: list[str] = sorted(set(EXPORT_SPECS) | ORT_QUANTIZED)


def export_one(fmt: str, weights: Path, imgsz: int, data: str, force: bool) -> dict[str, Any]:
    """Export a single format and move the artifact to its registry location."""
    from ultralytics import YOLO

    backend = BACKENDS[fmt]
    settings = get_settings()
    target = backend.path(settings)
    if target.exists() and not force:
        return {"format": fmt, "status": "skipped", "reason": "already exists", "path": str(target)}

    if fmt in ORT_QUANTIZED:
        return quantize_onnx_int8(weights, imgsz, target)

    kwargs = dict(EXPORT_SPECS[fmt])
    kwargs["imgsz"] = imgsz
    if fmt in NEEDS_CALIBRATION:
        kwargs["data"] = data

    print(f"\n--- exporting {fmt} ({kwargs}) ---")
    started = time.perf_counter()
    try:
        produced = Path(YOLO(str(weights)).export(**kwargs))
    except Exception as exc:  # noqa: BLE001 - one failed format must not stop the rest
        return {"format": fmt, "status": "failed", "reason": str(exc)[:400]}
    elapsed = time.perf_counter() - started

    target.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != target.resolve():
        if target.exists():
            _remove(target)
        shutil.move(str(produced), str(target))

    return {
        "format": fmt,
        "status": "ok",
        "path": str(target),
        "size_mb": round(_size_mb(target), 2),
        "export_s": round(elapsed, 1),
    }


def quantize_onnx_int8(weights: Path, imgsz: int, target: Path) -> dict[str, Any]:
    """Produce a genuinely 8-bit ONNX graph by delegating to ``calibrate.py``.

    There is exactly one INT8-ONNX route in this repo and it lives in
    ``ml/quantization/calibrate.py``: static QDQ quantization against held-out
    calibration footage, with the detection head kept in float. Reimplementing a
    second route here would produce a different model under the same filename
    depending on which script ran last.

    Dynamic quantization was tried first and rejected: it emits ``ConvInteger``,
    for which ONNX Runtime's CPU provider has no kernel, so the resulting model
    loads and then fails at the first inference. See docs/QUANTIZATION.md.
    """
    import calibrate

    print("\n--- exporting int8_onnx_cpu (static QDQ, see calibrate.py) ---")
    started = time.perf_counter()
    settings = get_settings()
    calibration_source = settings.assets_dir / "calibration.avi"
    if not calibration_source.exists():
        return {
            "format": "int8_onnx_cpu",
            "status": "failed",
            "reason": f"calibration clip missing at {calibration_source}; "
            "run ml/scripts/fetch_assets.py",
        }

    from ultralytics import YOLO

    try:
        fp32_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, half=False))
        summary = calibrate.quantize(
            fp32_path,
            target,
            calibration_source,
            calibrate.DEFAULT_CALIBRATION_FRAMES,
            imgsz,
        )
        fp32_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return {"format": "int8_onnx_cpu", "status": "failed", "reason": str(exc)[:400]}

    return {
        "format": "int8_onnx_cpu",
        "status": "ok",
        "path": str(target),
        "size_mb": summary["size_mb"],
        "export_s": round(time.perf_counter() - started, 1),
    }


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return path.stat().st_size / 1024**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--formats", nargs="+", choices=ALL_FORMATS, help="formats to export")
    parser.add_argument("--all", action="store_true", help="export every supported format")
    parser.add_argument(
        "--weights", help="source .pt weights (defaults to ml/models/weights/yolo11n.pt)"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--data",
        default="coco8.yaml",
        help="calibration data yaml for INT8 (its val split supplies the images)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-export even if the artifact exists"
    )
    args = parser.parse_args(argv)

    if not args.formats and not args.all:
        parser.error("pass --formats or --all")

    settings = get_settings()
    weights = Path(args.weights) if args.weights else settings.weights_dir / "yolo11n.pt"
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights} (run ml/scripts/fetch_assets.py)")

    formats = ALL_FORMATS if args.all else args.formats
    cuda = _cuda_available()
    results: list[dict[str, Any]] = []

    for fmt in formats:
        if fmt in GPU_ONLY and not cuda:
            results.append({"format": fmt, "status": "skipped", "reason": "no CUDA device"})
            continue
        results.append(export_one(fmt, weights, args.imgsz, args.data, args.force))

    print("\n=== export summary ===")
    for entry in results:
        line = f"{entry['format']:<15} {entry['status']}"
        if entry["status"] == "ok":
            line += f"  {entry['size_mb']} MB in {entry['export_s']}s  -> {entry['path']}"
        else:
            line += f"  ({entry.get('reason', '')})"
        print(line)

    return 0 if any(r["status"] == "ok" for r in results) else 1


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
