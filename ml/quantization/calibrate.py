"""Static INT8 quantization of the ONNX detector, with real calibration.

Why static and not dynamic: ONNX Runtime's dynamic path rewrites convolutions to
``ConvInteger``, which the CPU provider has no kernel for, so a dynamically
quantized detector loads and then fails at the first inference. Static
quantization in QDQ format produces ``QLinearConv``, which is supported and
genuinely faster.

Calibration uses frames from real footage rather than random tensors: activation
ranges are only meaningful if the calibration distribution resembles what the
model will see. Preprocessing here mirrors Ultralytics exactly (letterbox to a
square, BGR to RGB, scale to 0-1, CHW) because a mismatch would collect ranges
for inputs the model never actually receives.

Usage:
    python ml/quantization/calibrate.py
    python ml/quantization/calibrate.py --frames 200 --source path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.core.config import get_settings  # noqa: E402

DEFAULT_CALIBRATION_FRAMES = 128


def letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio and pad to a square, as Ultralytics does."""
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # 114 is the YOLO pad value
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """BGR frame to a batched NCHW float32 tensor in 0-1."""
    padded = letterbox(frame, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]


def iter_calibration_frames(source: Path, limit: int, size: int) -> Iterator[np.ndarray]:
    """Yield evenly spaced frames from a clip, or images from a directory."""
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for path in images[:limit]:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                yield preprocess(frame, size)
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"could not open calibration source: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or limit
        # Spread samples across the whole clip so calibration does not overfit to
        # one scene; the opening seconds of most footage are unrepresentative.
        step = max(1, total // limit)
        index = 0
        emitted = 0
        while emitted < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                yield preprocess(frame, size)
                emitted += 1
            index += 1
    finally:
        capture.release()


def build_reader(source: Path, limit: int, size: int, input_name: str) -> Any:
    """Construct an ONNX Runtime CalibrationDataReader over the frames."""
    from onnxruntime.quantization import CalibrationDataReader

    class FrameCalibrationReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iterator = iter_calibration_frames(source, limit, size)
            self.consumed = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            tensor = next(self._iterator, None)
            if tensor is None:
                return None
            self.consumed += 1
            return {input_name: tensor}

        def rewind(self) -> None:
            self._iterator = iter_calibration_frames(source, limit, size)
            self.consumed = 0

    return FrameCalibrationReader()


def head_node_names(model: Any) -> list[str]:
    """Names of the nodes belonging to the final detection head.

    Quantizing the head is what turns a working detector into one that returns
    nothing: the box-regression branch decodes distributions into coordinates,
    and 8-bit activation ranges there collapse the geometry even though the
    backbone quantizes cleanly. Measured on this model, quantizing the head took
    recall against the FP32 baseline to zero.

    The head is identified structurally -- the highest ``/model.N/`` index in the
    graph -- so this keeps working if the architecture depth changes.
    """
    import re

    pattern = re.compile(r"/model\.(\d+)/")
    indices = {
        int(match.group(1))
        for node in model.graph.node
        if (match := pattern.search(node.name or ""))
    }
    if not indices:
        return []
    last = max(indices)
    prefix = f"/model.{last}/"
    return [node.name for node in model.graph.node if node.name and prefix in node.name]


def quantize(
    fp32_onnx: Path,
    target: Path,
    calibration_source: Path,
    frames: int,
    imgsz: int,
) -> dict[str, Any]:
    """Run static QDQ quantization and return a summary."""
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model = onnx.load(str(fp32_onnx))
    input_name = model.graph.input[0].name
    excluded = head_node_names(model)
    print(f"keeping {len(excluded)} detection-head nodes in float")

    # Static quantization needs inferred shapes and folded constants; skipping
    # this step is the usual cause of "tensor not found" during calibration.
    prepared = fp32_onnx.with_name(f"{fp32_onnx.stem}_prepared.onnx")
    quant_pre_process(str(fp32_onnx), str(prepared), skip_symbolic_shape=False)

    reader = build_reader(calibration_source, frames, imgsz, input_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(prepared),
        model_output=str(target),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=excluded,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    prepared.unlink(missing_ok=True)

    return {
        "output": str(target),
        "size_mb": round(target.stat().st_size / 1024**2, 2),
        "calibration_frames": reader.consumed,
        "calibration_source": str(calibration_source),
        "imgsz": imgsz,
        "head_nodes_kept_float": len(excluded),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", help="source .pt weights")
    parser.add_argument("--source", help="calibration video or image directory")
    parser.add_argument("--frames", type=int, default=DEFAULT_CALIBRATION_FRAMES)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", help="output .onnx path")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    settings = get_settings()
    weights = Path(args.weights) if args.weights else settings.weights_dir / "yolo11n.pt"
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights} (run ml/scripts/fetch_assets.py)")

    # Defaults to the held-out calibration clip, never the demo clip that
    # accuracy is reported on.
    source = Path(args.source) if args.source else settings.assets_dir / "calibration.avi"
    if not source.exists():
        raise SystemExit(f"calibration source not found: {source} (run ml/scripts/fetch_assets.py)")

    target = Path(args.out) if args.out else settings.engines_dir / "yolo11n_int8.onnx"

    print(f"exporting FP32 ONNX from {weights.name}")
    fp32 = Path(YOLO(str(weights)).export(format="onnx", imgsz=args.imgsz, half=False))

    print(f"calibrating on {args.frames} frames from {source.name}")
    summary = quantize(fp32, target, source, args.frames, args.imgsz)
    fp32.unlink(missing_ok=True)

    print(
        f"wrote {summary['output']} ({summary['size_mb']} MB) "
        f"from {summary['calibration_frames']} calibration frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
