"""Fetch the pretrained weights and the bundled demo clip.

Neither artifact belongs in git -- weights are ~5 MB of binary that Ultralytics
already hosts, and the clip is third-party footage. Keeping this script as the
single source of truth means a fresh clone is one command away from running.

Usage:
    python ml/scripts/fetch_assets.py [--skip-video]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "ml" / "models" / "weights"
ASSETS_DIR = REPO_ROOT / "apps" / "api" / "assets"

WEIGHTS_NAME = "yolo11n.pt"

#: Demo and evaluation footage: 1080p, densely populated (~17 people per frame),
#: which exercises identity association far harder than sparse street footage
#: where the tracker rarely has to disambiguate anything.
SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4"

#: Calibration footage, deliberately *different* from the demo clip. Quantizing
#: on the same video used to report accuracy would flatter INT8: the activation
#: ranges would be fitted to the exact frames under test. Different scene,
#: different camera, same task.
CALIBRATION_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def fetch_weights() -> Path:
    """Download YOLO11n via Ultralytics and place it under ml/models/weights."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_DIR / WEIGHTS_NAME
    if target.exists():
        print(f"weights already present: {target}")
        return target

    from ultralytics import YOLO

    # Ultralytics downloads into the current working directory, so pull it there
    # and move it into place rather than fighting the library's path handling.
    YOLO(WEIGHTS_NAME)
    downloaded = Path.cwd() / WEIGHTS_NAME
    if downloaded.exists() and downloaded != target:
        shutil.move(str(downloaded), str(target))
    if not target.exists():
        raise SystemExit(f"could not locate downloaded weights at {downloaded}")
    print(f"weights ready: {target}")
    return target


def _download(url: str, target: Path, label: str) -> Path | None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"{label} already present: {target}")
        return target
    print(f"downloading {label} from {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            target.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - the app still runs without bundled media
        print(f"could not download {label}: {exc}", file=sys.stderr)
        return None
    print(f"{label} ready: {target} ({target.stat().st_size / 1024**2:.1f} MB)")
    return target


def fetch_sample_video() -> Path | None:
    """Download the demo clip used as the default streaming source."""
    result = _download(SAMPLE_VIDEO_URL, ASSETS_DIR / "sample.mp4", "demo clip")
    if result is None:
        print("upload your own video in the UI instead.", file=sys.stderr)
    return result


def fetch_calibration_video() -> Path | None:
    """Download the held-out clip used for INT8 calibration."""
    return _download(CALIBRATION_VIDEO_URL, ASSETS_DIR / "calibration.avi", "calibration clip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-video", action="store_true", help="only fetch model weights")
    args = parser.parse_args()

    fetch_weights()
    if not args.skip_video:
        fetch_sample_video()
        fetch_calibration_video()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
