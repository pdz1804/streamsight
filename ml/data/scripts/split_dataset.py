"""Build the calibration and parity splits, plus the data yaml the exporters read.

Two splits, two different jobs:

* **calib500** feeds INT8 calibration. Ultralytics' ``export(int8=True, data=...)``
  reads calibration images from the yaml's *val* split, never from a loose
  directory, so ``calib.yaml`` deliberately points ``val`` at calib500. That
  looks wrong until you know the exporter's contract, which is why it is written
  as a comment into the generated file as well.
* **val50** is the parity split: small enough to run against every backend in a
  loop, disjoint from calib500 so INT8 is never scored on the frames its
  activation ranges were fitted to.

Both are drawn with a seeded RNG from the sorted image list, so the same seed on
any machine selects the same images. The splits are drawn from one sample of
``calib + val`` images and then cut, which keeps them disjoint by construction
rather than by a filtering step that could be skipped.

YOLO-format label files are written alongside the images because Ultralytics
builds a real dataset object even when it only needs pixels for calibration, and
an images-only tree fails at load time.

Usage:
    python ml/data/scripts/split_dataset.py
    python ml/data/scripts/split_dataset.py --calib 500 --val 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_integrity import PROCESSED_DIR, RAW_DIR, utc_now

DEFAULT_SUBSET_DIR = PROCESSED_DIR / "coco_person_vehicle"
DEFAULT_IMAGES_DIR = RAW_DIR / "coco" / "val2017"

DEFAULT_CALIB_COUNT = 500
DEFAULT_VAL_COUNT = 50
DEFAULT_SEED = 0


def load_subset(subset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the filtered annotations and the class map produced by prepare_coco_subset."""
    annotations_path = subset_dir / "instances_val2017_person_vehicle.json"
    class_map_path = subset_dir / "class_map.json"
    for path in (annotations_path, class_map_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}\nrun: python ml/data/scripts/prepare_coco_subset.py")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    classes = json.loads(class_map_path.read_text(encoding="utf-8"))["classes"]
    return annotations, classes


def to_yolo_line(annotation: dict[str, Any], image: dict[str, Any], class_index: int) -> str | None:
    """Convert one COCO box to a normalised YOLO row, or None if it is degenerate.

    COCO carries zero-area and out-of-frame boxes; Ultralytics rejects a label
    file containing them, so they are dropped here rather than being allowed to
    fail the export hours later.
    """
    x, y, width, height = annotation["bbox"]
    img_w, img_h = image["width"], image["height"]
    if width <= 0 or height <= 0 or img_w <= 0 or img_h <= 0:
        return None

    cx = min(max((x + width / 2) / img_w, 0.0), 1.0)
    cy = min(max((y + height / 2) / img_h, 0.0), 1.0)
    nw = min(max(width / img_w, 0.0), 1.0)
    nh = min(max(height / img_h, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return f"{class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def write_split(
    name: str,
    images: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    category_to_index: dict[int, int],
    *,
    source_images: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Copy images and write labels for one split; skip files already in place."""
    image_dir = out_dir / "images" / name
    label_dir = out_dir / "labels" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    boxes = 0
    for image in images:
        source = source_images / image["file_name"]
        if not source.is_file():
            raise SystemExit(
                f"image missing: {source}\nrun: python ml/data/scripts/download_coco.py"
            )
        target = image_dir / image["file_name"]
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, target)
            copied += 1

        lines = [
            line
            for annotation in annotations_by_image.get(image["id"], [])
            if (
                line := to_yolo_line(
                    annotation, image, category_to_index[annotation["category_id"]]
                )
            )
        ]
        boxes += len(lines)
        (label_dir / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    print(f"{name}: {len(images)} images ({copied} copied), {boxes} boxes")
    # `copied` is deliberately not returned: it depends on what was already on
    # disk, and splits.json has to stay a pure function of (seed, counts, input)
    # so two machines can diff it.
    return {"name": name, "images": len(images), "boxes": boxes}


def write_yaml(
    path: Path,
    *,
    root: Path,
    train_split: str,
    val_split: str,
    class_names: list[str],
    note: str,
) -> None:
    """Emit an Ultralytics dataset yaml by hand.

    Written as text rather than via ``yaml.safe_dump`` so the explanatory comment
    survives -- the next person to open ``calib.yaml`` will otherwise assume the
    val split is a mistake.
    """
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    path.write_text(
        f"# Generated by ml/data/scripts/split_dataset.py on {utc_now()}\n"
        f"# {note}\n"
        f"path: {root.as_posix()}\n"
        f"train: images/{train_split}\n"
        f"val: images/{val_split}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--calib", type=int, default=DEFAULT_CALIB_COUNT)
    parser.add_argument("--val", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    subset, classes = load_subset(args.subset_dir)
    images = sorted(subset["images"], key=lambda i: i["id"])
    wanted = args.calib + args.val
    if len(images) < wanted:
        raise SystemExit(f"subset has {len(images)} images, need {wanted}")

    # Sample once and cut, so calib and val cannot overlap however the counts change.
    selected = random.Random(args.seed).sample(images, wanted)  # noqa: S311 - reproducibility
    calib_images = selected[: args.calib]
    val_images = selected[args.calib :]

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in subset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    category_to_index = {entry["coco_category_id"]: entry["subset_index"] for entry in classes}
    class_names = [entry["name"] for entry in classes]

    calib_name = f"calib{args.calib}"
    val_name = f"val{args.val}"
    out_dir: Path = args.subset_dir

    splits = [
        write_split(
            calib_name,
            calib_images,
            annotations_by_image,
            category_to_index,
            source_images=args.images,
            out_dir=out_dir,
        ),
        write_split(
            val_name,
            val_images,
            annotations_by_image,
            category_to_index,
            source_images=args.images,
            out_dir=out_dir,
        ),
    ]

    write_yaml(
        out_dir / "calib.yaml",
        root=out_dir,
        train_split=calib_name,
        val_split=calib_name,
        class_names=class_names,
        note=(
            f"val points at {calib_name} on purpose: Ultralytics INT8 export reads "
            "calibration images from the val split."
        ),
    )
    write_yaml(
        out_dir / f"{val_name}.yaml",
        root=out_dir,
        train_split=calib_name,
        val_split=val_name,
        class_names=class_names,
        note=f"parity split, disjoint from {calib_name} by construction.",
    )

    manifest = {
        "generated_at": utc_now(),
        "seed": args.seed,
        "source_images": str(args.images),
        "source_annotations": str(args.subset_dir / "instances_val2017_person_vehicle.json"),
        "classes": class_names,
        "splits": {
            calib_name: [i["file_name"] for i in calib_images],
            val_name: [i["file_name"] for i in val_images],
        },
        "counts": {split["name"]: split for split in splits},
    }
    manifest_path = out_dir / "splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"next: python ml/quantization/export_engines.py --all --data {out_dir / 'calib.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
