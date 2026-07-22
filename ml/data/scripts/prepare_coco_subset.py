"""Filter COCO val2017 down to the six person+vehicle classes StreamSight reports on.

Why a filtered annotation file rather than filtering at eval time: the PRD's
accuracy gate is a mAP *drop* between FP32 and INT8 on the same class set, and
the only way that number means anything is if both sides are scored against
identical ground truth. Baking the subset into a file makes the class set an
artifact that can be diffed and re-used, instead of a flag that two eval runs
can silently disagree on.

The original COCO category ids are preserved (person=1, bicycle=2, car=3,
motorcycle=4, bus=6, truck=8) rather than renumbered 0-5, because pycocotools
matches detections to ground truth by category id, and a detector that outputs
the standard 80-class indices would otherwise need an undocumented second
mapping. The contiguous 0-5 index that YOLO training and the data yaml need is
written alongside, in ``class_map.json``, so both consumers get an explicit
table instead of an assumption.

No sampling happens here, so there is nothing to seed: the output is a pure
function of the input annotation file, and images are emitted in sorted id
order. The seeded step is ``split_dataset.py``.

Usage:
    python ml/data/scripts/prepare_coco_subset.py
    python ml/data/scripts/prepare_coco_subset.py --include-empty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_integrity import PROCESSED_DIR, RAW_DIR, utc_now

#: The six classes named in the PRD, in the order that defines the contiguous
#: index used by the data yaml. Order is part of the contract: changing it
#: invalidates every label file and every model trained against them.
SUBSET_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

#: Index of each class in the 80-class COCO ordering that pretrained YOLO models
#: emit. Recorded so the eval path can map model output to subset ids without
#: hard-coding a second copy of the COCO class list.
YOLO80_CLASS_IDS = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

DEFAULT_ANNOTATIONS = RAW_DIR / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_OUT_DIR = PROCESSED_DIR / "coco_person_vehicle"
SUBSET_FILENAME = "instances_val2017_person_vehicle.json"
CLASS_MAP_FILENAME = "class_map.json"


def build_class_map(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the six class names against the source file's own category table.

    Reading the ids out of the annotation file rather than hard-coding them means
    a future COCO release that renumbers categories fails loudly on a missing
    name instead of silently scoring the wrong objects.
    """
    by_name = {category["name"]: category for category in categories}
    missing = [name for name in SUBSET_CLASSES if name not in by_name]
    if missing:
        raise SystemExit(f"annotation file has no category named: {', '.join(missing)}")
    return [
        {
            "subset_index": index,
            "name": name,
            "coco_category_id": by_name[name]["id"],
            "yolo80_class_id": YOLO80_CLASS_IDS[name],
            "supercategory": by_name[name].get("supercategory", ""),
        }
        for index, name in enumerate(SUBSET_CLASSES)
    ]


def filter_annotations(
    source: dict[str, Any],
    class_map: list[dict[str, Any]],
    *,
    include_empty: bool,
) -> dict[str, Any]:
    """Keep only the six classes, and only images that still have something in them.

    Images left with zero annotations are dropped by default. They are valid
    negatives, but they dilute a 5000-image benchmark with frames that contain no
    person or vehicle at all, and mAP is computed per class over the images that
    contain that class -- so the only thing they change is runtime.
    """
    keep_ids = {entry["coco_category_id"] for entry in class_map}
    annotations = sorted(
        (a for a in source["annotations"] if a["category_id"] in keep_ids),
        key=lambda a: a["id"],
    )

    if include_empty:
        images = sorted(source["images"], key=lambda i: i["id"])
    else:
        annotated = {a["image_id"] for a in annotations}
        images = sorted(
            (i for i in source["images"] if i["id"] in annotated), key=lambda i: i["id"]
        )

    categories = [c for c in source["categories"] if c["id"] in keep_ids]
    categories.sort(key=lambda c: c["id"])

    info = dict(source.get("info", {}))
    info["description"] = "COCO val2017 filtered to the StreamSight person+vehicle class set"
    info["streamsight_generated_at"] = utc_now()

    return {
        "info": info,
        "licenses": source.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="source instances_val2017.json (default: ml/data/raw/coco/annotations/...)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default: ml/data/processed/coco_person_vehicle)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="keep images with no person or vehicle instead of dropping them",
    )
    args = parser.parse_args(argv)

    if not args.annotations.is_file():
        raise SystemExit(
            f"annotations not found: {args.annotations}\n"
            "run: python ml/data/scripts/download_coco.py"
        )

    print(f"reading {args.annotations}")
    source: dict[str, Any] = json.loads(args.annotations.read_text(encoding="utf-8"))

    class_map = build_class_map(source["categories"])
    subset = filter_annotations(source, class_map, include_empty=args.include_empty)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.out_dir / SUBSET_FILENAME
    subset_path.write_text(json.dumps(subset), encoding="utf-8")

    class_map_path = args.out_dir / CLASS_MAP_FILENAME
    class_map_path.write_text(
        json.dumps(
            {
                "source": str(args.annotations),
                "generated_at": utc_now(),
                "classes": class_map,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    per_class = {entry["name"]: 0 for entry in class_map}
    id_to_name = {entry["coco_category_id"]: entry["name"] for entry in class_map}
    for annotation in subset["annotations"]:
        per_class[id_to_name[annotation["category_id"]]] += 1

    print(
        f"kept {len(subset['images'])}/{len(source['images'])} images and "
        f"{len(subset['annotations'])}/{len(source['annotations'])} annotations"
    )
    for name, count in per_class.items():
        print(f"  {name:<12} {count}")
    print(f"wrote {subset_path}")
    print(f"wrote {class_map_path}")
    print("next: python ml/data/scripts/split_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
