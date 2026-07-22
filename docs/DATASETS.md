# Datasets

How COCO and MOT17 get onto this machine, how they are verified, and what they cost in disk.

Nothing in `ml/data/` is committed. The scripts below are the only source of truth for what a
correct copy looks like.

## Where each step runs

| Step | Runs | GPU | Notes |
|---|---|---|---|
| `download_coco.py` | **LOCAL** | none | ~1 GB over the network, CPU-bound hashing |
| `download_mot.py` | **LOCAL** | none | needs a zip you downloaded by hand first |
| `prepare_coco_subset.py` | **LOCAL** | none | parses the val2017 annotation file in memory |
| `split_dataset.py` | **LOCAL** | none | copies 550 images, writes label files |
| INT8 export using `calib.yaml` | **LOCAL** | yes | see `docs/QUANTIZATION.md` |
| Colab fine-tune on the same subset | **CLOUD** | T4 | upload the `processed/` tree, not `raw/` |

Every dataset step is CPU-only and stays inside the 4 GB VRAM budget by not touching the GPU at
all. The only reason to move any of this to the cloud is upload convenience for training.

## Pipeline

```powershell
python ml/data/scripts/download_coco.py            # val2017 images + annotations
python ml/data/scripts/prepare_coco_subset.py      # filter to 6 classes
python ml/data/scripts/split_dataset.py            # calib500 + val50 + calib.yaml

python ml/data/scripts/download_mot.py --zip C:/path/to/MOT17.zip
```

Each script is resumable and re-running it is cheap: work already on disk is detected and skipped.

### `download_coco.py`

Fetches `val2017.zip` and `annotations_trainval2017.zip` and extracts them into
`ml/data/raw/coco/`.

The URL is the path-style S3 form, `https://s3.amazonaws.com/images.cocodataset.org/...`, not the
`http://images.cocodataset.org/...` form printed on cocodataset.org. Verified 2026-07-22: the
bucket hostname has no valid certificate of its own, so HTTPS against it fails hostname
verification, while the S3 endpoint serves byte-identical content (`Content-Length` 815,585,330 and
252,907,541 respectively) over a properly verified connection. The plain-HTTP host remains an
automatic fallback for networks that block S3; taking it prints a warning, because over HTTP the
pinned hash is the only integrity guarantee left.

Flags:

| Flag | Effect |
|---|---|
| `--dest PATH` | extraction root, default `ml/data/raw/coco` |
| `--skip-images` | annotations only (~241 MB instead of ~1 GB) |
| `--verify-only` | check what is on disk, touch the network not at all; exit 1 if incomplete |
| `--keep-archives` | keep the zips instead of deleting them after a verified extract |

### `prepare_coco_subset.py`

Filters `instances_val2017.json` down to the six classes the PRD reports on — person, bicycle, car,
motorcycle, bus, truck — and writes:

- `ml/data/processed/coco_person_vehicle/instances_val2017_person_vehicle.json`
- `ml/data/processed/coco_person_vehicle/class_map.json`

Original COCO category ids are preserved (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
because pycocotools matches detections by category id. `class_map.json` carries the three
numbering systems side by side — subset index 0-5, COCO category id, and the 80-class index a
pretrained YOLO emits — so nothing downstream has to guess.

Images left with no person or vehicle are dropped. `--include-empty` keeps them.

The output is a pure function of the input file: no sampling, images sorted by id. The seeded step
is the next one.

### `split_dataset.py`

Draws two disjoint splits from the filtered subset with a seeded RNG (`--seed`, default 0) and
writes them as a real Ultralytics dataset:

```
ml/data/processed/coco_person_vehicle/
  images/calib500/   labels/calib500/     500 images, INT8 calibration
  images/val50/      labels/val50/         50 images, FP32-vs-INT8 parity
  calib.yaml                               train + val both point at calib500
  val50.yaml                               val points at val50
  splits.json                              exact file list, seed, per-split box counts
```

`calib.yaml` pointing its **val** split at calib500 is deliberate, not a typo: Ultralytics'
`export(int8=True, data=...)` reads calibration images from the val split and nowhere else. The
generated file says so in a comment.

Labels are written in YOLO format because Ultralytics builds a full dataset object even when it
only needs pixels; an images-only directory fails at load. Degenerate COCO boxes (zero width or
height) are dropped rather than being allowed to fail the export later.

`splits.json` contains no machine-local state, so two people running the same seed produce files
that diff clean.

### `download_mot.py`

MOT17 is registration-gated. There is no keyless URL, and this script does not pretend there is:
run it without `--zip` and it prints the manual steps and exits 2.

**Manual steps:**

1. Create a free account at <https://motchallenge.net/login/>.
2. Open <https://motchallenge.net/data/MOT17/> and accept the licence terms.
3. Download `MOT17.zip` (train + test, all three public detectors — several GB; the page lists the
   current size) or `MOT17Labels.zip` if you only need ground truth.
4. `python ml/data/scripts/download_mot.py --zip C:/path/to/MOT17.zip`

The zip is never modified. It is hashed, extracted into `ml/data/raw/mot/`, and every extracted
sequence is checked structurally: a directory counts as a sequence when it has both `seqinfo.ini`
and `img1/`, and its `seqLength` must match the frames actually on disk. Sequences are located by
structure rather than by name because the different MOT bundles nest their contents differently.

A bundle missing some of the seven MOT17 train scenes is reported as partial but accepted — a
single-detector or labels-only bundle is a legitimate thing to evaluate against.

## Integrity: SHA256 pinned on first download

Neither cocodataset.org nor motchallenge.net publishes a per-file checksum that a script can
consume. The policy is therefore **pinned-on-first-download**:

1. The first acquisition that passes every structural check has its SHA256 written to
   `ml/data/manifests/coco.json` or `ml/data/manifests/mot.json`.
2. Every later run recomputes the hash and compares it to that pin. A mismatch is fatal and names
   both hashes.

Commit the manifests. They are what makes "the same data" a checkable claim across machines and
across a re-download months later.

**What this does and does not buy you.** It detects an archive that changed under you: a truncated
resume, a corrupted copy carried on a USB stick, a silent upstream replacement. It does **not**
authenticate the first download — there is no published hash to compare that one against. Anyone
claiming otherwise about a COCO download is overstating it. The other checks are what guard the
first fetch:

| Check | Catches |
|---|---|
| transferred bytes vs `Content-Length` | interrupted transfer |
| size floor (700 MB / 200 MB) | HTML error page served with a 200, truncation |
| size vs the length measured upstream | re-published archive (warning, not fatal) |
| zip member CRC, validated during extraction | corruption; the Python equivalent of `unzip` returning 0 |
| member paths checked before writing | a zip that would write outside the destination |
| 5000 `.jpg` + `instances_val2017.json` present | partial or interrupted extract |
| MOT `seqLength` vs frames on disk | partial or interrupted extract |

The size floors are floors, not equality gates. The exact upstream byte count is not a published
contract, so a legitimate re-publish produces a warning rather than a script nobody can run without
editing a constant.

## Disk

| Item | Size | Notes |
|---|---|---|
| `val2017.zip` | 778 MiB | deleted after a verified extract unless `--keep-archives` |
| `annotations_trainval2017.zip` | 241 MiB | same |
| `raw/coco/val2017/` | ≈ the archive | 5000 JPEGs, already compressed |
| `raw/coco/annotations/` | ~1 GB | mostly `instances_train2017.json`, which this project does not read |
| `processed/coco_person_vehicle/` | ~100 MB | 550 copied images plus labels |
| `raw/mot/` | several GB | depends on which bundle you downloaded |

Budget roughly **3 GB free for COCO** (peak, while the archives still exist) and whatever the MOT
bundle needs on top. The archive sizes above were measured by HEAD request on 2026-07-22; the
extracted sizes are estimates.

`--skip-images` is worth knowing about: if you only need the annotations — to inspect the class
distribution, or to regenerate the subset — it turns a 1 GB download into a 241 MB one.

---

Phu Nguyen — HCMC, VN
