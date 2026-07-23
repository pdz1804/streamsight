"""Download and verify COCO val2017 (images + annotations).

Source choice: the COCO archives live in the public S3 bucket
``images.cocodataset.org``, which needs no account, no API key and no client
library, so a fresh clone can fetch the evaluation data unattended.

The URL is the *path-style S3* form rather than the ``http://images.cocodataset.org/...``
one printed on cocodataset.org. Verified on 2026-07-22: the bucket's own
hostname has no valid certificate for HTTPS (the TLS name does not match), while
``https://s3.amazonaws.com/images.cocodataset.org/...`` serves byte-identical
content over a verified connection. Same bytes, real transport security. The
plain-HTTP hostname is kept as an automatic fallback for networks that block S3
directly, and taking it prints a warning, because over HTTP the only integrity
guarantee left is the pinned hash.

Integrity: COCO publishes no consumable per-file checksum, so this script pins
the SHA256 on the first successful download and compares against that pin on
every later run (see ``dataset_integrity`` for why that is the strongest
guarantee available here). Three independent checks have to pass before an
archive is trusted:

1. the transferred byte count matches the server's ``Content-Length``, clears a
   conservative floor and is compared against the length upstream served when
   these constants were measured -- this catches truncation and HTML error pages
   served with a 200;
2. extraction completes without error. ``zipfile`` validates each member's CRC
   as it reads it, so a clean extract is the Python equivalent of ``unzip``
   returning 0;
3. the extracted tree contains the expected members and file count (5000
   val2017 images, ``instances_val2017.json`` present).

Every step is resumable: a partial download continues via HTTP ``Range``, and an
already-extracted tree that passes verification is left alone.

Usage:
    python ml/data/scripts/download_coco.py
    python ml/data/scripts/download_coco.py --skip-images     # annotations only
    python ml/data/scripts/download_coco.py --verify-only     # no network access
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_integrity import (
    RAW_DIR,
    count_files,
    download_with_resume,
    human_bytes,
    load_manifest,
    pin_or_verify_sha256,
    save_manifest,
    verify_size,
)

MANIFEST_NAME = "coco"

#: Used to record extraction locations relative to the repo, so the committed
#: manifest does not carry one machine's absolute paths.
REPO_ROOT = Path(__file__).resolve().parents[3]

S3_PREFIX = "https://s3.amazonaws.com/images.cocodataset.org"
HTTP_PREFIX = "http://images.cocodataset.org"

IMAGES_PATH = "/zips/val2017.zip"
ANNOTATIONS_PATH = "/annotations/annotations_trainval2017.zip"

IMAGES_URL = S3_PREFIX + IMAGES_PATH
ANNOTATIONS_URL = S3_PREFIX + ANNOTATIONS_PATH

#: Content-Length served by both endpoints, measured by HEAD on 2026-07-22. A
#: difference is reported but not fatal; see ``verify_size``.
IMAGES_EXPECTED_BYTES = 815_585_330
ANNOTATIONS_EXPECTED_BYTES = 252_907_541

#: Hard floors. Anything smaller cannot be the archive whatever upstream does.
IMAGES_MIN_BYTES = 700 * 1024**2
ANNOTATIONS_MIN_BYTES = 200 * 1024**2

#: COCO val2017 has had exactly 5000 images since the 2017 split was published;
#: it is a fixed benchmark, so an inexact count means a broken extract.
VAL2017_IMAGE_COUNT = 5000

#: The only annotation file this project reads. The archive also carries the
#: train2017, captions and keypoints files, which are not required to be present
#: for the eval path to work.
REQUIRED_ANNOTATION = "instances_val2017.json"


def images_dir(dest: Path) -> Path:
    return dest / "val2017"


def annotations_dir(dest: Path) -> Path:
    return dest / "annotations"


def check_images(dest: Path) -> tuple[bool, str]:
    """Is the image tree complete?"""
    found = count_files(images_dir(dest), (".jpg",))
    ok = found == VAL2017_IMAGE_COUNT
    return ok, f"{found}/{VAL2017_IMAGE_COUNT} jpg files in {images_dir(dest)}"


def check_annotations(dest: Path) -> tuple[bool, str]:
    """Is the required annotation file present?"""
    target = annotations_dir(dest) / REQUIRED_ANNOTATION
    return target.exists(), f"{REQUIRED_ANNOTATION} {'present' if target.exists() else 'missing'}"


def extract(archive: Path, dest: Path) -> int:
    """Extract a zip into ``dest`` and return the number of members written.

    Members are checked for path traversal before anything is written: these
    archives come from a third party over the network, and ``extractall`` on
    Python 3.11 will happily follow an absolute or ``..`` member path.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        for name in members:
            resolved = (dest / name).resolve()
            if not resolved.is_relative_to(dest.resolve()):
                raise SystemExit(f"{archive.name} contains an unsafe member path: {name}")
        print(f"extracting {len(members)} members from {archive.name} ...")
        bundle.extractall(dest)
    return len(members)


def fetch(url_path: str, archive: Path) -> str:
    """Fetch over verified HTTPS, falling back to the plain-HTTP hostname.

    Returns the URL that actually served the bytes so the manifest records the
    real provenance rather than the one that was attempted first.
    """
    https_url = S3_PREFIX + url_path
    print(f"downloading {archive.name} from {https_url}")
    try:
        download_with_resume(https_url, archive)
    except OSError as exc:
        http_url = HTTP_PREFIX + url_path
        print(f"  HTTPS source failed ({exc}); falling back to {http_url}")
        print("  warning: plain HTTP - integrity rests entirely on the pinned SHA256 below")
        download_with_resume(http_url, archive, allow_plain_http=True)
        return http_url
    return https_url


def acquire(
    *,
    url_path: str,
    archive: Path,
    expected_bytes: int,
    minimum_bytes: int,
    manifest: dict[str, Any],
    key: str,
    dest: Path,
) -> None:
    """Download, verify and extract one archive, pinning its hash on the way."""
    if not archive.exists():
        url = fetch(url_path, archive)
    else:
        url = S3_PREFIX + url_path
        print(f"archive already present: {archive} ({human_bytes(archive.stat().st_size)})")

    verify_size(archive, expected=expected_bytes, minimum=minimum_bytes)
    _, newly_pinned = pin_or_verify_sha256(manifest, key, archive, source=url)
    print(f"  sha256 {'pinned' if newly_pinned else 'matches the pin'}")

    members = extract(archive, dest)
    # Recorded relative to the repo root: the manifest is committed as the
    # integrity pin, and an absolute path would bake one machine's layout into
    # a shared file. The field is informational -- nothing compares it -- so the
    # portable spelling costs nothing.
    try:
        location = dest.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        location = dest.resolve().as_posix()
    manifest["extracted"][key] = {"members": members, "path": location}


def report(dest: Path, *, skip_images: bool) -> bool:
    """Print the state of the extracted tree and say whether it is usable."""
    checks = [] if skip_images else [("images", check_images(dest))]
    checks.append(("annotations", check_annotations(dest)))
    ok = True
    for label, (passed, detail) in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}: {detail}")
        ok = ok and passed
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR / "coco",
        help="extraction root (default: ml/data/raw/coco)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="fetch annotations only (~241 MB instead of ~1 GB)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check what is already on disk and touch the network not at all",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep the zips after extraction instead of reclaiming the ~1 GB",
    )
    args = parser.parse_args(argv)

    dest: Path = args.dest
    manifest = load_manifest(MANIFEST_NAME)
    manifest["source"] = {"images": IMAGES_URL, "annotations": ANNOTATIONS_URL}

    if args.verify_only:
        print(f"verifying {dest}")
        ok = report(dest, skip_images=args.skip_images)
        pinned = {k: v.get("sha256") for k, v in manifest["archives"].items()}
        print(f"pinned archives: {pinned or 'none recorded yet'}")
        if not ok:
            print("dataset incomplete - run without --verify-only to fetch it")
        return 0 if ok else 1

    dest.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []

    if not args.skip_images:
        images_ok, detail = check_images(dest)
        if images_ok:
            print(f"images already extracted: {detail}")
        else:
            archive = dest / "val2017.zip"
            acquire(
                url_path=IMAGES_PATH,
                archive=archive,
                expected_bytes=IMAGES_EXPECTED_BYTES,
                minimum_bytes=IMAGES_MIN_BYTES,
                manifest=manifest,
                key="val2017.zip",
                dest=dest,
            )
            archives.append(archive)

    annotations_ok, detail = check_annotations(dest)
    if annotations_ok:
        print(f"annotations already extracted: {detail}")
    else:
        archive = dest / "annotations_trainval2017.zip"
        acquire(
            url_path=ANNOTATIONS_PATH,
            archive=archive,
            expected_bytes=ANNOTATIONS_EXPECTED_BYTES,
            minimum_bytes=ANNOTATIONS_MIN_BYTES,
            manifest=manifest,
            key="annotations_trainval2017.zip",
            dest=dest,
        )
        archives.append(archive)

    print("verifying extracted tree")
    ok = report(dest, skip_images=args.skip_images)
    if not ok:
        # Leave the archives in place: re-extracting is cheaper than re-downloading.
        raise SystemExit("extraction did not produce the expected files - archives kept")

    if not args.keep_archives:
        for archive in archives:
            archive.unlink(missing_ok=True)
            print(f"removed {archive.name} (use --keep-archives to retain)")

    path = save_manifest(MANIFEST_NAME, manifest)
    print(f"manifest written: {path}")
    print("next: python ml/data/scripts/prepare_coco_subset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
