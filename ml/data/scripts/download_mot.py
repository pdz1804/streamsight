"""Verify and extract a user-supplied MOT17 archive.

MOTChallenge puts MOT17 behind a registration form and a per-account download
link, so there is no keyless URL to fetch and this script does not pretend
otherwise: without ``--zip`` it prints the manual steps and exits non-zero. The
alternative -- scraping the download page or shipping someone's session cookie
-- would break the moment the site changes and would violate the terms the user
agreed to when registering.

What it does do is make the hand-carried zip trustworthy in the same way the
COCO archives are: extraction validates every member's CRC, the extracted tree
is checked for real MOT sequence structure (``seqinfo.ini`` + ``img1/``), and
the SHA256 is pinned into ``ml/data/manifests/mot.json`` on first use so the
same archive can be re-identified on another machine or after a re-download.

The expected sequence list is treated as a *warning*, not a gate: MOTChallenge
distributes MOT17 as several different bundles (all detectors, single detector,
train-only), and rejecting a valid bundle because it is not the largest one
would be wrong.

Usage:
    python ml/data/scripts/download_mot.py --zip D:/downloads/MOT17.zip
    python ml/data/scripts/download_mot.py --verify-only
"""

from __future__ import annotations

import argparse
import configparser
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_integrity import (
    RAW_DIR,
    human_bytes,
    load_manifest,
    pin_or_verify_sha256,
    save_manifest,
)

MANIFEST_NAME = "mot"

DOWNLOAD_PAGE = "https://motchallenge.net/data/MOT17/"
REGISTRATION_PAGE = "https://motchallenge.net/login/"

#: The seven base sequences in the MOT17 train split. Each is published three
#: times, once per public detector (DPM, FRCNN, SDP), so a full train bundle has
#: 21 directories for 7 distinct scenes.
EXPECTED_TRAIN_SEQUENCES = (
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
)

MANUAL_STEPS = f"""
MOT17 is registration-gated and cannot be downloaded without an account.

  1. Create a free account at {REGISTRATION_PAGE}
  2. Open {DOWNLOAD_PAGE} and accept the licence terms
  3. Download "MOT17.zip" (several GB: train+test, all three public
     detectors) or "MOT17Labels.zip" if you only need the ground truth.
     The page lists the current size of each bundle.
  4. Re-run this script pointing at the file you downloaded:

     python ml/data/scripts/download_mot.py --zip C:/path/to/MOT17.zip

The zip is never modified; it is verified, hashed and extracted into
ml/data/raw/mot/.
""".strip()


def find_sequences(root: Path) -> list[Path]:
    """Directories that look like real MOT sequences, wherever they sit in the tree.

    The search is structural rather than name-based because the zips nest their
    content differently (``MOT17/train/MOT17-02-DPM`` in one bundle, ``train/``
    at the root in another). A directory counts as a sequence when it carries
    the two things every MOT sequence has: ``seqinfo.ini`` and ``img1/``.
    """
    if not root.is_dir():
        return []
    found = [
        candidate.parent
        for candidate in root.rglob("seqinfo.ini")
        if (candidate.parent / "img1").is_dir()
    ]
    return sorted(found)


def describe_sequence(sequence: Path) -> dict[str, Any]:
    """Frame count and label availability for one sequence.

    ``seqLength`` is read from ``seqinfo.ini`` and compared against the images on
    disk: an interrupted extract shows up here as a short sequence long before it
    shows up as a nonsensical MOTA.
    """
    parser = configparser.ConfigParser()
    parser.read(sequence / "seqinfo.ini", encoding="utf-8")
    declared = parser.getint("Sequence", "seqLength", fallback=0)
    actual = sum(1 for p in (sequence / "img1").iterdir() if p.suffix.lower() == ".jpg")
    return {
        "name": sequence.name,
        "frames_declared": declared,
        "frames_on_disk": actual,
        "has_gt": (sequence / "gt" / "gt.txt").exists(),
        "has_det": (sequence / "det" / "det.txt").exists(),
        "complete": declared == 0 or declared == actual,
    }


def extract(archive: Path, dest: Path) -> int:
    """Extract the MOT zip, refusing members that would escape ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        for name in members:
            if not (dest / name).resolve().is_relative_to(dest.resolve()):
                raise SystemExit(f"{archive.name} contains an unsafe member path: {name}")
        print(f"extracting {len(members)} members from {archive.name} ...")
        bundle.extractall(dest)
    return len(members)


def report(dest: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Print sequence-level verification and return whether the tree is usable."""
    sequences = find_sequences(dest)
    if not sequences:
        print(f"  [FAIL] no MOT sequences found under {dest}")
        return False, []

    summaries = [describe_sequence(s) for s in sequences]
    for summary in summaries:
        status = "ok" if summary["complete"] else "FAIL"
        labels = "gt" if summary["has_gt"] else ("det-only" if summary["has_det"] else "no labels")
        print(
            f"  [{status}] {summary['name']}: "
            f"{summary['frames_on_disk']}/{summary['frames_declared'] or '?'} frames, {labels}"
        )

    bases = {name[:8] for name in (s["name"] for s in summaries)}
    missing = [s for s in EXPECTED_TRAIN_SEQUENCES if s not in bases]
    if missing:
        # Not fatal: partial bundles are a legitimate thing to evaluate on.
        print(f"  note: bundle does not contain {', '.join(missing)} - partial MOT17")

    if not any(s["has_gt"] for s in summaries):
        print("  note: no gt/gt.txt anywhere - this bundle cannot be scored, only tracked")

    return all(s["complete"] for s in summaries), summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip", type=Path, help="path to the MOT17 zip you downloaded manually")
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR / "mot",
        help="extraction root (default: ml/data/raw/mot)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check the already-extracted tree without extracting anything",
    )
    args = parser.parse_args(argv)

    dest: Path = args.dest
    manifest = load_manifest(MANIFEST_NAME)
    manifest["source"] = {"page": DOWNLOAD_PAGE, "acquisition": "manual, registration required"}

    if args.verify_only:
        print(f"verifying {dest}")
        ok, summaries = report(dest)
        print(f"{len(summaries)} sequence(s) found")
        return 0 if ok and summaries else 1

    if args.zip is None:
        print(MANUAL_STEPS, file=sys.stderr)
        return 2

    archive: Path = args.zip
    if not archive.is_file():
        raise SystemExit(f"no such file: {archive}")
    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"{archive} is not a zip archive")

    print(f"using {archive} ({human_bytes(archive.stat().st_size)})")
    _, newly_pinned = pin_or_verify_sha256(
        manifest, archive.name, archive, source="manual download from " + DOWNLOAD_PAGE
    )
    print(f"  sha256 {'pinned' if newly_pinned else 'matches the pin'}")

    members = extract(archive, dest)
    ok, summaries = report(dest)
    if not ok:
        raise SystemExit("extracted sequences are incomplete - re-download the archive")

    manifest["extracted"][archive.name] = {
        "members": members,
        "path": str(dest),
        "sequences": summaries,
    }
    path = save_manifest(MANIFEST_NAME, manifest)
    print(f"{len(summaries)} sequence(s) ready under {dest}")
    print(f"manifest written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
