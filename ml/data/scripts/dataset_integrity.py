"""Integrity helpers shared by the dataset download scripts.

Why one module: COCO arrives over keyless HTTP and MOT17 arrives as a zip the
user downloaded by hand after registering, but the *trust* story has to be
identical for both. Duplicating the hashing and manifest code across two
entry points is how the two halves would quietly drift apart.

Neither cocodataset.org nor motchallenge.net publishes a consumable per-file
checksum, so the policy is **pinned-on-first-download**: the first acquisition
that passes structural verification records its SHA256 into
``ml/data/manifests/``, and every later run compares against that record. This
detects an archive that *changed* under you -- a truncated resume, a silent
upstream replacement, a corrupted copy carried between machines. It cannot
detect a first download that was already wrong; that limitation is the price of
the upstream not publishing hashes and is stated in ``docs/DATASETS.md`` rather
than papered over here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "ml" / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_DIR = DATA_ROOT / "manifests"

#: 1 MiB. Large enough that hashing a 700 MB archive is I/O bound rather than
#: bound by Python loop overhead.
CHUNK_BYTES = 1024 * 1024


def human_bytes(count: int) -> str:
    """Format a byte count for humans watching a multi-hundred-megabyte download."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def utc_now() -> str:
    """Timestamp used in manifests, second precision -- sub-second is noise here."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """SHA256 of a file, streamed so a 700 MB archive never lands in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(name: str) -> Path:
    """Location of a dataset manifest, e.g. ``coco`` -> ml/data/manifests/coco.json."""
    return MANIFEST_DIR / f"{name}.json"


def load_manifest(name: str) -> dict[str, Any]:
    """Read a manifest, returning an empty skeleton when none has been pinned yet."""
    path = manifest_path(name)
    if not path.exists():
        return {"dataset": name, "archives": {}, "extracted": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("archives", {})
    data.setdefault("extracted", {})
    return data


def save_manifest(name: str, data: dict[str, Any]) -> Path:
    """Write a manifest atomically-ish; it is the only record of the pinned hashes."""
    path = manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = utc_now()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def pin_or_verify_sha256(
    manifest: dict[str, Any],
    key: str,
    archive: Path,
    *,
    source: str,
) -> tuple[str, bool]:
    """Compare an archive against the pinned hash, or pin it if this is the first run.

    Returns ``(digest, newly_pinned)``. A mismatch is fatal: continuing would
    extract bytes that provably differ from the ones the recorded metrics were
    produced against, which is worse than stopping.
    """
    print(f"hashing {archive.name} ({human_bytes(archive.stat().st_size)}) ...")
    digest = sha256_file(archive)
    recorded = manifest["archives"].get(key)

    if recorded and recorded.get("sha256") and recorded["sha256"] != digest:
        raise SystemExit(
            f"SHA256 mismatch for {archive.name}\n"
            f"  pinned:   {recorded['sha256']}\n"
            f"  on disk:  {digest}\n"
            f"delete {archive} and re-run, or delete the entry in "
            f"{manifest_path(manifest['dataset'])} if the change was intentional"
        )

    newly_pinned = recorded is None or not recorded.get("sha256")
    # Keep the original pin date: it is the provenance of the hash, and rewriting
    # it on every verification run would erase when the archive was first trusted.
    pinned_at = utc_now() if newly_pinned else recorded.get("pinned_at", utc_now())
    manifest["archives"][key] = {
        "source": source,
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "pinned_at": pinned_at,
    }
    return digest, newly_pinned


def count_files(directory: Path, suffixes: tuple[str, ...]) -> int:
    """Count files with the given suffixes, recursively and case-insensitively."""
    if not directory.is_dir():
        return 0
    lowered = tuple(s.lower() for s in suffixes)
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in lowered)


def _progress(copied: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {human_bytes(copied)}")
    else:
        sys.stdout.write(f"\r  {human_bytes(copied)} / {human_bytes(total)} ({copied / total:.0%})")
    sys.stdout.flush()


def download_with_resume(
    url: str, target: Path, *, timeout: int = 120, allow_plain_http: bool = False
) -> Path:
    """Download to ``target``, resuming a previous partial attempt when possible.

    A 780 MB archive over a domestic connection fails often enough that
    restarting from zero is a real cost, so bytes accumulate in a ``.part`` file
    and a ``Range`` request continues where the last attempt stopped. Servers
    that ignore ``Range`` answer 200 instead of 206, in which case the partial
    file is discarded rather than appended to -- appending to a full response is
    exactly how a corrupt archive with a plausible size gets created.

    Plain HTTP has to be opted into per call. One upstream here genuinely has no
    working HTTPS endpoint under its own hostname, and pretending otherwise by
    disabling certificate verification globally would weaken every other request
    in the process.
    """
    scheme = url.split("://", 1)[0].lower()
    if scheme == "http" and not allow_plain_http:
        raise SystemExit(f"refusing to download over plain HTTP: {url}")
    if scheme not in {"http", "https"}:
        raise SystemExit(f"unsupported URL scheme: {url}")

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    already = part.stat().st_size if part.exists() else 0

    request = urllib.request.Request(url)  # noqa: S310 - scheme checked above
    if already:
        print(f"resuming {target.name} at {human_bytes(already)}")
        request.add_header("Range", f"bytes={already}-")

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        resumed = getattr(response, "status", 200) == 206
        if already and not resumed:
            print("server ignored the range request - restarting the download")
            already = 0
        declared = int(response.headers.get("Content-Length") or 0)
        total = declared + already if resumed else declared
        copied = already
        with part.open("ab" if resumed else "wb") as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                copied += len(chunk)
                _progress(copied, total)
    sys.stdout.write("\n")

    if total and copied != total:
        raise SystemExit(
            f"{target.name}: transferred {copied} bytes but the server declared {total}"
        )
    part.replace(target)
    return target


def verify_size(archive: Path, *, expected: int, minimum: int) -> None:
    """Check an archive's length against the size seen upstream.

    Two thresholds with different force. Falling under ``minimum`` is fatal: at
    that point the file is a truncated transfer or an HTML error page served with
    a 200, and nothing downstream can succeed. Differing from ``expected`` is
    only a warning, because the upstream byte count is not published as a stable
    contract and a legitimate re-publish must not brick the script for someone
    who cannot edit the constant. The check with teeth is the pinned SHA256,
    which runs immediately after this one.
    """
    size = archive.stat().st_size
    if size < minimum:
        raise SystemExit(
            f"{archive.name} is {human_bytes(size)}, below the {human_bytes(minimum)} floor - "
            "the download is truncated or the URL served an error page"
        )
    if expected and size != expected:
        print(
            f"  warning: {archive.name} is {size} bytes, upstream served {expected} when this "
            "constant was measured - the archive may have been re-published"
        )
