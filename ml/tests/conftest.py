"""Import path for the ML harnesses under test.

The scripts in ``ml/eval`` and ``ml/data/scripts`` are operator entry points, not
an installed package -- they are run as ``python ml/eval/eval_coco.py``, which
puts their own directory on ``sys.path`` for free. Tests get no such favour once
they live in a sibling directory, so the two source directories are added here
rather than repeated as a ``sys.path.insert`` at the top of every test module.

These tests are the guards against silently-zero mAP: a broken box or category
conversion produces a plausible-looking report rather than an error, so CI runs
them on every push.
"""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

for source_dir in (ML_ROOT / "eval", ML_ROOT / "data" / "scripts"):
    path = str(source_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
