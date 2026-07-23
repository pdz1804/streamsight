"""Guards on the two pieces of directory layout the service actually depends on.

``Settings`` resolves weights, the demo clip, the SQLite log and ``.env`` from
``REPO_ROOT``, which is computed by counting parent directories up from
``app/core/config.py``. That count is invisible coupling: move the module one
level and every default path silently points somewhere that does not exist, and
the failure surfaces at runtime as a missing-weights error rather than here.

The Dockerfile says the same thing in a comment ("the repo layout is
load-bearing"). A comment does not fail a build; this does.
"""

from __future__ import annotations

from pathlib import Path

from app.core import config as config_module
from app.core.config import REPO_ROOT, Settings


def test_repo_root_resolves_to_something_that_looks_like_the_repo():
    for marker in ("ml", "apps", "pyproject.toml"):
        assert (REPO_ROOT / marker).exists(), (
            f"REPO_ROOT resolved to {REPO_ROOT}, which has no {marker!r}. "
            "app/core/config.py most likely moved without its parents[] index "
            "being updated."
        )


def test_repo_root_is_the_ancestor_config_actually_lives_under():
    """Catches the off-by-one that a marker check alone would miss.

    A too-shallow count can still land on a directory containing ``ml`` and
    ``apps`` if the repo is nested inside a similarly shaped tree, so pin the
    relationship to this file rather than to its contents.
    """
    here = Path(config_module.__file__).resolve()
    assert here.parent.name == "core"
    assert here.parents[4] == REPO_ROOT


def test_default_asset_paths_land_inside_the_repo():
    """Reads the declared defaults, not a constructed `Settings()`.

    `Settings()` merges `.env` and the `STREAMSIGHT_MODELS_DIR` /`_DATA_DIR` /
    `_ASSETS_DIR` overrides, which are documented and are exactly how a
    side-by-side A/B run points a second checkout at this one's weights. Asserting
    on an instance would turn that supported configuration into a red layout
    guard, blaming the directory structure for something the operator chose. The
    field defaults are what REPO_ROOT actually feeds, so they are what this checks.
    """
    defaults = {
        name: Settings.model_fields[name].default
        for name in ("models_dir", "data_dir", "assets_dir")
    }
    for name, path in defaults.items():
        assert isinstance(path, Path), f"{name} default is {path!r}, not a Path"
        assert REPO_ROOT in path.parents, f"default {name} ({path}) escapes the repo root"
