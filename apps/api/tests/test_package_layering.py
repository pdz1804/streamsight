"""The package layering is a claim in ARCHITECTURE.md; this makes it enforceable.

`app` is split into packages that are meant to form a strict order: a package may
import from packages below it and never from one above. That property is what
makes the dependency direction readable from the import lines, and it is what
stops an import cycle before it exists.

Prose in a document cannot fail a build. This can. It parses the imports rather
than executing them, so a violation is reported as a layering error rather than
as an ImportError at some unrelated call site.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1] / "app"

#: Low to high. A package may import from anything earlier in this tuple.
#:
#: `vision` sits below `inference` because the detector parses ByteTrack results
#: (`inference/detector.py` imports `vision/tracker.py`), not the other way round.
LAYERS: tuple[str, ...] = ("core", "telemetry", "vision", "inference", "streaming", "routers")

#: Modules directly under `app/` compose the application and may import anything.
TOP_LEVEL_MODULES = {"main", "dependencies"}

#: `dependencies.py` is the DI seam FastAPI's own layout puts at the package root,
#: and the routers are what consume it. Every other module must reach the runtime
#: through its own layer rather than through the wiring.
TOP_LEVEL_IMPORTERS = {"dependencies": {"routers"}}


def _package_of(module: Path) -> str | None:
    """The layer a module belongs to, or None if it sits at the package root."""
    relative = module.relative_to(APP_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else None


def _cross_package_imports(module: Path) -> list[tuple[str, int]]:
    """Every `from ..other import x` in `module`, as (package, line number).

    Only level-2 relative imports cross a package boundary. Level 1 is a sibling
    inside the same package and is always allowed.
    """
    # utf-8-sig, not utf-8: some sources carry a byte-order mark, which Python
    # itself accepts in source but which `ast.parse` rejects as a stray U+FEFF.
    tree = ast.parse(module.read_text(encoding="utf-8-sig"), filename=str(module))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
            found.append((node.module.split(".")[0], node.lineno))
    return found


def _modules() -> list[Path]:
    return sorted(p for p in APP_ROOT.rglob("*.py") if p.name != "__init__.py")


def test_every_package_is_declared():
    """A new package must be placed in the order deliberately, not silently."""
    packages = {p.name for p in APP_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert packages == set(LAYERS), (
        f"packages on disk {sorted(packages)} do not match the declared layering "
        f"{sorted(LAYERS)}. Add the new package to LAYERS at the right height."
    )


def test_top_level_modules_are_only_the_composition_root():
    found = {p.stem for p in APP_ROOT.glob("*.py") if p.stem != "__init__"}
    assert found == TOP_LEVEL_MODULES, (
        f"unexpected module(s) at the package root: {sorted(found - TOP_LEVEL_MODULES)}. "
        "Everything that is not application composition belongs in a layer."
    )


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(APP_ROOT)))
def test_imports_never_point_up_the_layering(module: Path):
    importer = _package_of(module)
    if importer is None:
        return  # main.py and dependencies.py compose everything; nothing is above them.

    for imported, lineno in _cross_package_imports(module):
        where = f"{module.relative_to(APP_ROOT)}:{lineno}"

        if imported in TOP_LEVEL_MODULES:
            allowed = TOP_LEVEL_IMPORTERS.get(imported, set())
            assert importer in allowed, (
                f"{where} imports the root module {imported!r}, which only "
                f"{sorted(allowed) or 'nothing'} may import. Reaching the composition root "
                "from inside a layer inverts the dependency direction."
            )
            continue

        assert imported in LAYERS, f"{where} imports unknown package {imported!r}"
        assert LAYERS.index(imported) < LAYERS.index(importer), (
            f"{where} imports {imported!r}, which sits at or above {importer!r} in the "
            f"layering {LAYERS}. Either the import is wrong or the order in LAYERS "
            "(and ARCHITECTURE.md) needs to change deliberately."
        )
