"""The code standards this repository holds itself to, checked mechanically."""

import re

from tests.conftest import REPO_ROOT

PACKAGES = (REPO_ROOT / "src", REPO_ROOT / "tests")


def test_package_inits_hold_only_english_comments() -> None:
    """An ``__init__.py`` describes its package in comments and holds no code."""
    inits = [path for root in PACKAGES for path in root.rglob("__init__.py")]
    assert inits, "no packages found"
    for path in inits:
        lines = path.read_text(encoding="utf-8").splitlines()
        code = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        assert not code, f"{path.relative_to(REPO_ROOT)} holds code: {code}"
        assert any(line.startswith("# ") for line in lines), f"{path} lacks a describing comment"


def test_only_absolute_imports() -> None:
    """Every import names its package in full; ruff's TID252 enforces the same rule."""
    relative = re.compile(r"^\s*from\s+\.")
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for root in PACKAGES
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if relative.match(line)
    ]
    assert offenders == []


def test_the_server_is_fastapi_not_starlette() -> None:
    """FastAPI is the web framework; no module reaches under it for Starlette."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if re.search(
            r"^\s*(from|import)\s+starlette", path.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    assert offenders == []
