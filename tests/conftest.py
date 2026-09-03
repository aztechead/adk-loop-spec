"""Shared fixtures: every test runs offline with dummy credentials."""

from __future__ import annotations

from pathlib import Path

import pytest

from devteam.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def dummy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construction checks these; no test may reach a real endpoint."""
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test")


@pytest.fixture()
def config() -> AppConfig:
    """The shipped config file, so tests validate what users actually run."""
    return load_config(REPO_ROOT / "config" / "devteam.yaml")
