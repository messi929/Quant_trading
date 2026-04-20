"""Shared pytest fixtures for V3 tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_save_dir(tmp_path) -> Path:
    """Isolated saved_models directory per test."""
    d = tmp_path / "saved_models"
    d.mkdir()
    return d


@pytest.fixture
def paper_account_fresh(tmp_save_dir) -> Path:
    """Empty paper account file for tests that need a clean broker state."""
    path = tmp_save_dir / "paper_account.json"
    data = {"cash_krw": 100_000_000, "positions": [], "trade_history": []}
    path.write_text(json.dumps(data))
    return path
