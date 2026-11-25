from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def data(root):
    return root / "data" / "phoenix-2026-11-22-matched-feedback"
