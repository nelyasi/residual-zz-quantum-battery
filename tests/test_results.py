"""Every released result is regenerated from the raw shots."""

import subprocess
import sys

import pytest

from zzbattery.archive import compare_files

PRODUCTS = ("work-summary.json", "work-by-cycle.csv", "energy-budget.csv", "branch-populations.json", "robustness.json")


@pytest.fixture(scope="module")
def regenerated(root, tmp_path_factory):
    output = tmp_path_factory.mktemp("results")
    subprocess.run(
        [sys.executable, str(root / "scripts" / "reproduce_results.py"), "--output", str(output)], check=True
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_measurements.py"),
            "--output",
            str(output / "hardware-audit.json"),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize("name", PRODUCTS + ("hardware-audit.json",))
def test_agrees_with_release(root, regenerated, name):
    assert compare_files(regenerated / name, root / "results" / name) == []


def test_headline_numbers(root):
    import json

    summary = json.loads((root / "results" / "work-summary.json").read_text())
    assert round(summary["primary_gain"], 4) == 0.1659
    assert summary["equivalence_supported"] is True
