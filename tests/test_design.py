"""The circuit design rebuilt from the seeds matches the archived manifests."""

import pytest

from zzbattery.archive import read_json
from zzbattery.circuits import calibration_circuits, experiment_batches


def test_calibration_design(data):
    manifest = read_json(data / "calibration" / "manifest.json")
    assert [circuit.metadata for circuit in calibration_circuits()] == manifest["circuits"]


@pytest.mark.slow
def test_measurement_design(data):
    calibration = read_json(data / "calibration" / "frozen-calibration.json")
    batches = list(experiment_batches(calibration, gate_samples=1))
    folders = sorted((data / "measurement").glob("batch-*"))
    assert len(batches) == len(folders) == 4
    for circuits, folder in zip(batches, folders, strict=False):
        assert [circuit.metadata for circuit in circuits] == read_json(folder / "manifest.json")["circuits"]
