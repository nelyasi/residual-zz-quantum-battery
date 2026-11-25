"""Integrity of the archived acquisitions and of the frozen calibration."""

from zzbattery.archive import compare_values, read_json, sha256_file, verify_checksums
from zzbattery.calibration import build_calibration


def acquisitions(data):
    return [data / "qualification", data / "calibration", *sorted((data / "measurement").glob("batch-*"))]


def test_checksums(data):
    folders = acquisitions(data)
    assert len(folders) == 6
    for folder in folders:
        assert verify_checksums(folder)["status"] == "complete"


def test_batches_use_frozen_calibration(data):
    frozen = sha256_file(data / "calibration" / "frozen-calibration.json")
    for folder in sorted((data / "measurement").glob("batch-*")):
        assert read_json(folder / "manifest.json")["calibration_sha256"] == frozen


def test_calibration_is_reconstructed(data):
    rebuilt = build_calibration(data / "calibration")
    frozen = read_json(data / "calibration" / "frozen-calibration.json")
    assert compare_values(rebuilt, frozen) == []
    assert rebuilt["controller_table"] == frozen["controller_table"]


def test_comparison_detects_changes(data):
    frozen = read_json(data / "calibration" / "frozen-calibration.json")
    altered = read_json(data / "calibration" / "frozen-calibration.json")
    altered["controller_table"]["1"][0] ^= 1
    altered["predictions"]["4"]["battery_before"] *= 1 + 1e-9
    assert len(compare_values(altered, frozen)) == 2
