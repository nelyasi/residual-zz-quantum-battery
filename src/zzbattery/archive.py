"""Reading and writing archived acquisitions.

Every acquisition folder contains ``circuits.qpy`` (compiled circuits),
``shots.json`` (one entry per circuit with the classical registers returned by the
sampler) and ``manifest.json`` (design metadata, job identifier and SHA-256 hashes).
"""

import csv
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


def sha256_file(path):
    """Return the SHA-256 hex digest of a file."""
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write JSON atomically with a stable layout (two-space indent, trailing newline)."""
    path = Path(path)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_run(folder):
    """Load a completed acquisition after checking its shot hash and circuit order."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("status") != "complete" or sha256_file(folder / "shots.json") != manifest["shots_sha256"]:
        raise ValueError(f"{folder}: incomplete acquisition or shot-record checksum mismatch.")
    rows = read_json(folder / "shots.json")
    if [row["metadata"] for row in rows] != manifest["circuits"]:
        raise ValueError(f"{folder}: measurement records do not follow the archived circuit order.")
    return manifest, rows


def verify_checksums(folder):
    """Check the circuit and shot hashes recorded in an acquisition manifest."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if sha256_file(folder / "circuits.qpy") != manifest["qpy_sha256"]:
        raise ValueError(f"{folder}: compiled-circuit checksum mismatch.")
    if sha256_file(folder / "shots.json") != manifest["shots_sha256"]:
        raise ValueError(f"{folder}: shot-record checksum mismatch.")
    return manifest


def compare_values(rebuilt, frozen, rtol=1e-12, path="value"):
    """Return the differences between two nested JSON-like values.

    Integers, strings and decisions must agree exactly. Floating-point values may
    differ in the last digits between linear-algebra libraries and are compared
    with a relative tolerance.
    """
    differences = []
    if isinstance(frozen, dict):
        if not isinstance(rebuilt, dict) or set(rebuilt) != set(frozen):
            return [f"{path}: keys differ"]
        for key in frozen:
            differences += compare_values(rebuilt[key], frozen[key], rtol, f"{path}.{key}")
    elif isinstance(frozen, list):
        if not isinstance(rebuilt, list) or len(rebuilt) != len(frozen):
            return [f"{path}: lengths differ"]
        for index, (left, right) in enumerate(zip(rebuilt, frozen, strict=False)):
            differences += compare_values(left, right, rtol, f"{path}[{index}]")
    elif isinstance(frozen, float):
        if not np.isclose(rebuilt, frozen, rtol=rtol, atol=1e-15):
            differences.append(f"{path}: {rebuilt!r} != {frozen!r}")
    elif rebuilt != frozen or type(rebuilt) is not type(frozen):
        differences.append(f"{path}: {rebuilt!r} != {frozen!r}")
    return differences


def _parse_csv(path):
    def convert(text):
        try:
            number = float(text)
        except ValueError:
            return text
        return int(text) if text.lstrip("-").isdigit() else number

    with Path(path).open(newline="") as stream:
        return [{key: convert(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def compare_files(rebuilt, released, rtol=1e-12):
    """Compare two JSON or CSV result files value by value (see :func:`compare_values`)."""
    loader = read_json if Path(released).suffix == ".json" else _parse_csv
    return compare_values(loader(rebuilt), loader(released), rtol, Path(released).name)
