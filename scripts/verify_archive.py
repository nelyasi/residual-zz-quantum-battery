"""Verify the integrity of the archived acquisitions and the frozen calibration.

Checks, in order:

1. the SHA-256 hashes of every compiled-circuit and shot file against its manifest;
2. that every measurement batch was staged with the archived frozen calibration;
3. that the frozen calibration (extraction rule and predictions) is reconstructed
   from the calibration shots alone;
4. that the hardware qualification met its pre-specified 90% criterion.

Usage: python scripts/verify_archive.py [--data DATA_DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from zzbattery.archive import compare_values, load_run, read_json, sha256_file, verify_checksums
from zzbattery.calibration import build_calibration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "phoenix-2026-11-22-matched-feedback"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = args.data
    folders = [data / "qualification", data / "calibration", *sorted((data / "measurement").glob("batch-*"))]

    for folder in folders:
        manifest = verify_checksums(folder)
        print(f"  checksums  {folder.relative_to(data)}  job {manifest['job_id']}")

    frozen_path = data / "calibration" / "frozen-calibration.json"
    frozen_hash = sha256_file(frozen_path)
    for folder in sorted((data / "measurement").glob("batch-*")):
        if read_json(folder / "manifest.json")["calibration_sha256"] != frozen_hash:
            sys.exit(f"{folder.name} was not staged with the archived frozen calibration.")
    print("  every measurement batch was staged with the frozen calibration")

    differences = compare_values(build_calibration(data / "calibration"), read_json(frozen_path))
    if differences:
        sys.exit("The calibration shots do not reproduce the frozen calibration:\n" + "\n".join(differences[:20]))
    print("  the calibration shots reproduce the frozen extraction rule and predictions")

    _, rows = load_run(data / "qualification")
    for row in rows:
        meta = row["metadata"]
        if meta.get("role") == "qualification":
            score = np.mean([int(bit) == meta["expected"] for bit in row["battery_out"]])
            if score < 0.90:
                sys.exit(f"Qualification probe {meta['expected']} scored {score:.3f} < 0.90.")
            print(f"  qualification probe (expected {meta['expected']}): {score:.4f}")
    print("Archive verified.")


if __name__ == "__main__":
    main()
