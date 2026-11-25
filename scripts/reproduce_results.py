"""Regenerate every released result of the matched-feedback experiment from the raw shots.

Writes to OUTPUT (default ``build/results``):

    work-summary.json       work estimates, intervals and the plateau test (Table III, Sec. VIII B)
    work-by-cycle.csv       the rows of work-summary.json as a table
    energy-budget.csv       stored and final energies (Fig. 4)
    branch-populations.json populations grouped by the selected operation (Fig. 7)
    robustness.json         plateau interval with 100,000 resamples and the uncorrected gain

With ``--check`` the files are compared with the released ``results/``: integers, strings and
decisions exactly, floating-point values to a relative tolerance of 1e-12 (the last digits
can differ between linear-algebra libraries).

Usage: python scripts/reproduce_results.py [--output DIR] [--check]
"""

import argparse
import csv
import sys
from pathlib import Path

from zzbattery.analysis import (
    branch_populations,
    check_energy_budget,
    collect_frequencies,
    energy_budget,
    matched_work_report,
    robustness,
)
from zzbattery.archive import compare_files, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phoenix-2026-11-22-matched-feedback"
RELEASED = ROOT / "results"
PRODUCTS = (
    "work-summary.json",
    "work-by-cycle.csv",
    "energy-budget.csv",
    "branch-populations.json",
    "robustness.json",
)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "results")
    parser.add_argument("--check", action="store_true", help="compare with the released results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    acquisition = DATA / "measurement"
    calibration_path = DATA / "calibration" / "frozen-calibration.json"
    calibration = read_json(calibration_path)

    report = matched_work_report(acquisition, calibration_path)
    write_json(args.output / "work-summary.json", report)
    write_csv(args.output / "work-by-cycle.csv", report["rows"])

    frequencies, _ = collect_frequencies(acquisition, calibration_path)
    budget = energy_budget(frequencies, calibration)
    check_energy_budget(budget, report)
    write_csv(args.output / "energy-budget.csv", budget)

    write_json(args.output / "branch-populations.json", branch_populations(acquisition, calibration_path))
    write_json(args.output / "robustness.json", robustness(frequencies, calibration))

    primary = report["primary_gain"]
    low, high = report["primary_gain_95_interval"]
    print(f"Daemonic gain at N = 16: {primary:.4f} hbar*omega, 95% interval [{low:.4f}, {high:.4f}]")
    print(f"Results written to {args.output}")

    if args.check:
        differences = [line for name in PRODUCTS for line in compare_files(args.output / name, RELEASED / name)]
        if differences:
            sys.exit("Differs from the released results:\n" + "\n".join(differences[:20]))
        print("All products agree with the released results.")


if __name__ == "__main__":
    main()
