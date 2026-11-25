"""Acquire the matched-feedback experiment on IBM Quantum Phoenix.

Not needed to reproduce the published analysis. The run proceeds in three stages,
each staged to disk before submission and resumable after interruption:

1. qualification: real-time branching probes and the three 20-cycle circuits;
2. calibration: assignment and single-cycle instrument circuits, from which the
   extraction rule is frozen before any measurement circuit is compiled;
3. measurement: four batches of 216 circuits (72 matched triples each).

The qualification circuits need a provisional decision table, supplied as JSON
(``{"1": [...], "2": [...], ...}``). In the archived run it came from an earlier
pilot calibration; the qualification checks only the control flow and enters no
reported estimate.

Usage: python scripts/run_matched_campaign.py RUN_DIR --qualification-table TABLE.json
"""

import argparse
from pathlib import Path

from zzbattery.analysis import matched_work_report
from zzbattery.archive import load_run, read_json, sha256_file, write_json
from zzbattery.calibration import build_calibration
from zzbattery.circuits import calibration_circuits, experiment_batches, qualification_circuits
from zzbattery.design import SHOTS
from zzbattery.hardware import compile_for_backend, connect, run_stage, stage, submit, x_gate_samples

QUALIFICATION_SHOTS = 128
QUALIFICATION_THRESHOLD = 0.90


def check_qualification(folder):
    _, rows = load_run(folder)
    for row in rows:
        meta = row["metadata"]
        if meta.get("role") != "qualification":
            continue
        score = sum(int(bit) == meta["expected"] for bit in row["battery_out"]) / len(row["battery_out"])
        print(f"Qualification probe (expected {meta['expected']}): {score:.4f}", flush=True)
        if score < QUALIFICATION_THRESHOLD:
            raise RuntimeError("Real-time feedback failed the pre-specified qualification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="output folder for the acquisition")
    parser.add_argument("--qualification-table", type=Path, required=True)
    args = parser.parse_args()
    service, backend = connect()
    samples = x_gate_samples(backend)
    args.run.mkdir(parents=True, exist_ok=True)

    qualification = args.run / "qualification"
    table = read_json(args.qualification_table)
    run_stage(
        service,
        backend,
        qualification,
        qualification_circuits(table, samples),
        "qualification_suffix",
        QUALIFICATION_SHOTS,
    )
    check_qualification(qualification)

    calibration_folder = args.run / "calibration"
    run_stage(service, backend, calibration_folder, calibration_circuits(), "calibration", SHOTS)
    calibration = build_calibration(calibration_folder)
    calibration_path = calibration_folder / "frozen-calibration.json"
    write_json(calibration_path, calibration)
    calibration_hash = sha256_file(calibration_path)

    measurement = args.run / "measurement"
    measurement.mkdir(exist_ok=True)
    batches = []
    for index, circuits in enumerate(experiment_batches(calibration, samples)):
        folder = measurement / f"batch-{index:02d}"
        batches.append(folder)
        if not (folder / "manifest.json").exists():
            stage(folder, compile_for_backend(backend, circuits), "experiment", SHOTS, calibration_hash)
        elif read_json(folder / "manifest.json")["calibration_sha256"] != calibration_hash:
            raise ValueError(f"{folder.name} was staged with a different calibration.")

    # Confirm the first full batch at the provider before submitting the others.
    run_stage(service, backend, batches[0])
    for folder in batches[1:]:
        status = read_json(folder / "manifest.json")["status"]
        if status == "staged":
            submit(backend, folder)
        elif status not in ("submitted", "complete"):
            raise RuntimeError(f"Unconfirmed submission in {folder.name}; check the provider job list.")
    for folder in batches[1:]:
        run_stage(service, backend, folder)

    report = matched_work_report(measurement, calibration_path)
    write_json(args.run / "work-summary.json", report)
    print(f"Daemonic gain at N = 16: {report['primary_gain']:.4f}", flush=True)


if __name__ == "__main__":
    main()
