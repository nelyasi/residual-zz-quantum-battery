"""Independent audit of the matched-feedback measurements.

This script deliberately does not import :mod:`zzbattery`. It re-reads the raw
records, checks the design (every circuit present once, charging phases shared by
prefix across cycle numbers and circuit types, the recorded feedback decisions
consistent with the frozen rule) and recomputes the primary estimates with its own
implementation of the bootstrap, including the branch-resolved counts at N = 16 and 20.

Usage: python scripts/audit_measurements.py [--data DATA_DIR] [--output FILE]
"""

import argparse
import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CYCLES = (1, 2, 4, 6, 8, 10, 12, 16, 20)
ARMS = ("before", "unconditional", "conditional")
SEQUENCES, SHOTS, DRAWS, SEED = 32, 384, 4000, 408619


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_records(measurement, calibration_path, calibration):
    """Return every shot record keyed by (sequence, cycle number, circuit type)."""
    rows = {}
    for path in sorted(measurement.glob("batch-*")):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise SystemExit(f"{path.name} is incomplete.")
        assert manifest["shots_per_circuit"] == SHOTS
        assert manifest["calibration_sha256"] == digest(calibration_path)
        assert manifest["shots_sha256"] == digest(path / "shots.json")
        assert manifest["qpy_sha256"] == digest(path / "circuits.qpy")
        records = json.loads((path / "shots.json").read_text())
        assert len(records) == len(manifest["circuits"])
        for record, circuit in zip(records, manifest["circuits"], strict=True):
            assert record["metadata"] == circuit
            n, sequence, arm = (circuit[key] for key in ("cycle_count", "sequence", "arm"))
            assert n == circuit["steps"]
            assert len(record["battery_out"]) == len(record["record"]) == SHOTS
            assert all(len(bits) == 4 * min(n, 2) and set(bits) <= {"0", "1"} for bits in record["record"])
            assert set(record["battery_out"]) <= {"0", "1"}
            if n > 2:
                assert len(record["history"]) == SHOTS
                assert all(len(bits) == 4 * (n - 2) and set(bits) <= {"0", "1"} for bits in record["history"])
            assert circuit["feedback_applied"] == (
                arm == "conditional" and any(calibration["controller_table"][str(n)])
            )
            assert circuit["fixed_flip"] == calibration["predictions"][str(n)]["fixed_flip"]
            assert circuit["feedback_window"] == min(2, n)
            key = (sequence, n, arm)
            assert key not in rows
            rows[key] = record
        print(path.name, manifest["job_id"], len(records) * SHOTS)
    assert set(rows) == set(product(range(SEQUENCES), CYCLES, ARMS))
    for sequence in range(SEQUENCES):
        reference = rows[sequence, 20, "before"]["metadata"]["charging_phases"]
        for n, arm in product(CYCLES, ARMS):
            assert rows[sequence, n, arm]["metadata"]["charging_phases"] == reference[:n]
    return rows


def audit(data):
    measurement = data / "measurement"
    calibration_path = data / "calibration" / "frozen-calibration.json"
    calibration = json.loads(calibration_path.read_text())
    rows = load_records(measurement, calibration_path, calibration)
    total = len(rows) * SHOTS
    print("Verified", len(rows), "circuits /", total, "shots; per cycle and arm", SEQUENCES * SHOTS)

    # Excited-readout frequency f1 and, per branch, (number of shots, number excited).
    f1 = np.empty((SEQUENCES, len(CYCLES), 3))
    branches = np.zeros((SEQUENCES, len(CYCLES), 3, 2, 2), int)
    for s, i, a in product(range(SEQUENCES), range(len(CYCLES)), range(3)):
        record = rows[s, CYCLES[i], ARMS[a]]
        y = np.array(list(map(int, record["battery_out"])))
        recent = np.array([int(bits, 2) for bits in record["record"]])
        x_selected = np.take(calibration["controller_table"][str(CYCLES[i])], recent)
        f1[s, i, a] = y.mean()
        for j in (0, 1):
            group = y[x_selected == j]
            branches[s, i, a, j] = len(group), group.sum()
    assert (branches.sum(axis=-2)[..., 0] == SHOTS).all()
    assert np.allclose(branches.sum(axis=-2)[..., 1] / SHOTS, f1)

    e01 = calibration["readout_response"][1][0]
    e10 = calibration["readout_response"][0][1]
    contrast = 1 - e01 - e10
    p = (f1.mean(axis=0) - e01) / contrast
    decreases = np.stack((p[:, 0] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 2]), axis=-1)
    per_sequence = (
        np.stack((f1[:, :, 0] - f1[:, :, 1], f1[:, :, 0] - f1[:, :, 2], f1[:, :, 1] - f1[:, :, 2]), axis=-1) / contrast
    )

    assignment = np.asarray(calibration["assignment_counts"], float)
    rng = np.random.default_rng(SEED)
    draw_errors = rng.beta(
        [assignment[1, 0] + 0.5, assignment[0, 1] + 0.5],
        [assignment[0, 0] + 0.5, assignment[1, 1] + 0.5],
        size=(DRAWS, 2),
    )
    resample = rng.integers(0, SEQUENCES, size=(DRAWS, SEQUENCES))
    draw_contrast = 1 - draw_errors.sum(axis=1)
    draw_f1 = f1[resample].mean(axis=1)
    draw_decreases = (
        np.stack(
            (
                draw_f1[..., 0] - draw_f1[..., 1],
                draw_f1[..., 0] - draw_f1[..., 2],
                draw_f1[..., 1] - draw_f1[..., 2],
            ),
            axis=-1,
        )
        / draw_contrast[:, None, None]
    )
    intervals = np.quantile(draw_decreases, (0.025, 0.975), axis=0)
    i16, i20 = CYCLES.index(16), CYCLES.index(20)
    plateau = np.quantile(draw_decreases[:, i20, 1] - draw_decreases[:, i16, 1], (0.05, 0.95))
    branch_totals = branches.sum(axis=0)
    draw_branches = branches[resample].sum(axis=1)

    result = {
        "total_completed_shots": int(total),
        "n16_gain": decreases[i16, 2].item(),
        "n16_gain_CI95": intervals[:, i16, 2].tolist(),
        "n20_minus_n16_conditional": (decreases[i20, 1] - decreases[i16, 1]).item(),
        "n20_minus_n16_conditional_CI90": plateau.tolist(),
        "equivalence_supported": bool(np.max(np.abs(plateau)) < 0.02),
        "rows": {},
    }
    for i, n in enumerate(CYCLES):
        row = {
            "readout_corrected_p1": dict(zip(ARMS, p[i].tolist(), strict=True)),
            "energy_decreases": decreases[i].tolist(),
            "energy_decreases_CI95": intervals[:, i, :].tolist(),
            "energy_decreases_SD": per_sequence[:, i].std(axis=0, ddof=1).tolist(),
        }
        print(n, p[i].round(7), decreases[i].round(7), intervals[:, i, 2].round(7))
        if n in (16, 20):
            counts = branch_totals[i]
            branch_p = (counts[..., 1] / counts[..., 0] - e01) / contrast
            draw_counts = draw_branches[:, i]
            draw_branch_p = (
                (draw_counts[..., 1] / draw_counts[..., 0]) - draw_errors[:, 0, None, None]
            ) / draw_contrast[:, None, None]
            row["branch_counts_by_arm_and_identity_then_x"] = counts.tolist()
            row["branch_p1_by_arm_and_identity_then_x"] = branch_p.tolist()
            row["branch_p1_CI95"] = np.quantile(draw_branch_p, (0.025, 0.975), axis=0).tolist()
            # Ideal conditional population predicted from each reference circuit, weighting
            # its branches by that circuit's own outcome frequencies.
            for reference, name in enumerate(("before", "unconditional")):
                weight = counts[reference, :, 0] / counts[reference, :, 0].sum()
                ideal = (weight * np.array([branch_p[reference, 0], 1 - branch_p[reference, 1]])).sum()
                draw_weights = draw_counts[:, reference, :, 0] / draw_counts[:, reference, :, 0].sum(
                    axis=-1, keepdims=True
                )
                draw_ideal = (
                    draw_weights
                    * np.stack((draw_branch_p[:, reference, 0], 1 - draw_branch_p[:, reference, 1]), axis=-1)
                ).sum(axis=-1)
                p_conditional = (draw_f1[:, i, 2] - draw_errors[:, 0]) / draw_contrast
                row[f"ideal_conditional_from_{name}"] = float(ideal)
                row[f"observed_minus_ideal_conditional_from_{name}"] = float(p[i, 2] - ideal)
                row[f"observed_minus_ideal_from_{name}_CI95"] = np.quantile(
                    p_conditional - draw_ideal, (0.025, 0.975)
                ).tolist()
            row["x_branch_inversion_residual_from_unconditional"] = float(branch_p[2, 1] - (1 - branch_p[1, 1]))
            row["i_branch_residual_from_unconditional"] = float(branch_p[2, 0] - branch_p[1, 0])
            row["x_branch_realized_gain_unconditional_minus_conditional"] = float(branch_p[1, 1] - branch_p[2, 1])
            row["x_branch_gain_CI95"] = np.quantile(
                draw_branch_p[:, 1, 1] - draw_branch_p[:, 2, 1], (0.025, 0.975)
            ).tolist()
        result["rows"][n] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "phoenix-2026-11-22-matched-feedback")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "results" / "hardware-audit.json")
    args = parser.parse_args()
    result = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Primary", result["n16_gain"], result["n16_gain_CI95"])
    print(
        "N=20 - 16 conditional",
        result["n20_minus_n16_conditional"],
        result["n20_minus_n16_conditional_CI90"],
        result["equivalence_supported"],
    )


if __name__ == "__main__":
    main()
