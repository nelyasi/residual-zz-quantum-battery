"""Work, energy and branch-population estimates for the matched comparison.

Battery populations are corrected for the binary readout response of the
independent calibration. Uncertainties use the 32 charging-phase sequences as the
resampling unit; the readout assignment rates are drawn from their count-plus-half
beta distributions (Appendix N of the manuscript).
"""

from itertools import product
from pathlib import Path

import numpy as np

from .archive import load_run, read_json, sha256_file
from .design import ARMS, CYCLES, FEEDBACK_RULE, MAX_RECENT_CYCLES, SEQUENCES, SHOTS

BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 408619
PRIMARY_CYCLES = 16
PLATEAU_CYCLES = (16, 20)
EQUIVALENCE_MARGIN = 0.02

ROBUSTNESS_DRAWS = 100_000
ROBUSTNESS_SEED = 813611

ENERGY_QUANTITIES = (
    "before_energy",
    "unconditional_energy",
    "conditional_energy",
    "unconditional_work",
    "conditional_work",
    "daemonic_gain",
)


# --------------------------------------------------------------------------- loading


def collect_frequencies(acquisition, calibration_path):
    """Excited-readout frequency of every circuit, shaped (sequence, cycle number, type).

    Returns the frequencies and the IBM job identifiers of the batches, after checking
    that every batch was staged with the frozen calibration and that the design is complete.
    """
    calibration_hash = sha256_file(calibration_path)
    cells, job_ids = {}, []
    for path in sorted(Path(acquisition).glob("batch-*")):
        manifest, rows = load_run(path)
        if manifest["kind"] != "experiment" or manifest["calibration_sha256"] != calibration_hash:
            raise ValueError("Every batch must use the archived frozen calibration.")
        if manifest["shots_per_circuit"] != SHOTS:
            raise ValueError("Unexpected number of shots per circuit.")
        for row in rows:
            meta = row["metadata"]
            key = (meta["sequence"], meta["cycle_count"], meta["arm"])
            if key in cells or len(row["battery_out"]) != SHOTS:
                raise ValueError("Duplicate circuit or incomplete shot record.")
            cells[key] = np.mean([int(bit) for bit in row["battery_out"]])
        job_ids.append(manifest["job_id"])
    expected = set(product(range(SEQUENCES), CYCLES, ARMS))
    if set(cells) != expected:
        raise ValueError(f"Expected {len(expected)} circuits, found {len(cells)}.")
    data = np.array([[[cells[s, n, a] for a in ARMS] for n in CYCLES] for s in range(SEQUENCES)])
    return data, job_ids


# ------------------------------------------------------------------ work estimates


def _energy_decreases(raw, readout_errors):
    """Readout-corrected decreases (pre - fixed, pre - conditional, fixed - conditional)."""
    e01, e10 = np.moveaxis(np.asarray(readout_errors), -1, 0)
    contrast = 1 - e01 - e10
    if np.any(contrast <= 0):
        raise ValueError("The readout response must distinguish ground and excited states.")
    before, unconditional, conditional = np.moveaxis(np.asarray(raw), -1, 0)
    return np.stack(
        [
            (before - unconditional) / contrast,
            (before - conditional) / contrast,
            (unconditional - conditional) / contrast,
        ],
        axis=-1,
    )


def work_summary(raw, calibration, draws=BOOTSTRAP_DRAWS):
    """Measured and ideal work for both extraction rules at every cycle number.

    The point estimates use the calibrated readout response. Intervals come from a
    common bootstrap over phase sequences with redrawn assignment rates. The plateau
    test compares the conditional work at N = 20 and N = 16 against a margin of 0.02.
    """
    raw = np.asarray(raw)
    if raw.shape != (SEQUENCES, len(CYCLES), len(ARMS)) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError("One finite battery-readout frequency is required per circuit.")
    counts = np.asarray(calibration["assignment_counts"], dtype=float)
    if counts.shape != (2, 2) or not counts.sum(axis=0).all():
        raise ValueError("Both assignment preparations are required.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    errors = rng.beta(
        [counts[1, 0] + 0.5, counts[0, 1] + 0.5],
        [counts[0, 0] + 0.5, counts[1, 1] + 0.5],
        size=(draws, 2),
    )
    indices = rng.integers(0, SEQUENCES, size=(draws, SEQUENCES))
    resampled = _energy_decreases(raw[indices].mean(axis=1), errors[:, None, :])
    point_errors = np.array([calibration["readout_response"][1][0], calibration["readout_response"][0][1]])
    per_sequence = _energy_decreases(raw, point_errors)
    work = per_sequence.mean(axis=0)
    sd = per_sequence.std(axis=0, ddof=1)
    interval = np.quantile(resampled, (0.025, 0.975), axis=0)
    lo, hi = (CYCLES.index(n) for n in PLATEAU_CYCLES)
    plateau_interval = np.quantile(resampled[:, hi, 1] - resampled[:, lo, 1], (0.05, 0.95))

    theory = calibration["predictions"]
    rows = []
    for i, n in enumerate(CYCLES):
        row = {"cycles": n, "shots_per_circuit_type": SEQUENCES * SHOTS}
        for j, quantity in enumerate(("unconditional_work", "conditional_work", "daemonic_gain")):
            row.update(
                {
                    f"{quantity}_model": theory[str(n)][quantity],
                    f"{quantity}_measured": float(work[i, j]),
                    f"{quantity}_SD": float(sd[i, j]),
                    f"{quantity}_95_lo": float(interval[0, i, j]),
                    f"{quantity}_95_hi": float(interval[1, i, j]),
                }
            )
        rows.append(row)
    primary = CYCLES.index(PRIMARY_CYCLES)
    return {
        "rows": rows,
        "bootstrap_draws": draws,
        "primary_cycles": PRIMARY_CYCLES,
        "primary_gain": rows[primary]["daemonic_gain_measured"],
        "primary_gain_95_interval": interval[:, primary, 2].tolist(),
        "stationarity_cycles": list(PLATEAU_CYCLES),
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "difference_90_interval": plateau_interval.tolist(),
        "equivalence_supported": bool(np.max(abs(plateau_interval)) < EQUIVALENCE_MARGIN),
    }


def matched_work_report(acquisition, calibration_path):
    """Work summary with provenance (job identifiers and calibration hash)."""
    calibration = read_json(calibration_path)
    frequencies, job_ids = collect_frequencies(acquisition, calibration_path)
    report = work_summary(frequencies, calibration)
    report.update(
        executed_job_ids=job_ids,
        source_calibration_sha256=sha256_file(calibration_path),
        feedback_rule=FEEDBACK_RULE,
        maximum_recent_cycles=MAX_RECENT_CYCLES,
    )
    return report


# ------------------------------------------------------------------- energy budget


def energy_budget(frequencies, calibration):
    """Stored and final energies in units of the gap, with sequence-paired differences."""
    response = np.asarray(calibration["readout_response"])
    contrast = response[1, 1] - response[1, 0]
    if contrast <= 0:
        raise ValueError("The readout response must distinguish the two preparations.")
    populations = (np.asarray(frequencies) - response[1, 0]) / contrast
    pre, fixed, feedback = (populations[..., ARMS.index(arm)] for arm in ARMS)
    per_sequence = np.stack((pre, fixed, feedback, pre - fixed, pre - feedback, fixed - feedback), axis=-1)
    means = per_sequence.mean(axis=0)
    deviations = per_sequence.std(axis=0, ddof=1)

    rows = []
    for index, cycles in enumerate(CYCLES):
        prediction = calibration["predictions"][str(cycles)]
        energy_before = prediction["battery_before"]
        energy_fixed = 1 - energy_before if prediction["fixed_flip"] else energy_before
        energy_feedback = energy_before - prediction["conditional_work"]
        model = (
            energy_before,
            energy_fixed,
            energy_feedback,
            energy_before - energy_fixed,
            energy_before - energy_feedback,
            energy_fixed - energy_feedback,
        )
        row = {"cycles": cycles, "shots_per_circuit_type": SEQUENCES * SHOTS}
        for j, name in enumerate(ENERGY_QUANTITIES):
            row[f"{name}_measured"] = float(means[index, j])
            row[f"{name}_SD"] = float(deviations[index, j])
            row[f"{name}_model"] = float(model[j])
        rows.append(row)
    return rows


def check_energy_budget(rows, report):
    """Confirm that the energy budget reproduces the work report to 1e-12."""
    for actual, expected in zip(rows, report["rows"], strict=True):
        for key in ("cycles", "shots_per_circuit_type"):
            if actual[key] != expected[key]:
                raise ValueError(f"Inconsistent {key} at N = {actual['cycles']}.")
        for name in ENERGY_QUANTITIES[3:]:
            for suffix in ("measured", "SD", "model"):
                key = f"{name}_{suffix}"
                if not np.isclose(actual[key], expected[key], atol=1e-12, rtol=0):
                    raise ValueError(f"{key} disagrees with the work report at N = {actual['cycles']}.")


# -------------------------------------------------------------- branch populations


def branch_populations(acquisition, calibration_path):
    """Final battery populations grouped by the operation the frozen rule selects (Fig. 7)."""
    calibration = read_json(calibration_path)
    if calibration.get("feedback_rule") != FEEDBACK_RULE:
        raise ValueError("Expected the frozen recent-outcome calibration.")
    response = np.asarray(calibration["readout_response"])
    e01 = response[1, 0]
    contrast = 1 - e01 - response[0, 1]
    if contrast <= 0:
        raise ValueError("The readout response must distinguish the two preparations.")
    expected = set(product(CYCLES, range(SEQUENCES), ARMS))
    statistics, jobs = {}, []
    calibration_hash = sha256_file(calibration_path)
    for path in sorted(Path(acquisition).glob("batch-*")):
        manifest, shots = load_run(path)
        if (
            manifest["kind"] != "experiment"
            or manifest["calibration_sha256"] != calibration_hash
            or manifest["shots_per_circuit"] != SHOTS
        ):
            raise ValueError("A batch does not match the frozen calibration or design.")
        jobs.append(manifest["job_id"])
        for entry in shots:
            meta = entry["metadata"]
            key = (meta["cycle_count"], meta["sequence"], meta["arm"])
            if key in statistics or key not in expected:
                raise ValueError("Duplicate circuit or circuit outside the acquisition plan.")
            outcomes, battery = entry["record"], entry["battery_out"]
            n = key[0]
            decisions = np.array(calibration["controller_table"][str(n)], dtype=bool)
            width = 4 * min(calibration["maximum_recent_cycles"], n)
            if (
                len(outcomes) != SHOTS
                or len(battery) != SHOTS
                or any(len(bits) != width or set(bits) - {"0", "1"} for bits in outcomes)
                or set(battery) - {"0", "1"}
            ):
                raise ValueError("Incomplete or invalid measurement outcomes.")
            selected_x = decisions[[int(bits, 2) for bits in outcomes]]
            observed = np.array(battery, dtype=int)
            statistics[key] = [observed[selected_x], observed[~selected_x]]
    if set(statistics) != expected:
        raise ValueError("Every circuit of the design is required.")

    rows = []
    for n in CYCLES:
        row = {"cycles": n}
        for i, operation in enumerate(("X", "I")):
            row[operation] = {}
            for arm in ("unconditional", "conditional"):
                groups = [statistics[n, sequence, arm][i] for sequence in range(SEQUENCES)]
                if any(group.size == 0 for group in groups):
                    raise ValueError("Every phase sequence must contribute to each branch.")
                shot_count = sum(group.size for group in groups)
                excited = sum(int(group.sum()) for group in groups)
                probabilities = (np.array([group.mean() for group in groups]) - e01) / contrast
                row[operation][arm] = {
                    "mean": float((excited / shot_count - e01) / contrast),
                    "SD": float(probabilities.std(ddof=1)),
                    "shot_count": shot_count,
                    "excited_readout_count": excited,
                    "phase_sequences": len(groups),
                    "minimum_shots_per_sequence": min(group.size for group in groups),
                }
        rows.append(row)
    return {
        "rows": rows,
        "executed_job_ids": jobs,
        "source_calibration_sha256": calibration_hash,
        "mean_definition": "Corrected excited-state probability using all pooled shots in each branch and arm.",
        "SD_definition": (
            "Sample standard deviation across the corrected branch proportions of the 32 phase "
            "sequences, each given equal weight. It includes finite-shot variation."
        ),
    }


# --------------------------------------------------------------------- robustness


def robustness(frequencies, calibration, draws=ROBUSTNESS_DRAWS):
    """Plateau interval with an independent seed, and the N = 16 gain before readout correction."""
    earlier, later = CYCLES.index(16), CYCLES.index(20)
    difference = (
        frequencies[:, later, 0] - frequencies[:, later, 2] - frequencies[:, earlier, 0] + frequencies[:, earlier, 2]
    )
    counts = np.asarray(calibration["assignment_counts"])
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    errors = rng.beta(
        [counts[1, 0] + 0.5, counts[0, 1] + 0.5],
        [counts[0, 0] + 0.5, counts[1, 1] + 0.5],
        size=(draws, 2),
    )
    selected = rng.integers(0, len(difference), size=(draws, len(difference)))
    bounds = np.quantile(difference[selected].mean(axis=1) / (1 - errors.sum(axis=1)), [0.05, 0.95])
    raw_gain = np.mean(frequencies[:, earlier, 1] - frequencies[:, earlier, 2])
    return {f"late_cycle_{draws}_draw_interval": bounds.tolist(), "raw_n16_gain": float(raw_gain)}
