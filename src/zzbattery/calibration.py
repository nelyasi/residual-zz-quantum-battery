"""Calibration of the sensing cycle and the frozen extraction rule.

The calibration acquisition contains two kinds of circuit, each prepared with the
battery in |0> or |1>:

* ``assignment``: a direct battery readout, which gives the binary readout response;
* ``instrument``: one sensing cycle without charging, which gives the joint
  distribution of the four-bit spectator outcome and the final battery readout.

Inverting the readout response gives the outcome matrices ``A_m`` (Eq. 18 of the
manuscript), composed with the charging map. The extraction rule and the model
predictions for every cycle number then follow from these matrices alone.
"""

import numpy as np

from .archive import load_run
from .design import CYCLES, FEEDBACK_RULE, MAX_RECENT_CYCLES, OUTCOMES_PER_CYCLE, SHOTS, Q


def outcome_matrices(assignment_counts, instrument_counts):
    """Estimate the readout response and the joint outcome/transition matrices.

    Parameters
    ----------
    assignment_counts : array (2, 2)
        Counts of the battery readout (row) for each prepared state (column).
    instrument_counts : array (16, 2, 2)
        Counts of spectator outcome ``m``, battery readout (row) and preparation (column).

    Returns
    -------
    response : array (2, 2)
        Column-stochastic readout response with a half count added to each cell.
    matrices : array (16, 2, 2)
        Outcome matrices ``A_m`` including the charging map with probability ``Q``.
    negative_mass : float
        Probability mass removed when projecting the inverted estimate onto
        nonnegative values.
    """
    assignment = np.asarray(assignment_counts, dtype=float) + 0.5
    assignment /= assignment.sum(axis=0)
    observations = np.asarray(instrument_counts, dtype=float) + 0.5
    observations /= observations.sum(axis=(0, 1), keepdims=True)
    corrected = np.einsum("ab,mbc->mac", np.linalg.inv(assignment), observations)
    clipped = np.maximum(corrected, 0)
    clipped /= clipped.sum(axis=(0, 1), keepdims=True)
    charge = np.array([[1 - Q, Q], [Q, 1 - Q]])
    return assignment, clipped @ charge, float(-np.minimum(corrected, 0).sum())


def count_calibration(rows):
    """Tabulate assignment and instrument counts from calibration shot records."""
    if {row["metadata"]["role"] for row in rows} != {"assignment", "instrument"}:
        raise ValueError("Expected the assignment and instrument calibration circuits.")
    assignment = np.zeros((2, 2), dtype=int)
    instrument = np.zeros((OUTCOMES_PER_CYCLE, 2, 2), dtype=int)
    for row in rows:
        meta = row["metadata"]
        final = np.array([int(bit) for bit in row["battery_out"]])
        if meta["role"] == "assignment":
            assignment[:, meta["prepared"]] += np.bincount(final, minlength=2)
        else:
            outcomes = np.array([int(bits, 2) for bits in row["record"]])
            np.add.at(instrument[:, :, meta["prepared"]], (outcomes, final), 1)
    expected = 16 * SHOTS
    if not ((assignment.sum(axis=0) == expected).all() and (instrument.sum(axis=(0, 1)) == expected).all()):
        raise ValueError("The calibration acquisition is incomplete.")
    return assignment, instrument


def freeze_controller(matrices, cycles=CYCLES):
    """Return the extraction decisions and the ideal-work predictions for every cycle number.

    The retained outcome index is little-endian in cycle and spectator; for two
    retained outcomes it is ``first + 16 * second``. The battery state before the
    retained window is propagated without conditioning through all earlier cycles.
    The controller applies X to the battery whenever the excited-state population
    conditioned on the retained outcomes exceeds one half.
    """
    matrices = np.asarray(matrices, dtype=float)
    transition = matrices.sum(axis=0)
    if (
        matrices.shape != (OUTCOMES_PER_CYCLE, 2, 2)
        or not np.isfinite(matrices).all()
        or (matrices < 0).any()
        or not np.allclose(transition.sum(axis=0), 1.0)
    ):
        raise ValueError("Expected 16 nonnegative outcome matrices of a Markov instrument.")

    table, forecasts = {}, {}
    for n in cycles:
        if n < 1:
            raise ValueError("Each comparison requires at least one sensing cycle.")
        retained = min(n, MAX_RECENT_CYCLES)
        prior = np.linalg.matrix_power(transition, n - retained) @ np.array([1.0, 0.0])
        joint = prior[None, :]
        for _ in range(retained):
            joint = np.einsum("mij,sj->msi", matrices, joint).reshape((-1, 2))

        flip = joint[:, 1] > joint[:, 0]
        pre = float(joint[:, 1].sum())
        conditional = float(np.sum(joint[flip, 1] - joint[flip, 0]))
        unconditional = max(0.0, 2 * pre - 1)
        table[str(n)] = flip.astype(int).tolist()
        forecasts[str(n)] = {
            "battery_before": pre,
            "unconditional_work": unconditional,
            "conditional_work": conditional,
            "daemonic_gain": conditional - unconditional,
            "fixed_flip": int(pre > 0.5),
            "conditional_flip_probability": float(joint[flip].sum()),
        }
    return table, forecasts


def build_calibration(folder):
    """Reconstruct the frozen calibration from the archived calibration acquisition."""
    manifest, rows = load_run(folder)
    assignment, instrument = count_calibration(rows)
    response, matrices, negative_mass = outcome_matrices(assignment, instrument)
    table, forecasts = freeze_controller(matrices)
    return {
        "assignment_counts": assignment.tolist(),
        "instrument_counts": instrument.tolist(),
        "readout_response": response.tolist(),
        "population_instrument": matrices.tolist(),
        "negative_mass_before_projection": negative_mass,
        "controller_table": table,
        "predictions": forecasts,
        "source_shots_sha256": manifest["shots_sha256"],
        "feedback_rule": FEEDBACK_RULE,
        "maximum_recent_cycles": MAX_RECENT_CYCLES,
    }
