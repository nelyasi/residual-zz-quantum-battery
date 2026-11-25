"""Ramsey calibration of the residual ZZ shifts (Appendix M of the manuscript).

Each spectator is prepared in |+>, idles next to the battery prepared in |0> or
|1>, and is read in X and Y. The complex signal z_{j,b} = <X_j>_b + i <Y_j>_b
rotates at a rate set by the battery state, and the slope of the unwrapped
differential phase arg(z_{j,1} z*_{j,0}) gives -2 pi zeta_j.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _bits(values, ndim, name):
    data = np.asarray(values)
    if data.ndim != ndim or not data.size or not np.all((data == 0) | (data == 1)):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array of bits")
    return data.astype(np.uint8, copy=False)


def _ramsey_trace(points, condition_on_battery):
    traces = defaultdict(list)
    diagnostics = []
    sensors = None
    for point in points:
        tau = float(point["delay_us"])
        prepared, axis = point["prepared"], str(point["axis"]).upper()
        if not np.isfinite(tau) or tau < 0 or prepared not in (0, 1) or axis not in ("X", "Y"):
            raise ValueError("Ramsey points need nonnegative delay_us, prepared 0/1 and axis X/Y")
        bits = _bits(point["ramsey"], 2, "ramsey")
        terminal = _bits(point["battery_out"], 1, "battery_out") if "battery_out" in point else None
        if condition_on_battery and terminal is None:
            raise ValueError("Conditioning on the battery requires terminal Z shots.")
        if terminal is not None and len(bits) != len(terminal):
            raise ValueError("Ramsey and battery_out must contain the same shots")
        if sensors is None:
            sensors = bits.shape[1]
        if bits.shape[1] != sensors:
            raise ValueError("Ramsey points must have a common spectator layout")
        matches = terminal == prepared if terminal is not None else None
        mask = matches if condition_on_battery else np.ones(len(bits), dtype=bool)
        traces[int(prepared), axis, tau].append(bits[mask])
        diagnostics.append(
            {
                "delay_us": tau,
                "prepared": int(prepared),
                "axis": axis,
                "shots": len(bits),
                "terminal_excited_fraction": float(terminal.mean()) if terminal is not None else None,
                "terminal_matches_preparation_fraction": float(matches.mean()) if matches is not None else None,
                "shots_used": int(mask.sum()),
            }
        )
    if sensors is None:
        raise ValueError("at least one Ramsey point is required")
    return {key: np.concatenate(shots) for key, shots in traces.items()}, diagnostics, sensors


def _fit_complex_ramsey(delays, quadratures, counts, max_abs_khz):
    """Fit A exp(-t/T2 - 2 pi i f t), with an independent complex offset phase in A."""
    reference = delays[0]
    time = delays - reference
    signal = quadratures[0] + 1j * quadratures[1]
    # Jeffreys smoothing keeps the variance finite if a setting gives all 0 or all 1.
    positive = (counts * (1 + quadratures) / 2 + 0.5) / (counts + 1)
    noise = np.sqrt(4 * positive * (1 - positive) / counts)
    weights = 1 / np.mean(noise**2, axis=0)
    span = time[-1]
    grid_size = min(4097, max(65, 1 + int(16 * max_abs_khz * span / 1000)))
    grid = np.linspace(-max_abs_khz, max_abs_khz, grid_size)
    harmonics = np.exp(-2j * np.pi * grid[:, None] * time[None, :] / 1000)
    projections = (harmonics.conj() @ (weights * signal)) / weights.sum()
    candidates = np.argsort(np.abs(projections))[-5:]
    decay0 = float(np.clip(-np.polyfit(time, np.log(np.maximum(np.abs(signal), 0.03)), 1)[0], 0, 1))

    def residual(parameters):
        a_re, a_im, freq, decay = parameters
        predicted = (a_re + 1j * a_im) * np.exp(-decay * time - 2j * np.pi * freq * time / 1000)
        difference = predicted - signal
        return np.concatenate((difference.real / noise[0], difference.imag / noise[1]))

    starts = ([projections[k].real, projections[k].imag, grid[k], decay0] for k in candidates)
    fit = min(
        (
            least_squares(
                residual,
                start,
                bounds=([-np.inf, -np.inf, -max_abs_khz, 0], [np.inf, np.inf, max_abs_khz, np.inf]),
                x_scale="jac",
            )
            for start in starts
        ),
        key=lambda result: result.cost,
    )
    a_re, a_im, freq, decay = fit.x
    degrees = 2 * len(time) - len(fit.x)
    jac = fit.jac
    frequency_se = None
    if degrees > 0 and np.linalg.matrix_rank(jac) == len(fit.x):
        covariance = np.linalg.pinv(jac.T @ jac) * max(1.0, 2 * fit.cost / degrees)
        frequency_se = float(np.sqrt(max(0.0, covariance[2, 2])))
    return {
        "frequency_khz": float(freq),
        "frequency_se_khz": frequency_se,
        "t2_us": float(1 / decay) if decay > 1e-10 else None,
        "phase_at_reference_rad": float(np.angle(a_re + 1j * a_im)),
        "reference_delay_us": float(reference),
        "amplitude_at_reference": float(np.hypot(a_re, a_im)),
        "reduced_chi_squared": float(2 * fit.cost / degrees),
        "at_frequency_bound": bool(abs(freq) >= max_abs_khz * (1 - 1e-5)),
        "optimizer_converged": bool(fit.success),
        "delays_us": delays.tolist(),
    }


def fit_ramsey(points, *, max_abs_khz=100.0, condition_on_battery=True):
    """Fit each spectator's signed conditional shift with measured X and Y shots.

    Points contain circuit metadata (`delay_us`, `prepared`, `axis`) and shot
    arrays (`ramsey`: shots x spectators, `battery_out`: shots). By default the
    fit keeps shots whose measured final battery matches preparation. This is
    an assignment-sensitive survival diagnostic. Frequencies are constrained
    by max_abs_khz, which must exclude aliases for the chosen delay grid.
    """
    if not np.isfinite(max_abs_khz) or max_abs_khz <= 0:
        raise ValueError("max_abs_khz must be positive and finite")
    traces, survival, spectators = _ramsey_trace(points, condition_on_battery)
    fits = {}
    for prepared in (0, 1):
        delays = np.array(
            sorted(
                {
                    delay
                    for bit, axis, delay in traces
                    if bit == prepared
                    and axis == "X"
                    and (prepared, "Y", delay) in traces
                    and min(len(traces[prepared, ax, delay]) for ax in ("X", "Y")) >= 2
                }
            )
        )
        if len(delays) < 3:
            raise ValueError(f"prepared={prepared} requires at least three paired X/Y delays with two usable shots")
        counts = np.array([[len(traces[prepared, axis, delay]) for delay in delays] for axis in ("X", "Y")])
        for spectator in range(spectators):
            means = np.array(
                [
                    [1 - 2 * traces[prepared, axis, delay][:, spectator].mean() for delay in delays]
                    for axis in ("X", "Y")
                ]
            )
            fits[prepared, spectator] = _fit_complex_ramsey(delays, means, counts, max_abs_khz)
    results = []
    for spectator in range(spectators):
        ground, excited = fits[0, spectator], fits[1, spectator]
        standard_errors = [fit["frequency_se_khz"] for fit in (ground, excited)]
        results.append(
            {
                "spectator": spectator,
                "zz_khz": excited["frequency_khz"] - ground["frequency_khz"],
                "zz_se_khz": float(np.hypot(*standard_errors)) if None not in standard_errors else None,
                "ground": ground,
                "excited": excited,
            }
        )
    return {
        "conditioned_on_measured_terminal_battery": bool(condition_on_battery),
        "spectators": results,
        "battery_survival": survival,
    }


def fit_differential_ramsey(points, *, condition_on_battery=False):
    """Estimate signed ZZ from the phase difference at each shared delay.

    A common Ramsey phase drift cancels before the line fit. The sampled
    delays must resolve changes of the differential phase smaller than pi.
    """
    traces, diagnostics, spectators = _ramsey_trace(points, condition_on_battery)
    delays = sorted(
        {
            key[2]
            for key in traces
            if all(
                (prepared, axis, key[2]) in traces and len(traces[prepared, axis, key[2]]) >= 2
                for prepared in (0, 1)
                for axis in ("X", "Y")
            )
        }
    )
    if len(delays) < 3:
        raise ValueError("Differential fitting needs at least three common X/Y delays.")
    vectors, variances = [], []
    for prepared in (0, 1):
        x, y = (np.stack([1 - 2 * traces[prepared, axis, t].mean(axis=0) for t in delays]) for axis in ("X", "Y"))
        nx, ny = (np.array([len(traces[prepared, axis, t]) for t in delays])[:, None] for axis in ("X", "Y"))
        squared = x * x + y * y
        if np.min(squared) < 0.01:
            raise ValueError("A Ramsey amplitude is below 0.1; differential phase is unresolved.")
        vectors.append(x + 1j * y)
        vx = np.maximum(1 - x * x, 1 / nx) / nx
        vy = np.maximum(1 - y * y, 1 / ny) / ny
        variances.append((y * y * vx + x * x * vy) / squared**2)
    phase = np.unwrap(np.angle(vectors[1] * np.conj(vectors[0])), axis=0)
    variance = variances[0] + variances[1]
    design = np.column_stack((np.ones(len(delays)), delays))
    results = []
    for j in range(spectators):
        weight = 1 / np.sqrt(variance[:, j])
        weighted = weight[:, None] * design
        offsets, slope = np.linalg.lstsq(weighted, weight * phase[:, j], rcond=None)[0]
        residual = weight * (phase[:, j] - offsets - slope * np.asarray(delays))
        reduced_chi2 = float(residual @ residual / (len(delays) - 2))
        covariance = np.linalg.inv(weighted.T @ weighted) * max(1.0, reduced_chi2)
        results.append(
            {
                "spectator": j,
                "zz_khz": float(-1000 * slope / (2 * np.pi)),
                "zz_se_khz": float(1000 * np.sqrt(covariance[1, 1]) / (2 * np.pi)),
                "reduced_chi_squared": reduced_chi2,
                "phase_rad": phase[:, j].tolist(),
                "phase_se_rad": np.sqrt(variance[:, j]).tolist(),
                "fit_phase_rad": (offsets + slope * np.asarray(delays)).tolist(),
            }
        )
    return {"spectators": results, "delays_us": delays, "battery_survival": diagnostics}


def load_ramsey_counts(path):
    """Expand a histogram of spectator bits (CSV: delay_us, prepared, axis, spectator_bits, shots)."""
    groups = defaultdict(list)
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            key = (float(row["delay_us"]), int(row["prepared"]), row["axis"])
            bits = np.array([int(bit) for bit in row["spectator_bits"]], dtype=np.uint8)
            groups[key].append(np.tile(bits, (int(row["shots"]), 1)))
    return [
        {"delay_us": delay, "prepared": prepared, "axis": axis, "ramsey": np.concatenate(parts)}
        for (delay, prepared, axis), parts in groups.items()
    ]
