"""Qiskit circuits for the battery, its spectators and the real-time extraction step.

Logical qubit 0 is the battery; logical qubits 1-4 are the spectators in the order
of :data:`zzbattery.design.LAYOUT`. No two-qubit gate is applied: the sensing
interaction is the static residual ZZ coupling during a common idle.
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import product
from math import isfinite, pi, tau
from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr

from .design import (
    ARMS,
    CALIBRATION_SEED,
    CYCLES,
    FEEDBACK_RULE,
    INTERLEAVE_SEED,
    MAX_RECENT_CYCLES,
    PHASE_SEED,
    READOUT_AXES_RAD,
    SEQUENCES,
    SPECTATORS,
    TAU_US,
    THETA,
    TRIPLES_PER_BATCH,
    ZZ_SHIFTS_KHZ,
)

Mode = Literal["residual", "digital"]


# ----------------------------------------------------------------------------- sensing


def contrast_axes(tau_us: float, couplings_khz: Sequence[float]) -> tuple[float, ...]:
    """Return the contrast-maximizing equatorial readout axes, alpha_j = pi/2 - phi_j/2."""
    return tuple(pi / 2 - pi * rate * tau_us / 1000 for rate in couplings_khz)


def _require_duration(duration: float, label: str) -> None:
    if not isfinite(duration) or duration < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")


def _idle(circuit: QuantumCircuit, duration_us: float) -> None:
    """Common idle on all qubits, bracketed by barriers to fix its placement."""
    circuit.barrier()
    if duration_us:
        circuit.delay(duration_us, circuit.qubits, unit="us")
    circuit.barrier()


def _readout_rotation(circuit: QuantumCircuit, qubit, alpha: float) -> None:
    """Map the equatorial state |+_alpha> onto |0> before a Z measurement."""
    circuit.rz(-alpha, qubit)
    circuit.h(qubit)


def build_record_circuit(
    *,
    theta: float,
    phases: Sequence[float],
    tau_us: float,
    couplings_khz: Sequence[float],
    axes_rad: Sequence[float] | None = None,
    mode: Mode = "residual",
    sensing: bool = True,
    readout_wait_us: float = 0.0,
    initial_battery: int = 0,
) -> QuantumCircuit:
    """Build repeated charging and sensing cycles followed by a battery readout.

    Each cycle applies the charging rotation ``R(theta, phase)`` to the battery,
    prepares the spectators in |+>, idles all qubits for ``tau_us``, rotates each
    spectator to its readout axis, measures and resets it. Record bit
    ``4 * cycle + j`` is 0 for the outcome ``+`` of spectator ``j``.

    In ``residual`` mode the conditional phase comes from the physical ZZ coupling.
    The ``digital`` mode replaces the idle by controlled phases implementing the
    ideal ZZ Hamiltonian, for local verification on a simulator.
    """
    _require_duration(tau_us, "The sensing time")
    _require_duration(readout_wait_us, "The readout wait")
    phases = tuple(float(phase) for phase in phases)
    couplings = tuple(float(coupling) for coupling in couplings_khz)
    if not isfinite(theta) or not phases or not all(map(isfinite, phases)):
        raise ValueError("A finite charging angle and at least one finite phase are required.")
    if not 1 <= len(couplings) <= SPECTATORS or not all(map(isfinite, couplings)):
        raise ValueError("Provide one to four finite signed couplings in kHz.")
    if mode not in ("residual", "digital"):
        raise ValueError("The mode must be 'residual' or 'digital'.")
    if initial_battery not in (0, 1):
        raise ValueError("The initial battery state must be 0 or 1.")

    n_spectators = len(couplings)
    axes = contrast_axes(tau_us, couplings) if axes_rad is None else tuple(axes_rad)
    if len(axes) != n_spectators or not all(map(isfinite, axes)):
        raise ValueError("Provide one finite readout axis per spectator.")

    device = QuantumRegister(n_spectators + 1, "device")
    battery_out = ClassicalRegister(1, "battery_out")
    record = ClassicalRegister(n_spectators * len(phases), "record") if sensing else None
    registers = (device, record, battery_out) if sensing else (device, battery_out)
    circuit = QuantumCircuit(*registers, name=f"battery_{mode}_{'sense' if sensing else 'idle'}")
    battery, spectators = device[0], device[1:]
    if initial_battery:
        circuit.x(battery)
        circuit.barrier()

    for step, phase in enumerate(phases):
        if theta:
            circuit.r(theta, phase, battery)
        if sensing:
            circuit.barrier()
            circuit.h(spectators)
            if mode == "residual":
                _idle(circuit, tau_us)
            else:
                circuit.barrier()
                for qubit, coupling in zip(spectators, couplings, strict=True):
                    circuit.cp(-tau * coupling * tau_us / 1000, battery, qubit)
                circuit.barrier()
            for qubit, alpha in zip(spectators, axes, strict=True):
                _readout_rotation(circuit, qubit, alpha)
            circuit.barrier()
            start = step * n_spectators
            circuit.measure(spectators, record[start : start + n_spectators])
            circuit.reset(spectators)
            circuit.barrier()
        elif mode == "residual":
            _idle(circuit, tau_us)
        if readout_wait_us:
            _idle(circuit, readout_wait_us)

    circuit.measure(battery, battery_out[0])
    circuit.metadata = {
        "mode": mode,
        "sensing": sensing,
        "steps": len(phases),
        "n_spectators": n_spectators,
        "theta": float(theta),
        "tau_us": float(tau_us),
        "couplings_khz": list(couplings),
        "axes_rad": list(axes),
        "readout_wait_us": float(readout_wait_us),
        "initial_battery": int(initial_battery),
        "feedback_applied": False,
    }
    return circuit


def build_ramsey_circuits(delays_us: Iterable[float], *, n_spectators: int = SPECTATORS) -> list[QuantumCircuit]:
    """Ramsey circuits conditioned on the battery basis state (Appendix M).

    Every spectator is read in X or Y and the battery in Z. The nominal idle
    excludes pulse durations, so phase fits require an intercept.
    """
    delays = tuple(float(delay) for delay in delays_us)
    if not delays:
        raise ValueError("At least one Ramsey delay is required.")
    for delay in delays:
        _require_duration(delay, "A Ramsey delay")
    circuits = []
    for index, delay in enumerate(delays):
        for prepared in (0, 1):
            for axis, alpha in (("X", 0.0), ("Y", pi / 2)):
                device = QuantumRegister(n_spectators + 1, "device")
                ramsey = ClassicalRegister(n_spectators, "ramsey")
                battery_out = ClassicalRegister(1, "battery_out")
                circuit = QuantumCircuit(device, ramsey, battery_out, name=f"ramsey_{index}_{prepared}_{axis}")
                if prepared:
                    circuit.x(device[0])
                circuit.barrier()
                circuit.h(device[1:])
                _idle(circuit, delay)
                for qubit in device[1:]:
                    _readout_rotation(circuit, qubit, alpha)
                circuit.barrier()
                circuit.measure(device[1:], ramsey)
                circuit.measure(device[0], battery_out[0])
                circuit.metadata = {"mode": "residual", "delay_us": delay, "prepared": prepared, "axis": axis}
                circuits.append(circuit)
    return circuits


def sensing_circuit(phases, initial_battery=0, theta=THETA):
    """Sensing circuit of the matched experiment with the archived couplings and axes."""
    return build_record_circuit(
        theta=theta,
        phases=phases,
        tau_us=TAU_US,
        couplings_khz=ZZ_SHIFTS_KHZ,
        axes_rad=READOUT_AXES_RAD,
        initial_battery=initial_battery,
    )


# ----------------------------------------------------------------- real-time predicate


@lru_cache(maxsize=2)
def _masked_regions(width):
    """Enumerate the inputs matched by every masked equality ``record & mask == value``."""
    values = range(1 << width)
    regions = []
    for mask in values:
        by_value = {}
        for value in values:
            masked = value & mask
            by_value[masked] = by_value.get(masked, 0) | (1 << value)
        regions.extend((mask, key, covered) for key, covered in by_value.items())
    return regions


@lru_cache(maxsize=32)
def _predicate_terms(truth, width):
    """Find a compact exact cover of the decision class with fewer terms."""
    positive = sum(1 << index for index, value in enumerate(truth) if value)
    negative = ((1 << len(truth)) - 1) ^ positive

    def cover(target):
        candidates = [region for region in _masked_regions(width) if region[2] & ~target == 0]
        remaining, selected = target, []
        while remaining:
            region = max(
                candidates,
                key=lambda item: ((item[2] & remaining).bit_count(), -item[0].bit_count()),
            )
            selected.append(region)
            remaining &= ~region[2]
        for region in reversed(selected.copy()):
            covered = 0
            for other in selected:
                if other != region:
                    covered |= other[2]
            if covered == target:
                selected.remove(region)
        return tuple((mask, value) for mask, value, _ in selected)

    direct, inverted = cover(positive), cover(negative)
    return (True, inverted) if len(inverted) < len(direct) else (False, direct)


def compact_predicate(record, decisions):
    """Classical expression that is true exactly for the record values selecting X.

    The predicate is a balanced OR of masked register comparisons, so the
    controller evaluates one Boolean condition per circuit.
    """
    decisions = tuple(bool(value) for value in decisions)
    if len(decisions) != 2 ** len(record):
        raise ValueError("The frozen rule must specify every retained outcome.")
    negate, terms = _predicate_terms(decisions, len(record))
    if not terms:
        first = expr.lift(record[0])
        combine = expr.logic_or if negate else expr.logic_and
        return combine(first, expr.logic_not(first))
    expressions = [expr.equal(expr.bit_and(record, mask), value) for mask, value in terms]
    while len(expressions) > 1:
        paired = [expr.logic_or(a, b) for a, b in zip(expressions[::2], expressions[1::2], strict=False)]
        expressions = paired + (expressions[-1:] if len(expressions) % 2 else [])
    return expr.logic_not(expressions[0]) if negate else expressions[0]


def feedback_circuit(phases, table, arm, fixed_flip, gate_samples):
    """One circuit of a matched triple.

    ``before`` measures the battery after the last sensing cycle. ``unconditional``
    applies the fixed operation and ``conditional`` applies X when the predicate on
    the most recent spectator outcomes is true. Both extraction circuits evaluate the
    same predicate and occupy one X-gate duration on either branch.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown circuit type {arm!r}.")
    if gate_samples <= 0:
        raise ValueError("The X-gate duration must be positive.")
    phases = tuple(phases)
    retained = min(len(phases), MAX_RECENT_CYCLES)
    decisions = table[str(len(phases))] if isinstance(table, Mapping) else table
    sensed = sensing_circuit(phases)
    device = sensed.qregs[0]
    full_record, out = sensed.cregs
    history_width = len(full_record) - SPECTATORS * retained
    history = (ClassicalRegister(history_width, "history"),) if history_width else ()
    record = ClassicalRegister(SPECTATORS * retained, "record")
    circuit = QuantumCircuit(device, *history, record, out)
    history_bits = tuple(history[0]) if history else ()
    bit_map = dict(zip(full_record, history_bits + tuple(record), strict=True))
    for instruction in sensed.data[:-1]:
        circuit.append(instruction.operation, instruction.qubits, tuple(bit_map[bit] for bit in instruction.clbits))

    battery = device[0]
    if arm != "before":
        with circuit.if_test(compact_predicate(record, decisions)) as otherwise:
            if arm == "conditional" or fixed_flip:
                circuit.x(battery)
            else:
                circuit.delay(gate_samples, battery, unit="dt")
        with otherwise:
            if arm == "unconditional" and fixed_flip:
                circuit.x(battery)
            else:
                circuit.delay(gate_samples, battery, unit="dt")
    circuit.measure(battery, out[0])
    circuit.metadata = {
        **sensed.metadata,
        "arm": arm,
        "fixed_flip": int(fixed_flip),
        "feedback_applied": arm == "conditional" and any(decisions),
        "feedback_window": retained,
        "feedback_rule": FEEDBACK_RULE,
    }
    return circuit


# ------------------------------------------------------------------ acquisition plans


def calibration_circuits():
    """Assignment and single-cycle instrument circuits, 16 repetitions each, shuffled."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    circuits = []
    for role, bit, repetition in product(("assignment", "instrument"), (0, 1), range(16)):
        if role == "assignment":
            device = QuantumRegister(5, "device")
            out = ClassicalRegister(1, "battery_out")
            circuit = QuantumCircuit(device, out)
            if bit:
                circuit.x(device[0])
            circuit.measure(device[0], out[0])
        else:
            circuit = sensing_circuit((0.0,), initial_battery=bit, theta=0)
        circuit.name = f"{role}_{bit}_{repetition}"
        circuit.metadata.update(role=role, prepared=bit, repetition=repetition)
        circuits.append(circuit)
    return [circuits[i] for i in rng.permutation(len(circuits))]


def charging_phases():
    """The 32 independent charging-phase sequences; a run of N cycles uses the first N."""
    return np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, size=(SEQUENCES, max(CYCLES)))


def qualification_circuits(table, gate_samples):
    """Hardware qualification run before the calibration.

    Two deterministic probes check that a measured bit routes the battery through the
    X branch (expected 1) or the delay branch (expected 0). Three 20-cycle circuits
    check that the sampler accepts the longest circuits of every type.
    """
    probes = []
    for expected in (0, 1):
        device = QuantumRegister(5, "device")
        result = ClassicalRegister(8, "record")
        out = ClassicalRegister(1, "battery_out")
        circuit = QuantumCircuit(device, result, out, name=f"recent_feedback_test_{expected}")
        if expected:
            circuit.x(device[1])
        circuit.measure(device[1], result[0])
        circuit.reset(device[1])
        circuit.measure(device[1], result[4])
        with circuit.if_test(expr.bit_xor(result[0], result[4])) as otherwise:
            circuit.x(device[0])
        with otherwise:
            circuit.delay(gate_samples, device[0], unit="dt")
        circuit.measure(device[0], out)
        circuit.metadata = {"role": "qualification", "expected": expected}
        probes.append(circuit)
    phases = np.random.default_rng(PHASE_SEED).uniform(0, 2 * np.pi, max(CYCLES))
    for arm in ARMS:
        circuit = feedback_circuit(phases, table, arm, 0, gate_samples)
        circuit.name = f"recent_max_cycles_{arm}"
        circuit.metadata.update(role="length_qualification")
        probes.append(circuit)
    return probes


def experiment_batches(calibration, gate_samples):
    """Yield the four acquisition batches of the matched comparison.

    The (cycle number, phase sequence) pairs are shuffled and grouped into batches of
    72; the three circuit types of each pair stay in the same batch and are shuffled
    among themselves.
    """
    sequences = charging_phases()
    order = list(product(CYCLES, range(SEQUENCES)))
    rng = np.random.default_rng(INTERLEAVE_SEED)
    table = calibration["controller_table"]
    ordered = [order[i] for i in rng.permutation(len(order))]
    for start in range(0, len(ordered), TRIPLES_PER_BATCH):
        circuits = []
        for n, sequence in ordered[start : start + TRIPLES_PER_BATCH]:
            phases = sequences[sequence, :n]
            fixed = calibration["predictions"][str(n)]["fixed_flip"]
            for index in rng.permutation(len(ARMS)):
                arm = ARMS[index]
                circuit = feedback_circuit(phases, table, arm, fixed, gate_samples)
                circuit.name = f"recent_n{n:02d}_s{sequence:02d}_{arm}"
                circuit.metadata.update(
                    role="experiment", cycle_count=n, sequence=sequence, charging_phases=phases.tolist()
                )
                circuits.append(circuit)
        yield circuits
