"""Single-cycle daemonic extraction on IBM Quantum Phoenix (Table I and Fig. 5 of the manuscript).

Protocol (one sensing round, four spectators)
---------------------------------------------
    battery  B : charge with R(theta, phi) = exp[-i theta/2 (cos phi X + sin phi Y)]
    spectators : reset (fresh |0>) -> H -> |+>
    idle tau   : static residual ZZ coupling + decoherence (no two-qubit gate is applied)
    readout    : each spectator measured along |+-_alpha> = (|0> +- e^{i alpha}|1>)/sqrt(2)
                 ("Z" readout = computational basis), mid-circuit
    extraction : battery-only unitary
                   simple   (unconditional) : U            from the passive state of rho_bar
                   daemonic (conditional)   : U_m, m=0..15 from the passive state of rho_m,
                                              applied with dynamic circuits (if_test on the record)
    energy     : battery measured in Z,  H_B = hbar*omega |1><1|, energies in units of hbar*omega

Following Elyasi, Rossi & Genoni, Quantum Sci. Technol. 10, 025017 (2025), the extraction
unitaries are precomputed offline for every outcome string from two models of the circuit:
    "ideal" : only the ZZ coupling (zeta_j fitted from calibration), no decoherence, perfect readout
    "noisy" : spectator phases/contrasts measured at the protocol idle times, battery Ramsey phase,
              T1/T2 of the battery during idle and readout, ZZ kicks during the readout window,
              and measured readout confusion matrices
and applied on hardware as classically controlled operations. A third, data-driven set
("tomography") is obtained from the conditional battery tomography measured in the main job.

Conventions
-----------
    logical qubit 0 = battery, 1..4 = spectators (physical layout chosen in the notebook)
    outcome integer m = sum_j o_j 2^j, o_j = observed bit of spectator j (0 <-> |+_alpha>)
    battery state rho_{bb'} with rho_01 = <0|rho|1>; Bloch r = (2Re rho_01, -2Im rho_01, rho_00-rho_11)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.synthesis import OneQubitEulerDecomposer

TWO_PI = 2 * np.pi
NSPEC = 4
NOUT = 2**NSPEC
_EULER = OneQubitEulerDecomposer("U3")


# ============================================================================ helpers
def outcome_bits(m):
    return [(m >> j) & 1 for j in range(NSPEC)]


def outcome_label(m, spec_names):
    return " ".join(f"{n}={b}" for n, b in zip(spec_names, outcome_bits(m), strict=False))


def wrap(x):
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def delay_samples(tau_us, dt):
    """IBM delays must be multiples of 16 dt."""
    return int(round(tau_us * 1e-6 / dt / 16)) * 16


def actual_tau_us(tau_us, dt):
    return delay_samples(tau_us, dt) * dt * 1e6


def charger_phases(sc):
    """phases used to realise the charger: one for a coherent charger, K equally spaced
    phases (pooled with equal weights) for the stochastic, phase-random charger."""
    if sc["charger"] == "coherent":
        return [float(sc.get("phase", 0.0))]
    K = int(sc.get("n_phases", 8))
    return list(TWO_PI * np.arange(K) / K)


def r_gate(theta, phi):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s * np.exp(-1j * phi)], [-1j * s * np.exp(1j * phi), c]])


def bloch(rho):
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def rho_from_bloch(r):
    x, y, z = r
    return 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])


def energy(rho):
    return float(np.real(rho[1, 1]))


def ergotropy(rho):
    rho = 0.5 * (rho + rho.conj().T)
    w = np.linalg.eigvalsh(rho)
    return float(max(np.real(rho[1, 1]) - w[0], 0.0))


def passive_unitary(rho, tol=1e-10):
    """Unitary U mapping rho to its passive state (larger eigenvalue on |0>).
    Canonical representative: diagonal phases removed (they commute with H_B)."""
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    if w[1] - w[0] < tol:
        return np.eye(2, dtype=complex)
    U = np.array([v[:, 1].conj(), v[:, 0].conj()])
    d0 = np.angle(U[0, 0]) if abs(U[0, 0]) > 1e-9 else np.angle(U[0, 1])
    d1 = np.angle(U[1, 1]) if abs(U[1, 1]) > 1e-9 else np.angle(U[1, 0])
    U = np.diag([np.exp(-1j * d0), np.exp(-1j * d1)]) @ U
    if abs(U[0, 1]) < 1e-12 and abs(U[1, 0]) < 1e-12:
        return np.eye(2, dtype=complex)
    return U


def is_identity(U, tol=1e-9):
    return abs(U[0, 1]) < tol and abs(U[1, 0]) < tol


def u3_angles(U):
    th, ph, la = _EULER.angles(U)
    return float(th), float(ph), float(la)


def energy_after(rho, U):
    return energy(U @ rho @ U.conj().T)


# ============================================================================ credentials
def connect(channel="ibm_quantum_platform"):
    """Open the runtime service from IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE or a saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token, instance = os.environ.get("IBM_QUANTUM_TOKEN"), os.environ.get("IBM_QUANTUM_INSTANCE")
    if token and instance:
        return QiskitRuntimeService(channel=channel, token=token, instance=instance)
    return QiskitRuntimeService(channel=channel)


def backend_properties(backend, layout, default_t1=200.0, default_t2=150.0, default_ro_us=1.6):
    t = backend.target
    props = {}
    for q in layout:
        p = t.qubit_properties[q] if t.qubit_properties else None
        m = t["measure"].get((q,)) if "measure" in t.operation_names else None
        props[q] = dict(
            T1_us=(p.t1 * 1e6 if p is not None and p.t1 else default_t1),
            T2_us=(p.t2 * 1e6 if p is not None and p.t2 else default_t2),
            readout_error=(m.error if m is not None and m.error is not None else np.nan),
            readout_us=(m.duration * 1e6 if m is not None and m.duration else default_ro_us),
        )
    return props


# ============================================================================ circuits
def _idle(qc, tau_us, dt, emulate=None):
    if emulate is not None:  # offline testing hook (never used on hardware)
        emulate(qc, actual_tau_us(tau_us, dt))
        return
    n = delay_samples(tau_us, dt)
    if n > 0:
        qc.delay(n, list(range(5)), unit="dt")


def _rotate(qc, qubits, alphas):
    """map |+_alpha> -> |0> before a Z measurement (alphas=None: computational basis)."""
    if alphas is None:
        return
    for q, a in zip(qubits, alphas, strict=False):
        qc.p(-float(a), q)
        qc.h(q)


def build_calibration_circuits(waits_us, dt, cal_shots, ro_shots, emulate=None):
    """Job 1: readout confusion, spectator Ramsey with the battery in |0>/|1> (ZZ),
    and battery Ramsey with the spectators in |0>."""
    out = []
    for v in (0, 1):
        qc = QuantumCircuit(QuantumRegister(5, "q"), ClassicalRegister(5, "c"))
        if v:
            qc.x(range(5))
        qc.barrier()
        qc.measure(range(5), range(5))
        out.append((dict(kind="readout", prep=v), qc, ro_shots))
    for t in waits_us:
        for b in (0, 1):
            for basis, a in (("X", 0.0), ("Y", np.pi / 2)):
                q, s, bb = QuantumRegister(5, "q"), ClassicalRegister(4, "s"), ClassicalRegister(1, "b")
                qc = QuantumCircuit(q, s, bb)
                if b:
                    qc.x(0)
                for j in range(1, 5):
                    qc.h(j)
                qc.barrier()
                _idle(qc, t, dt, emulate)
                qc.barrier()
                _rotate(qc, range(1, 5), [a] * 4)
                qc.measure(range(1, 5), s)
                qc.measure(0, bb[0])
                out.append((dict(kind="zz", wait=float(t), battery=b, basis=basis), qc, cal_shots))
        for basis, a in (("X", 0.0), ("Y", np.pi / 2)):
            q, s, bb = QuantumRegister(5, "q"), ClassicalRegister(4, "s"), ClassicalRegister(1, "b")
            qc = QuantumCircuit(q, s, bb)
            qc.h(0)
            qc.barrier()
            _idle(qc, t, dt, emulate)
            qc.barrier()
            _rotate(qc, [0], [a])
            qc.measure(range(1, 5), s)
            qc.measure(0, bb[0])
            out.append((dict(kind="battery_ramsey", wait=float(t), basis=basis), qc, cal_shots))
    return out


def _protocol_front(theta, phi, tau_us, dt, alphas, emulate=None):
    q, s, bb = QuantumRegister(5, "q"), ClassicalRegister(4, "s"), ClassicalRegister(1, "b")
    qc = QuantumCircuit(q, s, bb)
    if abs(theta) > 1e-12:
        qc.r(float(theta), float(phi), 0)
    for j in range(1, 5):
        qc.h(j)
    qc.barrier()
    _idle(qc, tau_us, dt, emulate)
    qc.barrier()
    _rotate(qc, range(1, 5), alphas)
    qc.measure(range(1, 5), s)  # mid-circuit measurement of the spectators
    return qc, s, bb


def build_main_circuits(
    scenarios, taus_us, dt, alphas_tab, unitary_sets, feedback="dynamic", include_tomography=True, emulate=None
):
    """Job 2 (and optional Job 3).
    unitary_sets = {set_name: {"simple": {(sc, tau): U}, "daemonic": {(sc, tau): [U_0..U_15]}}}"""
    out = []
    for sc in scenarios:
        phases = charger_phases(sc)
        shots = max(1, int(sc["shots"]) // len(phases))
        for tau in taus_us:
            alphas = alphas_tab[(sc["name"], tau)]
            for k, phi in enumerate(phases):
                base = dict(scenario=sc["name"], tau=float(tau), phase_index=k, phi=float(phi))
                if include_tomography:
                    for basis, a in (("X", 0.0), ("Y", np.pi / 2), ("Z", None)):
                        qc, s, bb = _protocol_front(sc["theta"], phi, tau, dt, alphas, emulate)
                        if a is not None:
                            _rotate(qc, [0], [a])
                        qc.measure(0, bb[0])
                        out.append((dict(base, kind="tomo", basis=basis), qc, shots))
                for name, uset in unitary_sets.items():
                    U = uset["simple"][(sc["name"], tau)]
                    qc, s, bb = _protocol_front(sc["theta"], phi, tau, dt, alphas, emulate)
                    if not is_identity(U):
                        qc.u(*u3_angles(U), 0)
                    qc.measure(0, bb[0])
                    out.append((dict(base, kind="simple", set=name), qc, shots))
                    Us = uset["daemonic"][(sc["name"], tau)]
                    if feedback == "dynamic":
                        qc, s, bb = _protocol_front(sc["theta"], phi, tau, dt, alphas, emulate)
                        for m, Um in enumerate(Us):
                            if is_identity(Um):
                                continue
                            with qc.if_test((s, m)):
                                qc.u(*u3_angles(Um), 0)
                        qc.measure(0, bb[0])
                        out.append((dict(base, kind="daemonic", set=name), qc, shots))
                    else:  # post-selection: one circuit per outcome, keep matching shots
                        for m, Um in enumerate(Us):
                            qc, s, bb = _protocol_front(sc["theta"], phi, tau, dt, alphas, emulate)
                            if not is_identity(Um):
                                qc.u(*u3_angles(Um), 0)
                            qc.measure(0, bb[0])
                            out.append((dict(base, kind="daemonic_ps", set=name, ps_outcome=m), qc, shots))
    return out


# ============================================================================ raw data
def pub_shots(pub):
    """per-shot spectator outcome m (int) and battery bit, in shot order."""
    d = pub.data
    s = np.fromiter((int(x, 2) for x in d.s.get_bitstrings()), dtype=np.int64)
    b = np.fromiter((int(x, 2) for x in d.b.get_bitstrings()), dtype=np.int64)
    return s, b


def collect(meta_list, result):
    rows = []
    for meta, pub in zip(meta_list, result, strict=False):
        if meta["kind"] == "readout":
            bits = pub.data.c.get_bitstrings()
            arr = np.array([[int(ch) for ch in x[::-1]] for x in bits], dtype=np.int8)
            rows.append(dict(meta=meta, c=arr))
        else:
            s, b = pub_shots(pub)
            rows.append(dict(meta=meta, s=s, b=b))
    return rows


# ============================================================================ calibration analysis
def _corrected_expval(bits, e01, e10):
    z_obs = 1 - 2 * np.mean(bits)
    return (z_obs - (e10 - e01)) / max(1 - e01 - e10, 1e-6)


def analyse_calibration(rows, dt):
    ro = {r["meta"]["prep"]: r["c"] for r in rows if r["meta"]["kind"] == "readout"}
    e01 = ro[0].mean(0)  # P(read 1 | prepared 0), per logical qubit
    e10 = 1 - ro[1].mean(0)  # P(read 0 | prepared 1)
    waits = sorted({r["meta"]["wait"] for r in rows if r["meta"]["kind"] != "readout"})
    psi = np.zeros((NSPEC, 2, len(waits)))
    eta = np.zeros_like(psi)
    chi = np.zeros(len(waits))
    etaB = np.zeros(len(waits))
    ev = {}
    for r in rows:
        m = r["meta"]
        if m["kind"] == "zz":
            for j in range(NSPEC):
                bits = (r["s"] >> j) & 1
                ev[("zz", m["wait"], m["battery"], j, m["basis"])] = _corrected_expval(bits, e01[j + 1], e10[j + 1])
        elif m["kind"] == "battery_ramsey":
            ev[("br", m["wait"], m["basis"])] = _corrected_expval(r["b"], e01[0], e10[0])
    for i, t in enumerate(waits):
        for j in range(NSPEC):
            for b in (0, 1):
                X, Y = ev[("zz", t, b, j, "X")], ev[("zz", t, b, j, "Y")]
                psi[j, b, i] = np.arctan2(Y, X)
                eta[j, b, i] = min(np.hypot(X, Y), 1.0)
        X, Y = ev[("br", t, "X")], ev[("br", t, "Y")]
        chi[i] = np.arctan2(Y, X)
        etaB[i] = min(np.hypot(X, Y), 1.0)
    t_act = np.array([actual_tau_us(t, dt) for t in waits])
    dpsi = np.unwrap(wrap(psi[:, 1, :] - psi[:, 0, :]), axis=1)
    dom = np.array([np.polyfit(t_act, dpsi[j], 1)[0] for j in range(NSPEC)])  # rad/us
    omB = np.polyfit(t_act, np.unwrap(chi), 1)[0] if len(waits) > 1 else 0.0
    return dict(
        waits=waits,
        t_actual=t_act.tolist(),
        e01=e01.tolist(),
        e10=e10.tolist(),
        psi=psi.tolist(),
        eta=eta.tolist(),
        chi=chi.tolist(),
        etaB=etaB.tolist(),
        dpsi=dpsi.tolist(),
        domega=dom.tolist(),
        zeta_kHz=(-dom / TWO_PI * 1e3).tolist(),
        omegaB=float(omB),
    )


def calibration_table(cal, spec_names):
    rows = []
    for j, n in enumerate(spec_names):
        rows.append(
            dict(spectator=n, zeta_kHz=cal["zeta_kHz"][j], readout_e01=cal["e01"][j + 1], readout_e10=cal["e10"][j + 1])
        )
    return pd.DataFrame(rows)


def _wait_index(cal, tau):
    w = np.array(cal["waits"])
    i = int(np.argmin(abs(w - tau)))
    if abs(w[i] - tau) > 1e-9:
        raise ValueError(f"idle time {tau} us was not calibrated (calibrated: {cal['waits']})")
    return i


# ============================================================================ readout axes
def readout_axes(sc, tau, cal):
    mode = sc["spectator_readout"]
    if mode == "Z":
        return None
    if mode == "X":
        return [0.0] * NSPEC
    if mode == "Y":
        return [np.pi / 2] * NSPEC
    if mode != "opt":
        raise ValueError(mode)
    i = _wait_index(cal, tau)
    grid = np.linspace(-np.pi, np.pi, 3601)
    al = []
    for j in range(NSPEC):
        p0, p1 = cal["psi"][j][0][i], cal["psi"][j][1][i]
        e0, e1 = cal["eta"][j][0][i], cal["eta"][j][1][i]
        c = 0.5 * (e0 * np.cos(p0 - grid) - e1 * np.cos(p1 - grid))
        al.append(float(grid[np.argmax(c)]))
    return al


# ============================================================================ circuit model
def _spectator_vectors(alpha):
    if alpha is None:
        return [np.array([1, 0], complex), np.array([0, 1], complex)]
    e = np.exp(1j * alpha)
    return [np.array([1, e]) / np.sqrt(2), np.array([1, -e]) / np.sqrt(2)]


def _spectator_block(psi_b, psi_bp, eta, g1, t):
    """Spectator operator attached to the battery element |b><b'| after the idle time t.
    Off-diagonals: measured phases psi_b and contrast eta (already include T1/T2 decay).
    Diagonal: amplitude damping of the spectator; a spectator that relaxes from |1> at t' < t
    has given the battery only the fraction t'/t of the conditional ZZ phase, then reads 0.
        M11 = e^{iD} e^{-g1 t} / 2,
        M00 = 1/2 + (g1/2) int_0^t e^{iD t'/t - g1 t'} dt',     D = psi_b - psi_b'."""
    D = float(wrap(psi_b - psi_bp))
    m11 = 0.5 * np.exp(1j * D - g1 * t)
    if g1 * t < 1e-12:
        m00 = 0.5 + 0.5 * g1 * t
    else:
        k = 1j * D / t - g1
        m00 = 0.5 + 0.5 * g1 * (np.exp(k * t) - 1) / k
    return np.array([[m00, 0.5 * eta * np.exp(-1j * psi_bp)], [0.5 * eta * np.exp(1j * psi_b), m11]])


def model_parameters(cal, props, layout, model, t_ff_us=0.5):
    B = layout[0]
    return dict(
        model=model,
        cal=cal,
        T1B=props[B]["T1_us"],
        T2B=props[B]["T2_us"],
        T1S=[props[q]["T1_us"] for q in layout[1:]],
        T_ro=max(props[q]["readout_us"] for q in layout[1:]) + t_ff_us,
    )


def conditional_states(sc, tau, alphas, mp):
    """Unnormalised conditional battery states A[m] (2x2) for the 16 observed outcomes.
    sum_m A[m] = unconditional state rho_bar. Phases of a stochastic charger are pooled
    with equal weights (the controller does not know phi)."""
    cal, noisy = mp["cal"], mp["model"] == "noisy"
    i = _wait_index(cal, tau)
    t = cal["t_actual"][i]
    K = np.zeros((NSPEC, 2, 2, 2), complex)  # K[j, o, b, b']
    for j in range(NSPEC):
        a = None if alphas is None else alphas[j]
        vecs = _spectator_vectors(a)
        if noisy:
            ps = [cal["psi"][j][0][i], cal["psi"][j][1][i]]
            et = [cal["eta"][j][0][i], cal["eta"][j][1][i]]
            e01, e10 = cal["e01"][j + 1], cal["e10"][j + 1]
            kick = cal["domega"][j] * mp["T_ro"]
        else:
            ps = [0.0, cal["domega"][j] * t]
            et = [1.0, 1.0]
            e01 = e10 = 0.0
            kick = 0.0
        R = np.array([[1 - e01, e10], [e01, 1 - e10]])  # R[o, s]
        g1 = (1.0 / mp["T1S"][j]) if noisy else 0.0  # spectator relaxation during the idle time
        for b in (0, 1):
            for bp in (0, 1):
                ec = np.sqrt(et[b] * et[bp])
                M = _spectator_block(ps[b], ps[bp], ec, g1, t)
                for o in (0, 1):
                    K[j, o, b, bp] = sum(
                        R[o, s] * (vecs[s].conj() @ M @ vecs[s]) * np.exp(1j * (b - bp) * s * kick) for s in (0, 1)
                    )
    A = np.zeros((NOUT, 2, 2), complex)
    phases = charger_phases(sc)
    for phi in phases:
        C = r_gate(sc["theta"], phi)
        rho = C @ np.diag([1, 0]).astype(complex) @ C.conj().T
        if noisy:  # battery during the idle time: measured Ramsey phase/contrast, T1 on populations
            g = 1 - np.exp(-t / mp["T1B"])
            p1 = rho[1, 1].real * (1 - g)
            coh = rho[0, 1] * cal["etaB"][i] * np.exp(-1j * cal["chi"][i])
            rho = np.array([[1 - p1, coh], [coh.conjugate(), p1]])
        for m in range(NOUT):
            bits = outcome_bits(m)
            M = rho.copy()
            for b in (0, 1):
                for bp in (0, 1):
                    M[b, bp] *= np.prod([K[j, bits[j], b, bp] for j in range(NSPEC)])
            if noisy:  # readout + feed-forward window: T1, T2, battery free phase
                T = mp["T_ro"]
                g = 1 - np.exp(-T / mp["T1B"])
                tr = M[0, 0] + M[1, 1]
                p1 = M[1, 1] * (1 - g)
                coh = M[0, 1] * np.exp(-T / mp["T2B"]) * np.exp(-1j * cal["omegaB"] * T)
                M = np.array([[tr - p1, coh], [np.conj(coh), p1]])
            A[m] += M / len(phases)
    return A


# ============================================================================ unitaries + tables
def unitaries_from_states(A):
    """simple unitary from rho_bar and daemonic unitaries from the conditional states."""
    rho_bar = A.sum(0)
    U_simple = passive_unitary(rho_bar)
    U_d = []
    for m in range(NOUT):
        p = np.real(np.trace(A[m]))
        U_d.append(passive_unitary(A[m] / p) if p > 1e-12 else np.eye(2, dtype=complex))
    return U_simple, U_d


def _urow(U):
    th, ph, la = u3_angles(U)
    return dict(
        U00_re=U[0, 0].real,
        U00_im=U[0, 0].imag,
        U01_re=U[0, 1].real,
        U01_im=U[0, 1].imag,
        U10_re=U[1, 0].real,
        U10_im=U[1, 0].imag,
        U11_re=U[1, 1].real,
        U11_im=U[1, 1].imag,
        u3_theta=th,
        u3_phi=ph,
        u3_lambda=la,
    )


def _srow(rho):
    r = bloch(rho)
    return dict(
        r_x=r[0], r_y=r[1], r_z=r[2], purity_r=float(np.linalg.norm(r)), energy=energy(rho), ergotropy=ergotropy(rho)
    )


def unitary_tables(set_name, states, scenarios, taus, dt, alphas_tab, spec_names):
    """states[(sc, tau)] = A[16,2,2]. Returns (simple_df, daemonic_df, summary_df, unitary dict)."""
    simple, daemonic, summary = [], [], []
    uset = {"simple": {}, "daemonic": {}}
    for sc in scenarios:
        for tau in taus:
            A = states[(sc["name"], tau)]
            Us, Ud = unitaries_from_states(A)
            uset["simple"][(sc["name"], tau)] = Us
            uset["daemonic"][(sc["name"], tau)] = Ud
            rho_bar = A.sum(0)
            al = alphas_tab[(sc["name"], tau)]
            common = dict(
                set=set_name,
                scenario=sc["name"],
                charger=sc["charger"],
                theta_rad=sc["theta"],
                spectator_readout=sc["spectator_readout"],
                tau_us=tau,
                tau_actual_us=actual_tau_us(tau, dt),
                readout_axes_rad=("Z basis" if al is None else ", ".join(f"{a:.4f}" for a in al)),
            )
            simple.append(dict(common, **_srow(rho_bar), **_urow(Us)))
            W_D = 0.0
            for m in range(NOUT):
                p = float(np.real(np.trace(A[m])))
                rho_m = A[m] / p if p > 1e-12 else np.eye(2) / 2
                row = dict(
                    common,
                    outcome=m,
                    outcome_bits=outcome_label(m, spec_names),
                    P_outcome=p,
                    **_srow(rho_m),
                    **_urow(Ud[m]),
                )
                daemonic.append(row)
                W_D += p * row["ergotropy"]
            summary.append(
                dict(
                    common,
                    energy=energy(rho_bar),
                    ergotropy_unconditional=ergotropy(rho_bar),
                    daemonic_ergotropy=W_D,
                    daemonic_gain=W_D - ergotropy(rho_bar),
                )
            )
    return pd.DataFrame(simple), pd.DataFrame(daemonic), pd.DataFrame(summary), uset


README_SIMPLE = [
    "Simple (unconditional) extraction unitaries: U maps the unconditional battery state rho_bar",
    "(spectator outcomes ignored) to its passive state. Energies in units of hbar*omega, H_B = |1><1|.",
    "U_ij = <i|U|j> with i,j in {0,1} (|0> = ground). u3_* are the Qiskit U(theta, phi, lambda) angles",
    "applied on hardware (global phase dropped). Diagonal phases of U are removed (they commute with H_B).",
    "Bloch vector: r_x = 2Re rho_01, r_y = -2Im rho_01, r_z = rho_00 - rho_11.",
    "Sheets: ideal_model, noisy_model (computed before the main job), tomography (from the measured",
    "unconditional state, added after the main job).",
]
README_DAEMONIC = [
    "Daemonic (conditional) extraction unitaries: one U_m per observed spectator record m = 0..15,",
    "m = sum_j o_j 2^j with o_j the observed bit of spectator j (0 = projection on |+_alpha>).",
    "outcome_bits lists the bits by physical spectator. P_outcome is the model (or measured) probability.",
    "U_m maps the conditional battery state rho_m to its passive state; applied with dynamic circuits",
    "(if_test on the 4-bit record). Daemonic ergotropy = sum_m P_m * ergotropy(rho_m) (sheet summary_*).",
    "Same conventions as the simple-ergotropy file.",
]


def write_excel(path, sheets, readme):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"README": readme}).to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)


# ============================================================================ main-job analysis
def _mitigate_p1(bits, e01, e10):
    n = len(bits)
    if n == 0:
        return np.nan, np.nan, 0
    p_obs = float(np.mean(bits))
    den = max(1 - e01 - e10, 1e-6)
    p = min(max((p_obs - e01) / den, 0.0), 1.0)
    se = np.sqrt(max(p_obs * (1 - p_obs), 1.0 / n) / n) / den
    return p, se, n


def _physical(r):
    n = np.linalg.norm(r)
    return r / n if n > 1 else r


def analyse_main(rows, cal, scenarios, taus, model_states, model_usets, spec_names, min_shots=20):
    """Returns (summary_df, per_outcome_df, tomography_states)."""
    e01B, e10B = cal["e01"][0], cal["e10"][0]
    summary, per_out, tomo_states = [], [], {}
    for sc in scenarios:
        for tau in taus:
            sel = [r for r in rows if r["meta"].get("scenario") == sc["name"] and r["meta"].get("tau") == tau]
            if not sel:
                continue
            tomo = {bs: [r for r in sel if r["meta"]["kind"] == "tomo" and r["meta"]["basis"] == bs] for bs in "XYZ"}
            have_tomo = all(len(v) for v in tomo.values())
            row = dict(
                scenario=sc["name"], charger=sc["charger"], spectator_readout=sc["spectator_readout"], tau_us=tau
            )
            # ---------------- model predictions
            for mname, A in model_states.items():
                Am = A[(sc["name"], tau)]
                rb = Am.sum(0)
                row[f"{mname}_ergotropy_unconditional"] = ergotropy(rb)
                row[f"{mname}_daemonic_ergotropy"] = sum(
                    float(np.real(np.trace(Am[m]))) * ergotropy(Am[m] / np.real(np.trace(Am[m])))
                    for m in range(NOUT)
                    if np.real(np.trace(Am[m])) > 1e-12
                )
            if "noisy" in model_states:
                An = model_states["noisy"][(sc["name"], tau)]
                for sname, us in model_usets.items():
                    Us, Ud = us["simple"][(sc["name"], tau)], us["daemonic"][(sc["name"], tau)]
                    rb = An.sum(0)
                    row[f"predicted_W_simple[{sname}]"] = energy(rb) - energy_after(rb, Us)
                    row[f"predicted_W_daemonic[{sname}]"] = energy(rb) - sum(
                        float(np.real(np.trace(An[m]))) * energy_after(An[m] / np.real(np.trace(An[m])), Ud[m])
                        for m in range(NOUT)
                        if np.real(np.trace(An[m])) > 1e-12
                    )
            # ---------------- measured energy before extraction (Z tomography circuits)
            if have_tomo:
                s_all = np.concatenate([r["s"] for bs in "XYZ" for r in tomo[bs]])
                P_m = np.bincount(s_all, minlength=NOUT) / len(s_all)
                bz = np.concatenate([r["b"] for r in tomo["Z"]])
                sz = np.concatenate([r["s"] for r in tomo["Z"]])
                E0, E0se, _ = _mitigate_p1(bz, e01B, e10B)
                row.update(E_before=E0, E_before_se=E0se)
                rvec = []
                for bs in "XYZ":
                    bb = np.concatenate([r["b"] for r in tomo[bs]])
                    p, _, _ = _mitigate_p1(bb, e01B, e10B)
                    rvec.append(1 - 2 * p)
                rbar = _physical(np.array(rvec))
                rho_bar = rho_from_bloch(rbar)
                A_exp = np.zeros((NOUT, 2, 2), complex)
                for m in range(NOUT):
                    rv, ok = [], True
                    for bs in "XYZ":
                        bb = np.concatenate([r["b"][r["s"] == m] for r in tomo[bs]])
                        if len(bb) < min_shots:
                            ok = False
                        p, _, _ = _mitigate_p1(bb, e01B, e10B)
                        rv.append(1 - 2 * p if np.isfinite(p) else 0.0)
                    rho_m = rho_from_bloch(_physical(np.array(rv)))
                    A_exp[m] = P_m[m] * rho_m
                    Em, _, nz = _mitigate_p1(bz[sz == m], e01B, e10B)
                    per_out.append(
                        dict(
                            scenario=sc["name"],
                            tau_us=tau,
                            outcome=m,
                            outcome_bits=outcome_label(m, spec_names),
                            P_measured=P_m[m],
                            tomography_reliable=ok,
                            E_before=Em,
                            ergotropy_tomography=ergotropy(rho_m),
                            **{f"tomo_r_{c}": v for c, v in zip("xyz", bloch(rho_m), strict=False)},
                        )
                    )
                tomo_states[(sc["name"], tau)] = A_exp
                row["tomography_ergotropy_unconditional"] = ergotropy(rho_bar)
                row["tomography_daemonic_ergotropy"] = sum(
                    P_m[m] * ergotropy(A_exp[m] / P_m[m]) for m in range(NOUT) if P_m[m] > 0
                )
            else:
                P_m = None
                E0 = E0se = np.nan
            # ---------------- measured work
            sets = sorted({r["meta"]["set"] for r in sel if "set" in r["meta"]})
            for sname in sets:
                simp = [r for r in sel if r["meta"]["kind"] == "simple" and r["meta"]["set"] == sname]
                if simp:
                    E1, E1se, _ = _mitigate_p1(np.concatenate([r["b"] for r in simp]), e01B, e10B)
                    row[f"E_after_simple[{sname}]"] = E1
                    row[f"W_simple[{sname}]"] = E0 - E1
                    row[f"W_simple_se[{sname}]"] = np.hypot(E0se, E1se)
                dyn = [r for r in sel if r["meta"]["kind"] == "daemonic" and r["meta"]["set"] == sname]
                if dyn:
                    bb = np.concatenate([r["b"] for r in dyn])
                    ss = np.concatenate([r["s"] for r in dyn])
                    E1, E1se, _ = _mitigate_p1(bb, e01B, e10B)
                    row[f"E_after_daemonic[{sname}]"] = E1
                    row[f"W_daemonic[{sname}]"] = E0 - E1
                    row[f"W_daemonic_se[{sname}]"] = np.hypot(E0se, E1se)
                    for m in range(NOUT):
                        Em, _, n = _mitigate_p1(bb[ss == m], e01B, e10B)
                        per_out.append(
                            dict(
                                scenario=sc["name"],
                                tau_us=tau,
                                outcome=m,
                                set=sname,
                                outcome_bits=outcome_label(m, spec_names),
                                shots=n,
                                E_after_daemonic=Em,
                            )
                        )
                ps = [r for r in sel if r["meta"]["kind"] == "daemonic_ps" and r["meta"]["set"] == sname]
                if ps and P_m is not None:
                    Eaft, var = 0.0, 0.0
                    for m in range(NOUT):
                        bb = np.concatenate([r["b"][r["s"] == m] for r in ps if r["meta"]["ps_outcome"] == m])
                        Em, sem, n = _mitigate_p1(bb, e01B, e10B)
                        if n:
                            Eaft += P_m[m] * Em
                            var += (P_m[m] * sem) ** 2
                    row[f"E_after_daemonic[{sname}]"] = Eaft
                    row[f"W_daemonic[{sname}]"] = E0 - Eaft
                    row[f"W_daemonic_se[{sname}]"] = np.hypot(E0se, np.sqrt(var))
            summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(per_out), tomo_states


# ============================================================================ plotting
def plot_results(summary, scenarios, sets, path=None):
    import matplotlib.pyplot as plt

    n = len(scenarios)
    fig, axs = plt.subplots(1, n, figsize=(5.6 * n, 4.3), squeeze=False)
    for ax, sc in zip(axs[0], scenarios, strict=False):
        d = summary[summary.scenario == sc["name"]].sort_values("tau_us")
        t = d.tau_us.values
        if "noisy_ergotropy_unconditional" in d:
            ax.plot(
                t,
                d["noisy_ergotropy_unconditional"],
                "o-",
                color="#2ca02c",
                ms=7,
                label=r"$\mathcal{E}_{\rm unc}$ (noisy model)",
            )
            ax.plot(
                t,
                d["noisy_daemonic_ergotropy"],
                "^-",
                color="#d62728",
                ms=8,
                label=r"$\overline{\mathcal{E}}$ daemonic (noisy model)",
            )
        if "ideal_daemonic_ergotropy" in d:
            ax.plot(
                t,
                d["ideal_daemonic_ergotropy"],
                "^--",
                color="#f4a6b7",
                ms=7,
                mfc="none",
                label=r"$\overline{\mathcal{E}}$ daemonic (ideal model)",
            )
        if "tomography_daemonic_ergotropy" in d:
            ax.plot(
                t,
                d["tomography_ergotropy_unconditional"],
                "x",
                color="#555555",
                ms=7,
                label=r"$\mathcal{E}_{\rm unc}$ (tomography)",
            )
            ax.plot(
                t,
                d["tomography_daemonic_ergotropy"],
                "+",
                color="#555555",
                ms=9,
                label=r"$\overline{\mathcal{E}}$ (tomography)",
            )
        style = {"ideal": ("D", "#1f77b4"), "noisy": ("*", "#9467bd"), "tomography": ("s", "#8c564b")}
        for sname in sets:
            mk, col = style.get(sname, ("o", "k"))
            if f"W_daemonic[{sname}]" in d:
                ax.errorbar(
                    t,
                    d[f"W_daemonic[{sname}]"],
                    d[f"W_daemonic_se[{sname}]"],
                    fmt=mk,
                    color=col,
                    ms=9 if mk == "*" else 7,
                    capsize=3,
                    label=rf"$\overline{{\mathcal{{W}}}}$ experiment, $U^{{\rm {sname}}}_m$",
                )
            if f"W_simple[{sname}]" in d and sname == ("noisy" if "noisy" in sets else sets[0]):
                ax.errorbar(
                    t,
                    d[f"W_simple[{sname}]"],
                    d[f"W_simple_se[{sname}]"],
                    fmt="o",
                    color="#2ca02c",
                    mfc="none",
                    ms=9,
                    capsize=3,
                    label=rf"$\mathcal{{W}}_{{\rm unc}}$ experiment, $U^{{\rm {sname}}}$",
                )
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel(r"idle time $\tau$ ($\mu$s)")
        ax.set_ylabel(r"work / $\hbar\omega$")
        ax.set_title(f"{sc['name']}: {sc['charger']} charger, {sc['spectator_readout']} readout", fontsize=10)
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170)
    return fig


# ============================================================================ persistence
def _keyed(d):
    return {f"{k[0]}|{k[1]}": v for k, v in d.items()}


def save_unitary_sets(path, usets):
    out = {}
    for name, u in usets.items():
        out[name] = {
            "simple": {k: [[[z.real, z.imag] for z in row] for row in U] for k, U in _keyed(u["simple"]).items()},
            "daemonic": {
                k: [[[[z.real, z.imag] for z in row] for row in U] for U in Us]
                for k, Us in _keyed(u["daemonic"]).items()
            },
        }
    json.dump(out, open(path, "w"))


def load_unitary_sets(path):
    raw = json.load(open(path))

    def dec(M):
        return np.array([[a + 1j * b for a, b in row] for row in M])

    out = {}
    for name, u in raw.items():

        def key(k):
            return (k.split("|")[0], float(k.split("|")[1]))

        out[name] = {
            "simple": {key(k): dec(v) for k, v in u["simple"].items()},
            "daemonic": {key(k): [dec(M) for M in v] for k, v in u["daemonic"].items()},
        }
    return out


def estimate_qpu_seconds(circuits, tau_overhead_us=8.0, rep_delay_us=250.0):
    tot = 0.0
    for meta, _qc, shots in circuits:
        t = meta.get("tau", meta.get("wait", 0.0)) or 0.0
        tot += shots * (t + tau_overhead_us + rep_delay_us) * 1e-6
    return tot
