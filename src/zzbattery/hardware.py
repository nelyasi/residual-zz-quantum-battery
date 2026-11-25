"""Staging, submission and retrieval on IBM Quantum.

Credentials are read from the environment variables ``IBM_QUANTUM_TOKEN`` and
``IBM_QUANTUM_INSTANCE`` or from a saved Qiskit Runtime account. Nothing in this
module is needed to reproduce the published analysis.

Every acquisition is staged to disk (compiled circuits and manifest) before
submission, and the job identifier is written to the manifest before results are
requested, so an interrupted run can be resumed without duplicate submissions.
"""

import json
import os
from pathlib import Path

from qiskit import qpy
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .archive import sha256_file, write_json
from .design import BACKEND, LAYOUT

TRANSPILER_SEED = 82
CONTROL_OPERATIONS = ("measure", "reset", "store", "if_else")


def connect():
    """Open the runtime service and check that the layout qubits are operational."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if token and instance:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(BACKEND)
    status, properties = backend.status(), backend.properties()
    faulty = sorted(set(properties.faulty_qubits()).intersection(LAYOUT))
    if not status.operational or faulty:
        raise RuntimeError(f"{BACKEND} is not operational or layout qubits are flagged faulty: {faulty}.")
    return service, backend


def x_gate_samples(backend):
    """Duration of the battery X gate in units of the backend sample time dt."""
    return max(1, round(backend.target["x"][(LAYOUT[0],)].duration / backend.dt))


def compile_for_backend(backend, circuits):
    """Compile at optimization level 0 on the fixed layout and check the control flow."""
    manager = generate_preset_pass_manager(
        optimization_level=0, backend=backend, initial_layout=list(LAYOUT), seed_transpiler=TRANSPILER_SEED
    )
    compiled = manager.run(circuits)
    for logical, physical in zip(circuits, compiled, strict=True):
        for operation in CONTROL_OPERATIONS:
            if logical.count_ops().get(operation, 0) != physical.count_ops().get(operation, 0):
                raise ValueError(f"Compilation changed the number of {operation} operations.")
        if any(item.operation.num_qubits == 2 and item.operation.name != "barrier" for item in physical.data):
            raise ValueError("The sensing sequence must use only the residual interaction.")
    return compiled


def stage(folder, circuits, kind, shots, calibration_sha256=None):
    """Write compiled circuits and a manifest; nothing is sent to the provider."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "circuits.qpy").open("wb") as stream:
        qpy.dump(circuits, stream)
    manifest = {
        "kind": kind,
        "backend": BACKEND,
        "layout": LAYOUT,
        "shots_per_circuit": shots,
        "calibration_sha256": calibration_sha256,
        "qpy_sha256": sha256_file(folder / "circuits.qpy"),
        "status": "staged",
        "job_id": None,
        "circuits": [circuit.metadata for circuit in circuits],
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def submit(backend, folder):
    """Submit staged circuits with dynamical decoupling and twirling disabled."""
    from qiskit_ibm_runtime import SamplerOptions, SamplerV2

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "staged" or manifest["job_id"]:
        raise RuntimeError(f"{folder.name} has already been submitted; retrieve it instead.")
    if sha256_file(folder / "circuits.qpy") != manifest["qpy_sha256"]:
        raise ValueError(f"{folder.name}: staged circuits fail the checksum.")
    with (folder / "circuits.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    options = SamplerOptions()
    options.dynamical_decoupling.enable = False  # decoupling would cancel the ZZ phase used for sensing
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False
    manifest["status"] = "submitting"
    write_json(folder / "manifest.json", manifest)
    job = SamplerV2(mode=backend, options=options).run(circuits, shots=manifest["shots_per_circuit"])
    manifest.update(status="submitted", job_id=job.job_id())
    write_json(folder / "manifest.json", manifest)
    print(f"Submitted {folder.name}: job {job.job_id()}", flush=True)


def retrieve(service, folder):
    """Download the classical registers of a submitted job into ``shots.json``."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest["job_id"]:
        raise ValueError(f"{folder.name} has no job identifier; check the provider job list.")
    result = service.job(manifest["job_id"]).result()
    rows = []
    for metadata, pub in zip(manifest["circuits"], result, strict=True):
        row = {"metadata": metadata, "battery_out": pub.data.battery_out.get_bitstrings()}
        for register in ("record", "history"):
            if hasattr(pub.data, register):
                row[register] = getattr(pub.data, register).get_bitstrings()
        for register in ("record", "history", "battery_out"):
            if register in row and len(row[register]) != manifest["shots_per_circuit"]:
                raise ValueError("The provider returned an unexpected number of shots.")
        rows.append(row)
    write_json(folder / "shots.json", rows)
    manifest.update(status="complete", shots_sha256=sha256_file(folder / "shots.json"))
    write_json(folder / "manifest.json", manifest)
    return rows


def run_stage(service, backend, folder, circuits=None, kind=None, shots=None):
    """Stage (if needed), submit (if needed) and retrieve one acquisition."""
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        if folder.exists() or circuits is None:
            raise RuntimeError(f"Incomplete staging in {folder}; inspect it before continuing.")
        stage(folder, compile_for_backend(backend, circuits), kind, shots)
    status = json.loads(manifest_path.read_text())["status"]
    if status == "staged":
        submit(backend, folder)
        status = "submitted"
    if status == "submitted":
        retrieve(service, folder)
        print(f"Retrieved {folder.name}", flush=True)
    elif status != "complete":
        raise RuntimeError(f"Unconfirmed submission in {folder.name}; check the provider job list.")
