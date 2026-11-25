# Single-cycle experiment, IBM Quantum Phoenix, 17 November 2026

One charging and sensing cycle followed by a fixed or a measurement-conditioned
extraction pulse (Sec. VIII A, Table I and Fig. 5 of the manuscript). Battery Q86,
spectators Q76, Q85, Q87 and Q96. Acquired with
`notebooks/single_cycle_experiment.ipynb` and `zzbattery.single_cycle`.

| File | Content |
|---|---|
| `manifest.json` | Design (sensing times, scenarios, shots), qubit properties and IBM job identifiers |
| `job1-metadata.json`, `job2-metadata.json`, `job3-metadata.json` | Circuit metadata of the calibration, main and tomography-unitary jobs |
| `calibration.json` | Readout confusion, Ramsey phases and contrasts, fitted ZZ shifts |
| `readout_axes.json` | Spectator readout axes per scenario and sensing time |
| `unitaries_model.json`, `unitaries_tomography.json` | Extraction unitaries for every outcome (ideal, noisy and tomography models) |
| `summary.csv` | Energies and work per scenario and sensing time |
| `per-outcome.csv`, `per-outcome-tomography-unitaries.csv` | Outcome-resolved populations, tomography and energies |
| `spectator-calibration.csv` | ZZ shifts and readout errors per spectator |

Scenario P2 (charging phases uniformly random over eight values, optimal spectator
axes) is the experiment reported in the manuscript; the physical feedback uses the
`noisy` extraction set. The shot records were reduced to the outcome-resolved
counts in `per-outcome.csv` at acquisition time.
