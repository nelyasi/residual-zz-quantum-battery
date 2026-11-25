# Matched-feedback experiment, IBM Quantum Phoenix, 22 November 2026

Repeated charging and sensing followed by a fixed or a measurement-conditioned
extraction pulse (Sec. VIII B, Figs. 4 and 7, Table III of the manuscript).

| Folder | IBM job | Circuits x shots | Content |
|---|---|---|---|
| `qualification/` | `dc1i6f5hdvas73bks8d0` | 5 x 128 | Real-time branching probes and the three 20-cycle circuits |
| `calibration/` | `dc1i5its2omc738al6g0` | 64 x 384 | Battery assignment and single-cycle instrument circuits |
| `measurement/batch-00` to `batch-03` | see manifests | 4 x 216 x 384 | 9 cycle numbers x 32 phase sequences x 3 circuit types |

## Files

* `circuits.qpy` - circuits as compiled for the device (Qiskit QPY).
* `manifest.json` - backend, layout, shots per circuit, IBM job identifier, SHA-256
  of the compiled circuits and of the shot records, and the metadata of every
  circuit in submission order. Measurement batches also record the SHA-256 of the
  frozen calibration they were staged with.
* `shots.json` - one entry per circuit, in manifest order, with the circuit metadata
  and the bitstrings of each classical register:
  * `battery_out`: final battery readout;
  * `record`: the most recent spectator outcomes (one cycle, or the last two);
  * `history`: earlier spectator outcomes (circuits with more than two cycles).

  Bitstrings are as returned by Qiskit, with the first bit of a register on the
  right. In `record`, the integer value is the earlier outcome plus sixteen times
  the later one; within a cycle, bit `j` belongs to spectator `j` (Q76, Q85, Q87, Q96).
  Entries at the same index of different registers belong to the same shot.
* `calibration/frozen-calibration.json` - counts, readout response, outcome
  matrices, the extraction decision for every retained outcome and cycle number,
  and the model predictions. It was computed from `calibration/shots.json` before
  the measurement circuits were compiled.

## Circuit metadata

`cycle_count`, `sequence` and `arm` (`before`, `unconditional`, `conditional`)
identify each measurement circuit; `charging_phases` lists the phases in radians.
Sequence `s` with `N` cycles uses the first `N` phases of sequence `s`.
`fixed_flip` is the fixed extraction operation and `feedback_applied` states whether
the conditional circuit can apply X for some outcome.
