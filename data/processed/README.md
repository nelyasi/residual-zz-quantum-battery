# Processed data

Tables behind the figures and tables of the manuscript that are not regenerated
from the raw shots of the matched-feedback campaign.

| File | Content | Source |
|---|---|---|
| `table1-single-cycle.csv` | Final excited-state counts without and with feedback after one cycle (Table I) | `../phoenix-2026-11-17-single-cycle/summary.csv`, scenario P2, extraction set `noisy` |
| `fig5a-populations-by-outcome.csv` | Battery excitation for each spectator outcome in the reference Z circuit and in the feedback circuit at 19.968 us (Fig. 5a) | `../phoenix-2026-11-17-single-cycle/per-outcome.csv` |
| `fig5b-damping-bound.csv` | Parameters of the amplitude-damping bound (Fig. 5b, Eq. L1) | Table I and the archived battery T1 |
| `fig6-ramsey-check.csv` | Differential Ramsey phases, weighted fits and the smaller Ramsey amplitude per spectator and delay (Fig. 6) | Independent Ramsey sweep, Appendix M |
| `table2-ramsey-shifts.csv` | Fitted residual ZZ shifts (Table II) | Independent Ramsey sweep, Appendix M |

Populations are corrected for the battery readout response with the assignment
rates of the independent calibration, P(1|0) = 0.0015 and P(0|1) = 0.0048.
Standard errors are binomial counting errors with these rates held fixed.

The Ramsey sweep (Fig. 6, Table II) is provided at the level of the fitted phases
and amplitudes; it was analysed with `zzbattery.ramsey.fit_differential_ramsey`.
