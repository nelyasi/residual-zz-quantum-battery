"""Fixed design of the matched-feedback experiment on IBM Quantum Phoenix.

All values were fixed before data acquisition. The seeds reproduce the charging
phases, the circuit order and the calibration order recorded in the archived
manifests (see ``tests/test_design.py``).
"""

from math import asin, sqrt

BACKEND = "ibm_phoenix"

#: Physical qubits: battery Q86 followed by the spectators Q76, Q85, Q87 and Q96.
LAYOUT = (86, 76, 85, 87, 96)
SPECTATORS = 4
OUTCOMES_PER_CYCLE = 2**SPECTATORS

#: Residual ZZ shifts (kHz) and spectator readout axes (rad) from the independent
#: Ramsey calibration at a nominal sensing time of 9.984 us.
ZZ_SHIFTS_KHZ = (11.33739724, 13.82386326, 17.31167797, 9.43922829)
READOUT_AXES_RAD = (0.94316690, 0.93694510, 1.97599439, 0.73543977)

#: Charging pulses change the occupation of an energy eigenstate with probability Q.
Q = 0.10
THETA = 2 * asin(sqrt(Q))
TAU_US = 10.0

CYCLES = (1, 2, 4, 6, 8, 10, 12, 16, 20)
SEQUENCES = 32
SHOTS = 384
ARMS = ("before", "unconditional", "conditional")

#: The controller conditions on the most recent two spectator outcomes.
MAX_RECENT_CYCLES = 2
FEEDBACK_RULE = "bayes_recent_outcomes"

CALIBRATION_SEED = 416689
PHASE_SEED = 824191
INTERLEAVE_SEED = 745207

#: Circuits per acquisition batch (three matched circuit types per phase sequence).
TRIPLES_PER_BATCH = 72
