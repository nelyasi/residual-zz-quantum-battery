"""Daemonic work extraction from a transmon quantum battery monitored via residual ZZ coupling.

The package contains the circuit construction, calibration, analysis and hardware
interfaces used for the experiments on IBM Quantum Phoenix reported in

    S. N. Elyasi, "Daemonic work extraction from a quantum battery monitored via
    residual ZZ coupling" (2026).

The offline analysis never contacts IBM Quantum; only :mod:`zzbattery.hardware`
and :mod:`zzbattery.single_cycle` require an account.
"""

__version__ = "1.0.0"
