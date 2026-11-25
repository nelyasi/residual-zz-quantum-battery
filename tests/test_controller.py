"""The real-time predicate implements the frozen decision table exactly."""

import numpy as np
import pytest

from zzbattery.archive import read_json
from zzbattery.calibration import freeze_controller
from zzbattery.circuits import _predicate_terms


def evaluate(terms, negate, value):
    hit = any(value & mask == target for mask, target in terms)
    return not hit if negate else hit


@pytest.mark.parametrize("cycles", ["1", "2", "20"])
def test_predicate_matches_table(data, cycles):
    table = read_json(data / "calibration" / "frozen-calibration.json")["controller_table"][cycles]
    width = int(np.log2(len(table)))
    negate, terms = _predicate_terms(tuple(bool(x) for x in table), width)
    assert [evaluate(terms, negate, value) for value in range(len(table))] == [bool(x) for x in table]


def test_random_tables():
    rng = np.random.default_rng(1)
    for _ in range(50):
        table = tuple(bool(x) for x in rng.integers(0, 2, 16))
        negate, terms = _predicate_terms(table, 4)
        assert [evaluate(terms, negate, value) for value in range(16)] == list(table)


def test_controller_flips_only_when_excited_is_likely():
    identity = np.zeros((16, 2, 2))
    identity[0] = np.eye(2)
    table, forecasts = freeze_controller(identity, cycles=(1, 2))
    assert not any(table["1"]) and forecasts["1"]["conditional_work"] == 0.0
