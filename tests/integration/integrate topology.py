from typing import Hashable
from unittest import mock

BoundaryCycle = frozenset


def CombinatorialMap(cycle_dict):
    m = mock.Mock()
    m.dict = cycle_dict
    m.boundary_cycles = m.dict.values()
    m.alpha.side_effect = lambda dart: dart[-1::-1]
    m.get_cycle.side_effect = lambda dart: next(key for key, value in m._dict.items() if dart in value)
    return m


def test_create_cmap():
    d = {"A": frozenset({"AE", "AB", "AC"}),
         "B": frozenset({"BD", "BA", "BC", "BE"}),
         "C": frozenset({"CB", "CD", "CE"})}
    cmap = CombinatorialMap(d)
    assert isinstance(list(cmap.boundary_cycles)[0], Hashable)
