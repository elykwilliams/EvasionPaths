# ************************************************************
# Copyright (c) 2020, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

import unittest
from unittest import TestCase

from cycle_labelling import *


class TopologicalState:
    @classmethod
    def boundary_cycles(cls):
        return []

    @classmethod
    def simplices(cls, dim):
        return []

    @classmethod
    def simplex2cycle(cls, simplex):
        return simplex

    @classmethod
    def is_connected_cycle(cls, cycle):
        return True

    @classmethod
    def is_connected_simplex(cls, cycle):
        return True


if __name__ == '__main__':
    unittest.main()


class TestRemoveEdge(TestCase):
    def setUp(self) -> None:
        self.cycle_labelling = CycleLabelling(TopologicalState())
        self.removed_cycles = ["A", "B"]
        self.added_cycles = ["AA"]

    def test_adds_bcycles(self):
        self.cycle_labelling._cycle_label = {"A": True, "B": True}
        self.cycle_labelling._remove_1simplex(self.removed_cycles, self.added_cycles)
        self.assertIn("AA", self.cycle_labelling)

    def tests_removes_bcycles(self):
        self.cycle_labelling._cycle_label = {"A": True, "B": True}
        self.cycle_labelling._remove_1simplex(self.removed_cycles, self.added_cycles)
        self.assertNotIn("A", self.cycle_labelling)
        self.assertNotIn("B", self.cycle_labelling)

    def test_joins_clear_clear(self):
        self.cycle_labelling._cycle_label = {"A": True, "B": True}
        self.cycle_labelling._remove_1simplex(self.removed_cycles, self.added_cycles)
        self.assertEqual(self.cycle_labelling["AA"], True)

    def test_joins_clear_contaminated(self):
        self.cycle_labelling._cycle_label = {"A": True, "B": False}
        self.cycle_labelling._remove_1simplex(self.removed_cycles, self.added_cycles)
        self.assertEqual(self.cycle_labelling["AA"], True)

    def test_joins_contaminated_contaminated(self):
        self.cycle_labelling._cycle_label = {"A": False, "B": False}
        self.cycle_labelling._remove_1simplex(self.removed_cycles, self.added_cycles)
        self.assertEqual(self.cycle_labelling["AA"], False)

    def test_cannot_add_mult_bcycles(self):
        added_cycles = self.added_cycles + ["BB"]
        self.cycle_labelling._cycle_label = {"A": False, "B": False}

        with self.assertRaises(AssertionError):
            self.cycle_labelling._remove_1simplex(self.removed_cycles, added_cycles)

    def test_allow_remove_mult_bycycles(self):
        removed_cycles = self.removed_cycles + ["C"]
        self.cycle_labelling._cycle_label = {"A": False, "B": False}

        try:
            self.cycle_labelling._remove_1simplex(removed_cycles, self.added_cycles)
        except Exception:
            self.fail()


class TestAddEdge(TestCase):
    def SetUp(self):
        self.cycle_labelling = CycleLabelling(TopologicalState())
        self.new_cycles = ["AA", "BB"]
        self.old_cycles = ["A"]

    def test_adds_new_cycles(self):
        self.fail()

    def test_removes_old_cycles(self):
        self.fail()

    def test_splits_cleared(self):
        self.fail()

    def test_splits_contaminated(self):
        self.fail()

    def test_cant_split_mult_bcycles(self):
        self.fail()

    def test_cant_split_bcycle_into_mult(self):
        self.fail()


