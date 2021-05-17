import unittest

from cycle_labelling import CycleLabellingTree


class TopologyStub:
    def __init__(self, boundary_cycles, simplices2):
        self.boundary_cycles = boundary_cycles
        self.simplices2 = simplices2
        self.alpha_cycle = "alpha"


class TestInitCycleLabelling(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = TopologyStub(["A", "B", "C", "D", "E"], ["B", "C"])
        self.cycle_labelling = CycleLabellingTree(self.topology)

    # All boundary cycles are added to the labelling
    def test_all_cycles_in_tree(self):

        for cycle in self.topology.boundary_cycles:
            self.assertIn(cycle, self.cycle_labelling.tree)

    # Make sure alpha cycle is set as root
    def test_alpha_cycle_is_root(self):
        self.assertEqual(self.cycle_labelling.tree.root, self.topology.alpha_cycle)

    ## test all simplices are false
    def test_simplices_false(self):
        for node in self.cycle_labelling.tree.children(self.topology.alpha_cycle):
            if node.identifier in self.topology.simplices2:
                self.assertEqual(node.data, False)
            else:
                self.assertEqual(node.data, True)

    ## Test all non-simplices are true
    def test_non_simplices_true(self):
        for node in self.cycle_labelling.tree.children(self.topology.alpha_cycle):
            if node.identifier not in self.topology.simplices2:
                self.assertEqual(node.data, True)
            else:
                self.assertEqual(node.data, False)
