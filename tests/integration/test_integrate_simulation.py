from unittest import mock

from combinatorial_map import BoundaryCycle2D
from time_stepping import EvasionPathSimulation


class TestAdd2Simplex:
    def test_init_simulation(self):
        sensor_network = mock.Mock()
        sensor_network.points = [(1.5, 0.5), (1, 1), (0.5, 1.6), (0, 1), (0, 0), (1, 0)]
        sensor_network.sensing_radius = 0.5
        sim = EvasionPathSimulation(sensor_network, 0.01, 10)

        c1 = frozenset({(0, 5), (5, 1), (1, 0)})
        c2 = frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)})
        c3 = frozenset({(2, 1), (1, 3), (3, 2)})
        c4 = frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
        assert sim.cycle_label.dict == {BoundaryCycle2D(c1): False,
                                        BoundaryCycle2D(c2): False,
                                        BoundaryCycle2D(c3): True,
                                        BoundaryCycle2D(c4): True}

    def test_do_single_timestep(self):
        sensor_network = mock.Mock()
        sensor_network.points = [(1.5, 0.5), (1, 1), (0.5, 1.6), (0, 1), (0, 0), (1, 0)]
        sensor_network.sensing_radius = 0.5
        sim = EvasionPathSimulation(sensor_network, 0.01, 10)

        sensor_network.points = [(1.5, 0.5), (1, 1), (0.5, 1.5), (0, 1), (0, 0), (1, 0)]
        sim.do_timestep()

        c1 = frozenset({(0, 5), (5, 1), (1, 0)})
        c2 = frozenset({(5, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)})
        c3 = frozenset({(2, 1), (1, 3), (3, 2)})
        c4 = frozenset({(3, 1), (1, 5), (5, 4), (4, 3)})
        assert sim.cycle_label.dict == {BoundaryCycle2D(c1): False,
                                        BoundaryCycle2D(c2): False,
                                        BoundaryCycle2D(c3): False,
                                        BoundaryCycle2D(c4): True}
