import numpy as np

from boundary_geometry import BunimovichStadium, RectangularDomain
from motion_model import Viscek
from sensor_network import Sensor, SensorNetwork


def test_viscek_advances_with_synchronized_heading():
    domain = RectangularDomain(x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0)
    motion_model = Viscek(large_dt=1.0, radius=1.0, noise_scale=0.0)

    sensor_a = Sensor(np.array([0.0, 0.0]), np.array([1.0, 0.0]), sensing_radius=0.1)
    sensor_b = Sensor(np.array([0.1, 0.0]), np.array([0.0, 1.0]), sensing_radius=0.1)
    network = SensorNetwork([sensor_a, sensor_b], motion_model, fence=[], sensing_radius=0.1, domain=domain)

    network.move(1.0)

    expected_disp = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(sensor_a.pos, expected_disp, atol=1e-9)
    assert np.allclose(sensor_b.pos, np.array([0.1, 0.0]) + expected_disp, atol=1e-9)


def test_bunimovich_boundary_anchor_handles_grazing_start():
    domain = BunimovichStadium(w=1.0, r=1.0, L=4.0)
    old_pos = np.array([0.25, 1.0], dtype=float)
    new_pos = np.array([0.45, 1.0 + 1.0e-10], dtype=float)

    intersection = domain.get_intersection_point(old_pos, new_pos)

    assert np.allclose(intersection, old_pos, atol=1e-8)
