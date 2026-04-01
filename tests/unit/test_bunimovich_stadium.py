import numpy as np

from boundary_geometry import BunimovichStadium


def test_horizontal_endcap_intersection_uses_circle():
    domain = BunimovichStadium(w=1.0, r=1.0, L=4.0)
    old_pos = np.array([0.9, 0.2], dtype=float)
    new_pos = np.array([2.0, 0.2], dtype=float)

    intersection = domain.get_intersection_point(old_pos, new_pos)

    expected_x = 1.0 + np.sqrt(1.0 - 0.2 ** 2)
    assert np.allclose(intersection, np.array([expected_x, 0.2]), atol=1e-9)


def test_vertical_stadium_intersection_uses_flat_segment():
    domain = BunimovichStadium(w=1.0, r=1.0, L=4.0)
    old_pos = np.array([0.25, 0.8], dtype=float)
    new_pos = np.array([0.25, 1.3], dtype=float)

    intersection = domain.get_intersection_point(old_pos, new_pos)

    assert np.allclose(intersection, np.array([0.25, 1.0]), atol=1e-9)
