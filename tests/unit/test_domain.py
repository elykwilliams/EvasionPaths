from unittest import mock
from unittest.mock import patch

import pytest

from sensor_network import Sensor

from boundary_geometry import UnitCube
from motion_model import BilliardMotion

sensing_radius = 0.35
unit_cube = UnitCube(spacing=sensing_radius)
motion_model = BilliardMotion(domain=unit_cube)


class TestMotion:
    def test_reflection_detection(self):
        dt = 0.01
        velocity = motion_model.initial_vel(1)
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=velocity, sensing_radius=0.35)
        motion_model.update_position(sensor, dt)
        if sensor.pos not in unit_cube:
            in_domain = False
        else:
            in_domain = True

        assert in_domain == False
    def test_reflection_operation(self):
        dt = 0.1
        velocity = motion_model.initial_vel(1)
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=velocity, sensing_radius=0.35)
        motion_model.update_position(sensor, dt)
        if sensor.pos not in unit_cube:
            unit_cube.reflect(sensor)
        sensor_reflected = False
        for coordinate in sensor.pos:
            if coordinate < 1:
                sensor_reflected = True
        assert sensor_reflected
    def test_reflection_call(self):
        dt = 0.1
        velocity = motion_model.initial_vel(1)
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=velocity, sensing_radius=0.35)
        motion_model.local_update(sensor,dt)
        sensor_moved = False
        for coordinate in range(3):
            if sensor.pos[coordinate] != sensor.old_pos[coordinate]:
                sensor_moved = True
        assert sensor_moved


