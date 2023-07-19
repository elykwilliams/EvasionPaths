# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************

from unittest import mock

import numpy as np

from boundary_geometry import UnitCube
from motion_model import BilliardMotion
from sensor_network import Sensor


class TestReflection:
    """ These tests test that a sensor that the reflection are happening as expected"""
    r = 0.35
    dt = 0.01
    domain = UnitCube()
    motion_model = BilliardMotion(domain=domain)

    def test_reflection_detection(self):
        """
            Test that we detect when a sensor is move out of the domain
            Technically this is a test on the Square reflector
        """
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=[1, 0, 0], sensing_radius=self.r)
        self.motion_model.update_position(sensor, dt=self.dt)
        assert sensor.pos not in self.domain

    def test_reflection_operation(self):
        """
            Test that Domain.reflect results in sensor that is inside the domain
            Should parametrize this to check all faces
        """
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=[1, 0, 0], sensing_radius=self.r)
        self.motion_model.update_position(sensor, dt=self.dt)
        self.domain.reflect(sensor)
        assert sensor.pos in self.domain

    def test_reflection_called(self):
        """
            Test that motion model update is reflecting the sensors
        """
        unit_cube = UnitCube()
        unit_cube.reflect = mock.Mock(return_value=np.array([0.99, 0.99, 0.99]))
        motion_model = BilliardMotion(domain=unit_cube)
        sensor = Sensor(position=[0.999, 0.999, 0.999], vel=[1, 0, 0], sensing_radius=self.r)
        motion_model.local_update(sensor, self.dt)

        unit_cube.reflect.assert_called()



