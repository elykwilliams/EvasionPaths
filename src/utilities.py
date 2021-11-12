# ************************************************************
# Copyright (c) 2021, Kyle Williams - All Rights Reserved.
# You may use, distribute and modify this code under the
# terms of the BSD-3 license. You should have received a copy
# of the BSD-3 license with this file.
# If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ************************************************************

from abc import ABC
from math import atan2

import numpy
from numpy import cos, sin
from numpy.linalg import norm


## Convert Cartesian coordinates to polar.
def cart2pol(p: numpy.array) -> list:
    return [norm(p), atan2(p[1], p[0])]


## Convert Polar coordinates to cartesian.
def pol2cart(p: list) -> list:
    return [p[0] * cos(p[1]), p[0] * sin(p[1])]


## Compute the set theoretic difference between two lists.
def set_difference(list1: list, list2: list) -> list:
    return [x for x in list1 if x not in list2]


## Determine if list is subset list.
def is_subset(list1: list, list2: list) -> bool:
    return set(list1).issubset(set(list2))


## Base Exception Class.
# All errors relating to an evasion path simulation should be derived from this class.
# User scripts should only need to catch this class.
class EvasionPathError(Exception, ABC):
    message = f'Error! An internal error has been detected.\n\n'


## Exception indicating that atomic transition not found.
# This should that two or more atomic transitions
# happen simultaneously. This is sometimes a problem for manufactured
# simulations. It can also indicate that a sensor has broken free of the
# virtual boundary and is interfering with the fence boundary cycle.
class MaxRecursionDepth(EvasionPathError):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return (
            f'{self.message}Max Recursion depth exceeded!\n\n{self.state_change}.\n\n'
            f'This exception was raised because the adaptive timestep was unable to resolve' 
            f'a small enough time step so that the topological change is atomic. This may be ' 
            f'because your timestep was too large. It can also often be likely in manufactured ' 
            f'simulations. It could also indicate that a sensor has left the domain and is ' 
            f'interacting with the fence sensors.'
        )


## Exception indicating non-atomic state change.
# This exception should be raised when a function that requires an atomic change
# is given a non-atomic change.
class InvalidStateChange(EvasionPathError):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self) -> str:
        return (
            f'{self.message}Invalid State Change\n\n{self.state_change}.\n'
            'This exception occurs when a function that requires an atomic change'
            'is given a non-atomic change. All exceptions of this sort should be handled '
            'internally.'
        )


## Exception indicating that missing boundary cycle.
# This exception should be raised when an boundary cycle that should be in
# the cycle labelling is not.
class CycleNotFound(EvasionPathError):
    def __init__(self, boundary_cycle):
        self.b = boundary_cycle

    def __str__(self):
        return (
            f'{self.message}Attempted to retrieve labelling for {self.b}, '
            'but this cycle was not found in the cycle labelling.\n'
            'This most likely has occurred because you are updating '
            'the boundary cycle labelling manually and not using the update() function.'
        )


## Exception to be used when simulation unable to initialize.
class InitializationError(EvasionPathError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"{self.message}Unable to initialize for the following reason:\n{self.msg}"


## Exception to be raised by SIGALERT.
# Use this exception when you only allow a simulation to run for a fixed amount of time.
class TimedOutExc(Exception):
    pass


## Exception to be raised if simulation is over.
# When creating animations, we specify the number frames to show. If the simulation
# finishes before the given number of frames, throw this exception to exit out of the
# animation loop.
class SimulationOver(Exception):
    pass
