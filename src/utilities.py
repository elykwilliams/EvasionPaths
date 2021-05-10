from numpy.linalg import norm
from numpy import cos, sin
from math import atan2


def cart2pol(p):
    return [norm(p), atan2(p[1], p[0])]


def pol2cart(p):
    return [p[0]*cos(p[1]), p[0]*sin(p[1])]


def set_difference(list1, list2):
    return list(set(list1).difference(set(list2)))


def is_subset(list1, list2):
    return set(list1).issubset(set(list2))


## Exception indicating that atomic transition not found.
# This should that two or more atomic transitions
# happen simultaneously. This is sometimes a problem for manufactured
# simulations. It can also indicate that a sensor has broken free of the
# virtual boundary and is interfering with the fence boundary cycle.
class MaxRecursionDepth(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self):
        return f"Max Recursion depth exceeded! \n\n{self.state_change}"


## Exception indicating non-atomic state change.
# This exception should be raised when a function that requires an atomic change
# is given a non-atomic change.
class InvalidStateChange(Exception):
    def __init__(self, state_change):
        self.state_change = state_change

    def __str__(self) -> str:
        return "Invalid State Change \n\n" \
               + str(self.state_change)
