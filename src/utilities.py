from numpy.linalg import norm
from numpy import cos, sin
from math import atan2


def cart2pol(p):
    return [norm(p), atan2(p[1], p[0])]


def pol2cart(p):
    return [p[0]*cos(p[1]), p[0]*sin(p[1])]
