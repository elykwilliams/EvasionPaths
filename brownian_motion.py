from itertools import filterfalse
from numpy import random, sqrt


def at_boundary(point):
    return point[0] <= 0 or point[1] <= 0 or point[0] >= 1 or point[1] >= 1


def epsilon():
    sigma = 0.05
    dt = 0.01
    return sigma*sqrt(dt)*random.normal(0, 1)


def update_position(points):
    boundary = list(filter(at_boundary, points))
    interior = filterfalse(at_boundary, points)

    interior = [(x+epsilon(), y+epsilon()) for (x, y) in interior]
    for i, (x, y) in enumerate(interior):
        if x >= 1:
            interior[i] = (2 - x - epsilon(), y)
        if x <= 0:
            interior[i] = (-x + epsilon(), y)
        if y >= 1:
            interior[i] = (x, 2 - y - epsilon())
        if y <= 0:
            interior[i] = (x, -y + epsilon())

    return boundary + interior
