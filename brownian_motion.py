from itertools import filterfalse
from numpy import random, sqrt



def at_boundary(point):
    return point[0]<=0 or point[1]<=0 or point[0]>=1 or point[1]>=1


def epsilon():
    sigma = 0.05
    dt = 0.01
    return sigma*sqrt(dt)*random.normal(0, 1)


def update_position(points):
    boundary = list(filter(at_boundary, points))
    interior = filterfalse(at_boundary, points)

    interior = [(x+epsilon(), y+epsilon()) for (x, y) in interior]
    for i, point in enumerate(interior):
        if point[0] >= 1:
            interior[i] = (2 - point[0] - epsilon(), point[1])
        if point[0] <= 0:
            interior[i] = (-point[0] + epsilon(), point[1])
        if point[1] >= 1:
            interior[i] = (point[0], 2 - point[1] - epsilon())
        if point[0] <= 0:
            interior[i] = (point[0], -point[1] + epsilon())

    return boundary + interior
            

        
    
