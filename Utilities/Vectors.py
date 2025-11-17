import numpy as np
import math

def vec_sub(p2, p1):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    z_diff = p2[2] - p1[2]
    return [x_diff, y_diff, z_diff]

def vec_sub(p2, p1):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    z_diff = p2[2] - p1[2]
    return [x_diff, y_diff, z_diff]

def vec_add(p1, p2):
    return [a+b for a,b in zip(p1,p2)]

def vec_dot(p1, p2):
    return ((p1[0]*p2[0]) + (p1[1]*p2[1]) + (p1[2]*p2[2]))

def scalMul(vec, scal):
    return [scal*v for v in vec]

def elemMult(arr1, arr2):
    return [a*b for a,b in zip(arr1,arr2)]

def mag(vec) -> float:
    """
        Returns ||vec||
    """
    return math.sqrt(sum([v**2 for v in vec]))

def dot(p1, p2) -> float:
    """
        Returns P1 . P2
    """
    return sum([a * b for a, b in zip(p1, p2)])

def sub(p2, p1):
    """
        Returns P2 - P1
    """
    return [a-b for a,b in zip(p2, p1)]

def normalise(vec):
    #return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    m = mag(vec)
    if m == 0:
        print(m)
        print(vec)
    return [x/m for x in vec]

def cross(p1, p2):
    x = p1[1] * p2[2] - p1[2] * p2[1]
    y = p1[2] * p2[0] - p1[0] * p2[2]
    z = p1[0] * p2[1] - p1[1] * p2[0]
    return [x, y, z]

def sort_vertices_ccw(vertices):
    """Written by ChatGpt"""
    n = len(vertices)
    centroid = [sum([p[0] for p in vertices]) / n, sum([p[1] for p in vertices]) / n]
    vertices = sorted(vertices, key=lambda p: (math.atan2(p[1]-centroid[1], p[0]-centroid[0])+2*math.pi)%(2*math.pi))
    return np.array(vertices)
