from math import sqrt
from math import acos
import numpy as np
import json
from math import sin, cos

def rotate_about_axis(vertices: np.array, theta : float, u : np.array):
    '''
        Implemented from https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

        Maybe similar to this - https://math.stackexchange.com/questions/856666/how-can-i-transform-a-3d-triangle-to-xy-plane ? 
        Deffo similar to this - https://www.gamedev.net/forums/topic/27456-rotating-a-triangle-onto-the-x-y-plane/

        Params
        ------
        vertices : np.array
            list of vertices to rotate (Nx3)
        theta : float
            angle of rotation, in radians
        u : np.array
            axis of rotation (1x3) vector ux, uy, uz components are the rotation axis. 
        Returns
        -------
        new_vertices : np.array
            the rotated vertices
    '''
    ux = u[0]
    uy = u[1]
    uz = u[2]

    c = cos(theta)
    s = sin(theta)

    R11 = c + (ux * ux * (1 - c))
    R12 = (ux * uy * (1 - c)) - (uz * s)
    R13 = (ux * uz * (1 - c)) + (uy * s)

    R21 = (uy * ux * (1 - c)) + (uz * s)
    R22 = c + (uy * uy * (1 - c))
    R23 = (uy * uz * (1 - c)) - (ux * s)

    R31 = (uz * ux * (1 - c)) - (uy * s)
    R32 = (uz * uy * (1 - c)) + (ux * s)
    R33 = c + (uz * uz * (1 - c))
    
    new_vertices = []
    for v in vertices:
        x,y,z = v
        xp = (R11 * x) + (R12 * y) + (R13 * z)
        yp = (R21 * x) + (R22 * y) + (R23 * z)
        zp = (R31 * x) + (R32 * y) + (R33 * z)
        new_vertices.append(np.array([xp, yp, zp]))
    return np.array(new_vertices)


def angle_3n_3d(p1:np.array, p2:np.array, p3:np.array) -> float:
    # try px -> np.array incase theyre not arrays
    # exit gracefully if not
    a = p2 - p1
    b = p3 - p1
    return acos((dot(a,b)) / (mag(a) * mag(b)))

def dot(p1: np.array, p2: np.array) -> float:
    return sum([x*y for x,y in zip(p1,p2)])

def mag(p1: np.array) -> float:
    return sqrt(sum([x**2 for x in p1]))

def dist_between_points(p1, p2):
    return mag(p2-p1)

def centroid_of_vertices(vertices):
    """
        Calculates centroid (average position) of a set of nodes.For a uniform set of nodes, this is the center
        point. For a non-uniform set, this is weighted towards the higher density of 
    """
    average_x = 0.0
    average_y = 0.0
    average_z = 0.0
    for v in vertices:
        average_x += v[0]
        average_y += v[1]
        average_z += v[2]
    average_x = average_x/len(vertices)
    average_y = average_y/len(vertices)
    average_z = average_z/len(vertices)
    return (average_x, average_y, average_z)

def center_point_of_vertices(vertices):
    """
        Calculates center of the bounding box. This method is independent of mesh uniformity
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds(vertices)
    cp_x = xmin + ((xmax - xmin)/2.0)
    cp_y = ymin + ((ymax - ymin)/2.0)
    cp_z = zmin + ((zmax - zmin)/2.0)
    return cp_x, cp_y, cp_z


def bounds(vertices):
    v_array = np.array(vertices)
    v_array = v_array.T
    xes = v_array[0]
    yes = v_array[1]
    zes = v_array[2]
    xmin, xmax, ymin, ymax, zmin, zmax = min(xes), max(xes), min(yes), max(yes), min(zes), max(zes)
    return xmin, xmax, ymin, ymax, zmin, zmax