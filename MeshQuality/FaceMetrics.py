import json
from math import asin, sqrt, acos, degrees
from matplotlib import pyplot as plt
import os, sys

import numpy as np
import scipy as sp

sys.path.append(os.path.dirname("FileRW"))
sys.path.append(os.path.dirname("Utilities"))
from FileRW.Mesh import Mesh
from Utilities.PointClouds import dist_between_points, rotate_about_axis
from Utilities.Vectors import cross
from Utilities.Vectors import mag
from Utilities.Vectors import dot
from Utilities.Vectors import sub
from Utilities.Vectors import normalise
from Utilities.Vectors import sort_vertices_ccw

def face_aspect_ratio(ff : Mesh, ignore=[1,2,3,4]):
    ars = []
    for t in ff.boundary_triangles:
        if t[3] not in ignore:
            p1 = ff.nodes[t[0]]
            p2 = ff.nodes[t[1]]
            p3 = ff.nodes[t[2]]
            ars.append(_calc_aspect_ratio(p1, p2, p3))
    return ars

def _calc_aspect_ratio(p1, p2, p3):
    '''
        https://stackoverflow.com/questions/10289752/aspect-ratio-of-a-triangle-of-a-meshed-surface
    '''
    a = dist_between_points(p1,p2)
    b = dist_between_points(p2,p3)
    c = dist_between_points(p3,p1)

    s = (a+b+c)/2
    ar = (a*b*c)/(8*(s-a)*(s-b)*(s-c))

    return ar

# f_size (3)
def f_size_3(ff: Mesh, ignore=[1,2,3,4], ff_r: Mesh = None):
    """
        Calculates relative size for each boundary triangle in ff. 

        If ff_r is provided, the reference area used is the triangels 
        area in the baseline mesh
        If not, the average area of all triangles in ff is used. 
    """

    ff_areas = f_area_3(ff, ignore)
    if ff_r == None:
        ref_areas = [sum(ff_areas)/len(ff_areas)] * len(ff_areas)
    else:
        ref_areas = f_area_3(ff_r, ignore)
    
    taus = [a/b for a,b in zip(ff_areas, ref_areas)]
    f_size = [min([t, 1.0/t]) for t in taus]
    return f_size

def f_area_3(ff : Mesh, ignore=[1,2,3,4]):
    areas = []
    for t in ff.boundary_triangles:
        if t[3] in ignore:
            continue
        p1,p2,p3 = ff.nodes[t[0]], ff.nodes[t[1]], ff.nodes[t[2]]
        a1 = _calc_area_3(p1,p2,p3)
        areas.append(a1)
    return areas

def _calc_area_3(p1, p2, p3):
    '''
        https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
    '''
    A = sub(p2,p1)
    B = sub(p3,p1)
    return 0.5 * mag(cross(A, B))

# f_shape (3)
# f_skew (3)
def f_skew_3(ff : Mesh, ignore=[1,2,3,4], shift=True, reduce_dim=False):
    skews = [] # called shape in the og paper...
    gl_to_lc, vertices, faces_3, faces_4 = ff.get_multiple_surface_vertices_and_faces(ignore=ignore)
    for f in faces_3:
        p1, p2, p3 = vertices[f[1]], vertices[f[2]], vertices[f[3]]
        if shift:
            p1, p2, p3 = _shift_tri_2d(p1, p2, p3)
        if reduce_dim:
            p1 = np.array(p1[:-1], dtype=np.float64)
            p2 = np.array(p2[:-1], dtype=np.float64)
            p3 = np.array(p3[:-1], dtype=np.float64)
        shape_skew = _calc_skew(p1, p2, p3)
        skews.append(shape_skew)
    return skews

def _shift_tri_2d(p1, p2, p3):
    '''
        triangle in 3d space creates a 3x2 Jacobian which we cant calculate the determinent for
        so first need to transform to 2d space using the following method:
        https://www.gamedev.net/forums/topic/27456-rotating-a-triangle-onto-the-x-y-plane/
    '''
    # initial coordinate
    p3 = p3 - p1
    p2 = p2 - p1
    p1 = p1 - p1

    # calculate the normal of the triangle and note z normal
    t_norm = normalise(cross(p2 - p1, p3 - p1))
    z_norm = [0,0,1]

    # calculate axis to rotate about - normal to triangle norm and z norm
    a_cross = cross(t_norm, z_norm)
    
    if sum(a_cross) == 0:
        # special cases where the points already lie in x,y plane so just return 2d version of them
        # need to specify np.float64 otherwise it bugs out for some reason and thinkgs its data type is dtype('O')
        p1, p2, p3 = [np.array(p, np.float64) for p in [p1, p2, p3]]
        return p1[:-1], p2[:-1], p3[:-1]

    axis = normalise(a_cross)
    
    # calculate angle of rotation
    # note - using sin with cross prod causes errors
    theta = acos(dot(t_norm, z_norm)/(mag(t_norm) * mag(z_norm)))
    
    # rotate the points using https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    p1, p2, p3 = rotate_about_axis([p1, p2, p3], theta, axis)

    # drop to 2d - p1 should be 0 still, and z coord of p2,p3 should be 0. 
    return p1[:-1], p2[:-1], p3[:-1]

def _calc_skew(p1, p2, p3):
    '''
        f_shape function form mesh morphing paper.

        Using the shape quality metric from knupp2003algebraic. Unclear from de2007mesh 
        exactly what metric is used so assuming this one.
    '''
    Ak = np.array([p2 - p1, p3 - p1]).T
    alpha = np.linalg.det(Ak)
    if alpha <= 0:
        #print(f"Warning - degenerate... {p1}{p2}{p3}, checking inverse")
        Ak = np.array([p3 - p1, p2 - p1]).T
        alpha = np.linalg.det(Ak)
        if alpha <= 0:
            #print(f"Still degenerate....")
            return 0.0
        #print("warning - degenerate. 0.0 skew")
        return 0.0

    lam = np.dot(Ak.T, Ak)
    f_shape = (sqrt(3) * alpha) / (lam[0][0] + lam[1][1] - lam[0][1])
    return f_shape

# f_ss (3)
# aka f_size_skew_3
def f_size_shape_3(ff: Mesh, ignore=[1,2,3,4], alpha=1.0, beta=1.0, ff_r=None):
    f_size = f_size_3(ff, ignore, ff_r)
    f_shape = f_skew_3(ff, ignore)
    f_ss = [(f_sz**alpha) * (f_sh**beta) for f_sz, f_sh in zip(f_size, f_shape)]
    return f_ss

# f_size (4)
def f_size_4(ff: Mesh, ignore=[1,2,3,4], ff_r: Mesh = None):
    """
        Calculates relative size for each boundary triangle in ff. 

        If ff_r is provided, the reference area used is the triangels 
        area in the baseline mesh
        If not, the average area of all triangles in ff is used. 
    """

    ff_areas = f_area_4(ff, ignore)
    if ff_r == None:
        ref_areas = [sum(ff_areas)/len(ff_areas)] * len(ff_areas)
    else:
        ref_areas = f_area_4(ff_r, ignore)
    
    taus = [a/b for a,b in zip(ff_areas, ref_areas)]
    f_size = [min([t, 1.0/t]) for t in taus]
    return f_size

def f_area_4(ff : Mesh, ignore=[1,2,3,4]):
    areas = []
    for t in ff.boundary_quads:
        if t[4] in ignore:
            continue
        p1,p2,p3,p4 = ff.nodes[t[0]], ff.nodes[t[1]], ff.nodes[t[2]], ff.nodes[t[3]]
        a1 = _face_area_4(p1,p2,p3,p4)
        areas.append(a1)
    return areas

def _face_area_4(p1, p2, p3, p4):
    """Custom calculation that splits into triangles."""
    vertices = [p1,p2,p3,p4]
    vertices = sort_vertices_ccw(vertices)
    p1,p2,p3,p4 = vertices
    return _calc_area_3(p1, p3, p4) + _calc_area_3(p1, p2, p3)

# f_skew (4)
def f_skew_4(ff: Mesh, ignore=[1,2,3,4], shift=True, reduce_dim=False):
    skews = [] # called shape in the og paper...
    gl_to_lc, vertices, faces_3, faces_4 = ff.get_multiple_surface_vertices_and_faces(ignore=ignore)
    for f in faces_4:
        p1, p2, p3, p4 = vertices[f[1]], vertices[f[2]], vertices[f[3]], vertices[f[4]]
        if shift:
            p1, p2, p3, p4 = _shift_quad_2d(p1, p2, p3, p4)
        if reduce_dim:
            p1 = np.array(p1[:-1], dtype=np.float64)
            p2 = np.array(p2[:-1], dtype=np.float64)
            p3 = np.array(p3[:-1], dtype=np.float64)
            p4 = np.array(p4[:-1], dtype=np.float64)
        skew = _calc_skew_4(p1, p2, p3, p4)
        skews.append(skew)
    return skews

def _shift_quad_2d(p1, p2, p3, p4):
    """Written by ChatGpt - however result is identical to triangle code above"""
    vertices = np.array([p1, p2, p3, p4])
    # Calculate the normal vector of the plane containing the quadrilateral
    normal = np.cross(vertices[1] - vertices[0], vertices[3] - vertices[0])
    normal /= np.linalg.norm(normal)

    # Calculate the rotation matrix that maps the normal vector to the z-axis
    v = np.array([0, 0, 1])
    rot_axis = np.cross(normal, v)
    if np.allclose(rot_axis, 0):
        # Handle the case where normal == v
        R = np.identity(3)
    else:
        rot_angle = np.arccos(np.dot(normal, v))
        rot_axis /= np.linalg.norm(rot_axis)
        skew_symmetric = np.array([[0, -rot_axis[2], rot_axis[1]],
                                [rot_axis[2], 0, -rot_axis[0]],
                                [-rot_axis[1], rot_axis[0], 0]])
        R = np.identity(3) + np.sin(rot_angle) * skew_symmetric + (1 - np.cos(rot_angle)) * np.dot(skew_symmetric, skew_symmetric)

    # Apply the rotation to the quadrilateral
    vertices_rotated = np.dot(R, vertices.T).T
    return vertices_rotated

def _calc_skew_4(p1, p2, p3, p4):
    '''
        f_skew function from knupp2003algebraic.
    '''
    Aks, lambdaKs = lam_aks(p1, p2, p3, p4)
    return 4.0 / sum([sqrt(lambdaKs[k][0][0] * lambdaKs[k][1][1])/np.linalg.det(Aks[k]) for k in range(4)])

def lam_aks(p1, p2, p3, p4):
    """assumes px are 3x1"""
    p1, p2, p3, p4 = p1[:-1], p2[:-1], p3[:-1], p4[:-1]
    ps = [p1, p2, p3, p4]
    Aks = [np.array([ps[(k+1)%4] - ps[k%4], ps[(k+3)%4] - ps[k%4]]).T for k in range(4)]
    lambdaKs = [np.dot(Ak.T, Ak) for Ak in Aks]
    return Aks, lambdaKs

# f_shape (4)
def f_shape_4(ff : Mesh, ignore=[1,2,3,4], shift=True, reduce_dim=False):
    shapes = [] # called shape in the og paper...
    gl_to_lc, vertices, faces_3, faces_4 = ff.get_multiple_surface_vertices_and_faces(ignore=ignore)
    for f in faces_4:
        p1, p2, p3, p4 = vertices[f[1]], vertices[f[2]], vertices[f[3]], vertices[f[4]]
        if shift:
            p1, p2, p3, p4 = _shift_quad_2d(p1, p2, p3, p4)
        if reduce_dim:
            p1 = np.array(p1[:-1], dtype=np.float64)
            p2 = np.array(p2[:-1], dtype=np.float64)
            p3 = np.array(p3[:-1], dtype=np.float64)
            p4 = np.array(p4[:-1], dtype=np.float64)
        shape = _calc_shape_4(p1, p2, p3, p4)
        shapes.append(shape)
    return shapes

def _calc_shape_4(p1, p2, p3, p4):
    """f_shape from knupp2003"""
    Aks, lambdaKs = lam_aks(p1, p2, p3, p4)
    return 8.0 / sum([(lambdaKs[k][0][0] + lambdaKs[k][1][1])/np.linalg.det(Aks[k]) for k in range(4)])

# f_ss (4)
def f_size_skew_4(ff: Mesh, ignore=[1,2,3,4], alpha=1.0, beta=1.0, ff_r=None):
    f_size = f_size_4(ff, ignore, ff_r)
    f_shape = f_skew_4(ff, ignore)
    f_ss = [(f_sz**alpha) * (f_sh**beta) for f_sz, f_sh in zip(f_size, f_shape)]
    return f_ss

def f_size_shape_4(ff: Mesh, ignore=[1,2,3,4], alpha=1.0, beta=1.0, ff_r=None):
    f_size = f_size_4(ff, ignore, ff_r)
    f_shape = f_shape_4(ff, ignore)
    f_ss = [(f_sz**alpha) * (f_sh**beta) for f_sz, f_sh in zip(f_size, f_shape)]
    return f_ss
