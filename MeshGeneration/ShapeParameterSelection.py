import math
import numpy as np
import os, sys

sys.path.append(os.path.dirname("MeshGeneration"))
sys.path.append(os.path.dirname("Utilities"))
#from mfirst.Utilities.Math import euc_norm
from MeshGeneration.mec import get_mec
from Utilities.PointClouds import dist_between_points
from MeshGeneration.BasisFunctions import multiquadric

def int_to_func(en):
    en = int(en)
    if en == 1:
        return one
    elif en == 2:
        return radius
    elif en == 3:
        return hardy
    elif en == 4:
        return franke
    elif en == 5:
        return rippa_optimal_c
    else:
        return one

def str_to_func(en):
    en = str(en)
    if en == "one":
        return one
    elif en == "radius":
        return radius
    elif en == "hardy":
        return hardy
    elif en == "franke":
        return franke
    elif en == "rippa_optimal_c":
        return rippa_optimal_c
    else:
        return one

def get_shape(en):
    if type(en) == int:
        return int_to_func(en)
    elif type(en) == str:
        return str_to_func(en)

def radius(centers=[], known_values=[], bf=multiquadric):
    return 1.5 * max(abs(min(known_values)), abs(max(known_values)))

def one(centers=[], known_values=[], bf=multiquadric):
    return 1.0

def hardy(centers=[], known_values=[], bf=multiquadric):
    # need to add reference to paper here
    def dist_to_nearest_neighbour(p_id, points):
        distances = []
        for i in range(0, len(points)):
            if i != p_id:
                distances.append(np.linalg.norm(points[p_id] - points[i]))
        return min(distances)
    d = 1.0
    di = []
    N = len(centers)
    for i in range(0, N):
        di.append(dist_to_nearest_neighbour(i, centers))
    d = sum(di)/float(N)
    c = 0.815*d
    return c

def franke(centers=[], known_values=[], bf=multiquadric):
    N = len(centers)
    d = 1.0
    xy_points = []
    xz_points = []
    yz_points = []
    for p in centers:
        xy_points.append([p[0], p[1]])
        xz_points.append([p[0], p[2]])
        yz_points.append([p[1], p[2]])
    _, _, r_xy = get_mec(xy_points)
    _, _, r_xz = get_mec(xy_points)
    _, _, r_yz = get_mec(xy_points)
    r = max([r_xy, r_xz, r_yz])
    D = 2.0*r 
    d = D/math.sqrt(N)
    c = 1.25*d
    return c

def rippa_optimal_c(centers=[], known_values=[], bf=multiquadric):
    def get_l1_norm_cost_of_c(P, f, bf=multiquadric, c=1.0):
        N = len(P)

        A  = np.empty((N,N))
        E  = np.empty((N,1))
        for i in range(0, N):
            for j in range(0, N):
                r = dist_between_points(P[i], P[j])
                A[i][j] = bf(r, c)
        
        a = np.linalg.solve(A, f)
        e = np.identity(N)

        for k in range(0,N):
            ak = a[k]
            ek = e[ : , k] # column matrix 
            xk = np.linalg.solve(A, ek)
            Ek = ak/xk[k]
            E[k] = Ek
        cost = np.linalg.norm(E, 1)
        return cost/float(N)

    cs = np.arange( 0.01, 1.5, 0.01) # maybe need to calculate these based on the centers? or use a smarter algorithm?
    costs = []
    one_pc = (1.5-0.01)/100.0
    pc = 0.0
    for c in cs:
        cost = get_l1_norm_cost_of_c(centers, known_values, basis_function=bf, c=c)
        costs.append(cost)
        if pc*one_pc < c/(1.5-0.01):
            pc += 1.0
            print("Currently c = {:.3f} and at {}%".format(c, pc), end='\r')
    optimal_c = cs[costs.index(min(costs))]
    return optimal_c