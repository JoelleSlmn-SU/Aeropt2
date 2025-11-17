import math
import numpy as np

def euc_norm(p1, p2):
    if type(p1) == np.float64:
        return abs(p2 - p1)
    n = len(p1)
    tot = 0.0
    for i in range(n):
        tot += (p2[i] - p1[i])**2
    return math.sqrt(tot)