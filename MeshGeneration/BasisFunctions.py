import math


def int_to_func(en):
    en = int(en)
    if en == 1:
        return multiquadric
    elif en == 2:
        return gaussian
    elif en == 3:
        return inverse_multiquadric
    elif en == 4:
        return wendland_c_0
    elif en == 5:
        return wendland_c_2
    elif en == 6:
        return multiquadric_new
    elif en == 7:
        return gaussian_new
    elif en == 8:
        return inverse_multiquadric_new
    else:
        return multiquadric

def str_to_func(en):
    en = str(en)
    if en == "multiquadric":
        return multiquadric
    elif en == "gaussian":
        return gaussian
    elif en == "inverse_multiquadric":
        return inverse_multiquadric
    elif en == "wendland_c_2":
        return wendland_c_2
    elif en == "wendland_c_4":
        return wendland_c_4
    elif en == "wendland_c_0":
        return wendland_c_0
    elif en == "multiquadric_new":
        return multiquadric_new
    elif en == "gaussian_new":
        return gaussian_new
    elif en == "inverse_multiquadric_new":
        return inverse_multiquadric_new
    else:
        return multiquadric

def get_bf(en):
    if type(en) == int:
        return int_to_func(en)
    elif type(en) == str:
        return str_to_func(en)
    else:
        return multiquadric

# Spline type seems dodge. never really got it to work 
# i think
def spline_type(r, c=1.0, n=1):
    if n%2 == 1: # odd
        return r**n
    elif r == 0.0:
        return r
    else: # even
        return (r**n)*math.log(r)

## Globally Supported Functions
# These are the original implementations.
# they are mostly "old type" functions
def multiquadric(r, c=1.0):
    return math.sqrt((r**2)+(c**2))

def gaussian(r, c=1.0):
    if c == 0.0:
        return 1.0
    ret = math.exp(-((r**2)/(c**2)))
    return ret

def inverse_multiquadric(r, c=1.0):
    try:
        return 1.0/(math.sqrt((r**2)+(c**2)))
    except ZeroDivisionError:
        print(f"Zero Divison Error: r = {r},  c = {c}")
        raise

def inverse_quadric(r, c=1.0):
    return 1.0/((r**2)+(c**2))

# These are the "new type" implementations. slight
#  modifications of the OG ones.
def multiquadric_new(r, c=1.0):
    return math.sqrt(((c*r)**2)+(1))

def gaussian_new(r, c=1.0):
    return math.exp(-((r**2)*(c**2)))

def inverse_multiquadric_new(r, c=1.0):
    return 1.0/(math.sqrt(((c*r)**2)+(1)))

## Locally Supported Functions (Compact)

def wendland_c_0(r, c=1.0):
    if float(c) == 0.0:
        return 0.0
    R = c
    eta = r/R
    if r >= 0 and r <= R:
        return ((1 - eta)**2) 
    else:
        return 0.0

def wendland_c_2(r, c=1.0):
    if float(c) == 0.0:
        return 0.0
    R = c
    eta = r/R
    if r >= 0 and r <= R:
        return ((1 - eta)**4) * ((4 * eta) + 1)
    return 0.0

def wendland_c_4(r, c=1.0):
    if float(c) == 0.0:
        return 0.0
    R = c
    eta = r/R
    if r >= 0 and r <= R:
        return ((1 - eta)**6) * ((35 * (eta**2)) + (18 * eta) + 3)
    return 0.0


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import numpy as np

    xs = [x for x in np.arange(-1.0, 1.0, (2.0/1000.0))]
    cols = ['b', 'y', 'g', 'r', 'purple']
    
    # gaussian
    plt.figure()
    plt.title("gaussian")
    eps = [0.1, 0.5, 1.0, 2.0, 10.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [gaussian(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)

    # gaussian_new
    plt.figure()
    plt.title("gaussian_new")
    eps = [0.1, 0.5, 1.0, 2.0, 10.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [gaussian_new(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)

    # multiquadric
    plt.figure()
    plt.title("multiquadric")
    eps = [0.1, 0.5, 1.0, 2.0, 3.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [multiquadric(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)

    # multiquadric new
    plt.figure()
    plt.title("multiquadric_new")
    eps = [0.1, 0.5, 1.0, 2.0, 3.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [multiquadric_new(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)

    # inverse_multiquadric
    plt.figure()
    plt.title("inverse_multiquadric")
    eps = [0.1, 0.5, 1.0, 2.0, 3.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [inverse_multiquadric(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)
    
    # inverse_multiquadric_new
    plt.figure()
    plt.title("inverse_multiquadric_new")
    eps = [0.1, 0.5, 1.0, 2.0, 3.0]
    for i, ep in enumerate(eps):
        plt.scatter(xs, [inverse_multiquadric_new(x, ep) for x in xs], c=cols[i], label=f"{eps[i]}")
    plt.legend(loc=1)

    plt.show()
