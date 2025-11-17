import numpy as np
from scipy.stats import norm

def EI(f, mu, sigma, objective=-1):
    """
    Calculate the expected improvement acquisition function.

    Parameters:
    - f: best observed value found so far.
    - mu: Mean value at each point x
    - sigma: Standard deviation at each point x
    - objective - [-1,1] -1 for minimisation, 1 for maximisation

    Returns:
    - The expected_improvement values at the given input points.
    """
    
    #print(f"best = {f}")
    improvement = []
    for m,s in zip(mu, sigma):
        #print(type(mu))
        #print(type(sigma))
        z = objective*(m-f)/s
        #print(f"y (mu): {m}")
        #print(f"s (s): {s}")
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        #print(f"PHI, (cdf): {cdf}")
        #print(f"phi, (pdf): {pdf}")
        a = s*((z*cdf) + (pdf))
        #print(f"ei: {a}")
        #input("pause")

        improvement.append(a)
    return np.array(improvement)

def POI(f, mu, sigma, objective=-1):
    
    """
    Calculate the probability_of_improvement acquisition function.

    Parameters:
    - f: best observed value found so far.
    - mu: Mean value at each point x
    - sigma: Standard deviation at each point x
    - objective - [-1,1] -1 for minimisation, 1 for maximisation
    Returns:
    - The probability_of_improvement values at the given input points.
    """
    improvement = []
    for m,s in zip(mu, sigma):
        z = objective*(m-f)/s
        cdf = norm.cdf(z)
        a = cdf
        improvement.append(a)
    return np.array(improvement)

def UCB(f, mu, sigma, objective=-1):
    
    """
    Calculate the upper_confidence_bound acquisition function.

    Parameters:
    - f: best observed value found so far.
    - mu: Mean value at each point x
    - sigma: Standard deviation at each point x
    - objective - [-1,1] -1 for minimisation, 1 for maximisation
    Returns:
    - The upper_confidence_bound values at the given input points.
    """
    improvement = []
    for m,s in zip(mu, sigma):
        a = m + s
        improvement.append(a)
    return np.array(improvement)
