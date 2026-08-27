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


# ---------------------------------------------------------------------------
# Constrained BO additions (Gardner et al., 2014 "Bayesian Optimization with
# Inequality Constraints"). Nothing above this line is touched — EI/POI/UCB
# keep their existing (unconstrained) behaviour and signatures.
# ---------------------------------------------------------------------------

def feasibility_probability(mu_c, sigma_c, limit, sense="<="):
    """
    Probability that a black-box constraint is satisfied at each point,
    under a GP posterior N(mu_c, sigma_c^2) on the constraint metric.

    Parameters:
    - mu_c, sigma_c: posterior mean / STANDARD DEVIATION (not variance) of the
      constraint metric at each point, same length arrays.
    - limit: the constraint threshold, in the same units as mu_c.
    - sense: "<=" for P(c(x) <= limit), ">=" for P(c(x) >= limit).

    Returns:
    - array of probabilities in [0, 1], one per input point.
    """
    mu_c = np.asarray(mu_c, dtype=float)
    sigma_c = np.asarray(sigma_c, dtype=float)
    # Guard against a (near-)zero-variance posterior (e.g. exact repeat of a
    # training point) turning into a divide-by-zero / NaN.
    sigma_c = np.maximum(sigma_c, 1e-12)

    if sense == "<=":
        z = (limit - mu_c) / sigma_c
    elif sense == ">=":
        z = (mu_c - limit) / sigma_c
    else:
        raise ValueError(f"sense must be '<=' or '>=', got {sense!r}")

    return norm.cdf(z)


def constrained_EI(f, mu, sigma, constraint_posteriors, objective=-1):
    """
    Feasibility-weighted Expected Improvement.

    acquisition(x) = EI(x) * prod_j P(constraint_j satisfied at x)

    Parameters:
    - f: best FEASIBLE observed value so far (see
      BayesianOptimiser.Y_best_feasible). Using the unconstrained best here
      is a common mistake -- it can bias search away from the constraint
      boundary where the true optimum of a constrained problem often sits.
    - mu, sigma: objective GP posterior mean / std at each point.
    - constraint_posteriors: list of dicts, one per constraint, each with
      keys 'mu', 'sigma' (posterior mean/std of that constraint metric,
      same length as mu), 'limit' (float) and 'sense' ('<=' or '>=').
    - objective: passed through to EI (-1 minimise / 1 maximise).

    Returns:
    - array of acquisition values, one per input point. If `f` is None (no
      feasible point observed yet), returns the feasibility product alone --
      i.e. the acquisition purely rewards moving into the feasible region,
      since "expected improvement" is undefined without a feasible
      incumbent.
    """
    mu = np.asarray(mu, dtype=float)
    feas = np.ones_like(mu)
    for cp in constraint_posteriors:
        feas = feas * feasibility_probability(cp["mu"], cp["sigma"], cp["limit"], cp.get("sense", "<="))

    if f is None:
        return feas

    ei = EI(f, mu, sigma, objective=objective)
    return ei * feas