"""
    custom gaussian process implementation
"""
from typing import Tuple
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

from Optimisation.BayesianOptimisation.kernels import Kernel
from Utilities.Matrices import inv

counter = 0

class GP:
    """GP class."""
    def __init__(self, kernel: Kernel, noise_variance: float, X=None, Y=None) -> None:
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X = X
        self.Y = Y

    def k(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Covariance matrix between locations X1 and X2. X1 and X2 are 1D arrays."""
        result = np.zeros(shape=(X1.shape[0], X2.shape[0]))
        for row_id, x1 in enumerate(X1):
            for col_id, x2 in enumerate(X2):
                result[row_id, col_id] = self.kernel.covariance(x1, x2)
        return result

    def prior(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        mean  = np.zeros(len(x))
        covar = self.k(x, x)
        return mean, covar

    def posterior(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Return the posterior mean and covariance,
            with the weights used in the posterior mean for plotting.
            
            x = unknown
            X = known
            Y = vals at known
            Compute k(X, X).
        """

        if len(self.X) == 0:
            return self.prior(x)
        
        kXX = self.k(self.X, self.X)

        # inv is (k(X, X) + Σ)⁻¹.
        kXXi = inv(kXX + (np.eye(len(self.X)) * np.square(self.noise_variance)))

        # Compute k(x, X)
        kxX = self.k(x, self.X)

        # Compute k(x, x).
        kxx = self.k(x, x)

        # Compute posterior mean: k(x, X) * (k(X, X) + Σ)⁻¹ * Y
        mean = np.matmul(np.matmul(kxX, kXXi), self.Y)

        # Compute posterior covariance: k(x, x) - k(x, X) * (k(X, X) + Σ)⁻¹ * k(x, X)ᵀ
        covariance = kxx - np.matmul(np.matmul(kxX, kXXi), kxX.T)

        return mean, covariance

    def posterior_predictor(self):
        """returns funciton predictor"""

        def func(x):
            if len(x.shape) == 1:
                x = np.array([x])
            
            #m, c, _ = self.posterior(x)
            m, c = self.posterior(x)
            
            return m, np.diag(c)
        
        return func

    def log_likelihood(self):
        n = len(self.Y)
        Ky = self.k(self.X,self.X) + ((np.eye(n) * (self.noise_variance**2)) + 1e-8)
        data_fit     = -0.5 * np.matmul(self.Y.T, np.matmul(inv(Ky), self.Y))#[0][0]
        norm_const   = -0.5*n*np.log(2*np.pi)
        complex_term = -0.5*np.log(np.linalg.det(Ky))
        log_marginal =  data_fit + norm_const + complex_term
        return log_marginal
    
    def minimise_log_likelihood(self, maxiter=10000):
        
        n = len(self.X[0]) + 1

        def eval_func(p):
            self.kernel.noise_variance = p[0]
            self.kernel.lengthscale = p[1:]
            log_marginal = -self.log_likelihood()
            #print(f"Kernenl: {self.kernel} = {log_marginal}")
            return log_marginal
        
        init_guess = [1.0]*n
        b = Bounds([1e-5]*n, [1e8]*n)
        res = minimize(eval_func, init_guess, method="Nelder-Mead", bounds=b, options={"maxiter":maxiter, "adaptive":False})
        print(self.kernel)
        xopt = res["x"]
        print(f"minimum log marginal: {res['fun']}")
        print(f"Noise: {xopt[0]}")
        for i in range(1, len(xopt)):
            print(f"Len {i}: {xopt[i]}")
        print(f"noise var squared - {self.kernel.noise_variance**2}")

