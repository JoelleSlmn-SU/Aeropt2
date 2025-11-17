"""
    kernels
"""

from math import sqrt

import numpy as np

# Base Classes
class Kernel:
    """Base class for all kernels"""

    def __init__(self, lengthscale=1.0, noise_variance:float=1.0) -> None:
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance

    def covariance(self, x_1: float, x_2: float) -> float:
        """Return the covariance between location X1 and X2."""
        raise NotImplementedError()
    
    def __str__(self):
        ret  = f"{self.__class__.__name__}:\n"
        ret += f"  Noise: \n\t{self.noise_variance}\n"
        ret += f"  Lengthscale:"
        for l in self.lengthscale:
            ret += f"\n\t{l}"
        return ret

## Stationary Kernels
class StationaryKernel(Kernel):
    """
        Stationary kernels are dependent only on difference between points.
        Isotropic stationary kernels are dependent only on the euclidean 
        distance between points. this class is isotropic too
    """

    def covariance(self, x_1: float, x_2: float) -> float:
        super().covariance(x_1, x_2)
        raise NotImplementedError

    def r(self, x_1, x_2) -> float:
        """
            Support function to calculate difference between points.
            Includes handling of range of inputs
        """
        if isinstance(x_1, float):
            return self.r_1d(x_1, x_2)
        elif isinstance(x_1, np.float64):
            return self.r_1d(x_1, x_2)
        elif isinstance(x_1, int):
            return self.r_1d(x_1, x_2)
        elif isinstance(x_1, list):
            return self.r_nd(x_1, x_2)
        elif isinstance(x_1, np.ndarray):
            return self.r_nd(x_1, x_2)
        else:
            print(f"Unhandled datatype {x_1} - {str(type(x_1))}")
            return None

    def r_1d(self, x_1 ,x_2):
        """returns straight euclidean distance between points"""
        if self.lengthscale == 0.0:
            raise ValueError("Error - zero lengthscale")
        return sqrt((x_1 - x_2)**2 / (self.lengthscale**2))

    def r_nd(self, x_1, x_2):
        """returns euc distance where each dimension is scaled using length"""
        # check inputs are the same length. exit if not
        if len(x_1) != len(x_2):
            #print(x_1)
            #print(x_2)
            raise ValueError()
        
        # get the length scale. check if ARD or not. also check for zeros
        length_scale = self.lengthscale
        if not hasattr(self.lengthscale, "__len__"):
            length_scale = [self.lengthscale]*len(x_1)
        if any([x == 0.0 for x in length_scale]):
            raise ValueError()
        
        # calculate sqrt of sum of squares normalised by lengthscale
        #print(f"calculating with {length_scale}")
        totsum = 0.0
        for i, l in enumerate(length_scale):
            totsum += ((x_1[i] - x_2[i])**2)/(l**2)
        return sqrt(totsum)

class GammaExponentialKernel(StationaryKernel):
    """ 
        0 < gamma <= 2
        gamma = 1, equivalent to Matern12 and Exponential kernels.
        gamma = 2, equivalent to RBF      and SquaredExponential kernels.

        Defined in:
            rasmussen2006gaussian (p 86, 94)
    """

    def __init__(self, *args, gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def covariance(self, x_1: float, x_2: float) -> float:
        return self.noise_variance * np.exp(-(np.power(self.r(x_1, x_2), self.gamma)))

class RBFKernel(GammaExponentialKernel):
    """
        RBF Kernels are a special case of the gamma class where gamma=2
        this reduces them to the standard gaussian basis function 

        strictly speaking - there is an extra 2 in the denominator of the 
        exponent for the rbf and squared exponential kernel however this 
        is ommited as it is balanced out by the hyperparameters

        Defined in:
            rasmussen2006gaussian (p 14, 94)
            chugh2020towards (p 15)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(gamma=2.0, *args, **kwargs)

class SquaredExponentialKernel(GammaExponentialKernel):
    """
        commonly presented as a psuedonym for the rbf kernel
        special case of the gamma class where gamma=2

        strictly speaking - there is an extra 2 in the denominator of the 
        exponent for the rbf and squared exponential kernel however this 
        is ommited as it is balanced out by the hyperparameters

        Defined in:
            rasmussen2006gaussian (p 14, 83, 94)
            chugh2020towards (p 15)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(gamma=2.0, *args, **kwargs)

class ExponentialKernel(GammaExponentialKernel):
    """
        Exponential Kernels are a special case of the gamma class where gamma=1

        Defined in:
            rasmussen2006gaussian (p 85, 94)
            chugh2020towards (p 15)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(gamma=1.0, *args, **kwargs)

class Mat12Kern(GammaExponentialKernel):
    """
        Matern 12 Kernels are equivalent to the special case 
        of the gamma class where gamma=1

        they are derived from matern class for nu = 1/2

        Defined in:
            rasmussen2006gaussian (p 84, 94)
            chugh2020towards (p 15)
            shahriari2015taking (p 157 - Section III B)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(gamma=1.0, *args, **kwargs)

class Mat32Kern(StationaryKernel):
    """
        Matern 32 Kernels are derived from matern class for nu = 1/2

        Defined in:
            rasmussen2006gaussian (p 84, 94)
            chugh2020towards (p 15)
            shahriari2015taking (p 157 - Section III B)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def covariance(self, x_1: np.array, x_2: np.array) -> float:
        r = self.r(x_1, x_2)
        sigma = self.noise_variance ** 2
        exp   = np.exp(-sqrt(3)*r)
        extra = 1.0 + sqrt(3)*r
        return sigma * exp * extra

class Mat52Kern(StationaryKernel):
    """
        Matern 52 Kernels are derived from matern class for nu = 1/2

        Defined in:
            rasmussen2006gaussian (p 84, 94)
            chugh2020towards (p 15)
            shahriari2015taking (p 157 - Section III B)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def covariance(self, x_1: np.array, x_2: np.array) -> float:
        r     = self.r(x_1, x_2)
        sigma = self.noise_variance ** 2
        exp   = np.exp(-sqrt(5)*r)
        extra = 1.0 + sqrt(5)*r + (5.0/3.0)*r*r
        return sigma * exp * extra


## TODO
class DotProductKernel(Kernel):
    def covariance(self, X1: float, X2: float) -> float:
        return super().covariance(X1, X2)

    def r(self, X1, X2):
        return X1 * X2 # plus sigma

# Stationary Kernels
class ConstantKernel(StationaryKernel):
    def covariance(self, X1: float, X2: float) -> float:
        return (self.noise_variance**2) * self.lengthscale


class RationalQuadraticKernel(StationaryKernel):
    def __init__(self, lengthscale, alpha, noise_variance: float=0.000000001) -> None:
        super().__init__(lengthscale, noise_variance)
        self.alpha = alpha

    def covariance(self, X1: float, X2: float) -> float:
        alpha = self.alpha
        r     = self.r(X1,X2)
        sigma = self.noise_variance ** 2
        extra = (1.0 + r*r/2.0*alpha) ** -alpha
        return sigma * extra

# Non-Stationary Kernels (TODO - finish)
class PeriodicKernel(Kernel):
    def __init__(self, lengthscale: float, noise_variance: float, period: float=2*np.pi) -> None:
        super().__init__(lengthscale, noise_variance)
        self.period = period

    def covariance(self, X1: float, X2: float) -> float:
        ex = np.exp(-2*(np.square(np.sin(np.pi * (np.abs(X1-X2)) / self.period)))/np.square(self.lengthscale))
        result = np.square(self.noise_variance) * ex
        return result

class LinearKernel(Kernel):
    def covariance(self, X1: float, X2: float) -> float:
        result = self.noise_variance * (X1 - self.lengthscale) * (X2 - self.lengthscale)
        return result

class NeuralNetworkKernel(Kernel):
    # Not implemented yet
    pass

class MultiLayerPerceptron(NeuralNetworkKernel):
    # exactly the same as NeuralNetworkKernel so just subclass it and move on.
    pass


# Compound Kernel Functions
class CompoundKernel():
    def __init__(self, kernels: list) -> None:
        self.kernels = kernels

class SumKernel(CompoundKernel):
    def covariance(self, X1: float, X2: float) -> float:
        result = sum([k.covariance(X1, X2) for k in self.kernels])
        return result

class ProductKernel(CompoundKernel):
    def covariance(self, X1: float, X2: float) -> float:
        result = np.prod([k.covariance(X1, X2) for k in self.kernels])
        return result
