import numpy as np
from scipy import linalg

def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = linalg.lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")

def force_F_ordered(A):
    """
    return a F ordered version of A, assuming A is triangular
    """
    if A.flags['F_CONTIGUOUS']:
        return A
    print("why are your arrays not F order?")
    return np.asfortranarray(A)

def dtrtri(L):
    """
    Inverts a Cholesky lower triangular matrix

    :param L: lower triangular matrix
    :rtype: inverse of L

    """

    L = force_F_ordered(L)
    return linalg.lapack.dtrtri(L, lower=1)[0]

def dpotri(A, lower=1):
    """
    Wrapper for lapack dpotri function

    DPOTRI - compute the inverse of a real symmetric positive
    definite matrix A using the Cholesky factorization A =
    U**T*U or A = L*L**T computed by DPOTRF

    :param A: Matrix A
    :param lower: is matrix lower (true) or upper (false)
    :returns: A inverse

    """

    A = force_F_ordered(A)
    R, info = linalg.lapack.dpotri(A, lower=lower) #needs to be zero here, seems to be a scipy bug

    symmetrify(R)
    return R, info

def symmetrify(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]

def inv(A):
    L = jitchol(A)
    Ai, _ = dpotri(L, lower=1)
    symmetrify(Ai)
    return Ai
