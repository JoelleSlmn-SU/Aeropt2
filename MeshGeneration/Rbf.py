from math import sqrt
import os, sys

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname('Utilities'))
from Utilities.Math import *

def constructA(N, S, bf="multiquadric", c=1.0, polynomial=False):
    # Construct A -  interpolation matrix
    if polynomial:
        A = np.zeros((N+4,N+4)) # ensures 0 in bottom right corner
    else:
        A = np.zeros((N,N)) 
    # M
    for i in range(0, N):
        for j in range(0, N):
            r = euc_norm(S[i], S[j])
            A[i][j] = bf(r, c)
    if polynomial:
        # P and Pt
        for i in range(0,N):
            A[i][N] = 1.0
            A[N][i] = 1.0
        for i in range(0,N):
            for j in range(N+1,N+4):
                A[i][j] = S[i][j-N-1]
                A[j][i] = S[i][j-N-1]
    try:
        cond = np.linalg.cond(A) # get condition number for debugging/metrics
    except:
        print("error calculating condition number... which i dont think is a good thing.")
    print(f"({N} nodes)condition number of A = {cond}")
    return A, cond

def train(S, f, bf="multiquadric", c=1.0, plot=False, polynomial=False):
    """
        Solves Ax=G
            A = ((M,P),(Pt,0))
            x = (lambda,beta)
            G = (f,0)
        Arguments
        ---------
        S  = sources (NxM) matrix, N = number of sources, M = dimension of source vectors.\n
        f  = known interpoland results. Indexes should match with S.\n
        bf = Basis function to use. Function should accept 2 floats, r and c. r = distance, c = shape.\n
        c  = shape parameter.
        
        Returns
        -------
        lambda, betas, cond
    """
    N = len(S)
    try:
        assert len(f) == N
    except AssertionError as aex:
        print(f"N = {N}")
        print(f"lenf = {len(f)}")
        raise
    A, cond = constructA(N, S, bf, c, polynomial)

    # Construct G
    if polynomial:
        G = np.zeros((N+4,1))
    else:
        G = np.zeros((N,1))
    for i, fi in enumerate(f):
        G[i] = fi
    
    # solve to find x
    x = None
    try:    
        x = np.linalg.solve(A, G)
    except np.linalg.LinAlgError as laex:
        print("Singular matrix using np.linalg.solve. Trying np.linalg.lstsq instead")
        x = np.linalg.lstsq(A, G, rcond=None)[0]
    except Exception as ex:
        print("Unexpected error with linalg sovle")
        print(ex)
        raise
    
    # extract lambdas and betas.
    if polynomial:
        lambdas = x[:N].T[0] # Should be length N. .T[0] avoids weird reshaping issue later on.
        betas   = x[N:].T[0] # should be length 4. .T[0] avoids weird reshaping issue later on.
    else:
        lambdas = x.T[0]
        betas   = [0.0] * 4
    return lambdas, betas, cond
    
def predict(L, B, S, T, bf="multiquadric", c=1.0):
    """
        Calculates all RBF predictions for target nodes. 
        Arguments
        ---------
        L  = lambdas. these are the weights at each center, S.\n
        B  = betas. these are the polynomial coefficients.\n
        S  = sources (NxM) matrix, N = number of sources, M = dimension of source vectors.\n
        T  = Target nodes to predict. Note sources should be in this list too and should map to f from training data.\n
        bf = Basis function to use. Function should accept 2 floats, r and c. r = distance, c = shape.\n
        c  = shape parameter.
        
        Returns
        -------
        vals = predictions for target nodes. 
    """
    N = len(S)
    assert len(L) == N
    assert len(S) == N
    assert len(B) == 4
    
    def calc_val(t):
        val = 0.0

        for j in range(N):
            r = euc_norm(t, S[j])
            val += L[j]*bf(r, c)
        
        #val += B[0] 
        #val += B[1] * t[0]
        #val += B[2] * t[1]
        #val += B[3] * t[2]
        return val

    vals = [calc_val(t) for t in T]

    return vals

def train_3d(S, F, bfs, cs):
    """
        Uses this modules train function to operate over a 3d set of prediction values.
        Makes the function slightly easier to use and apply. 
    """
    N = len(S)
    assert len(S) == N
    assert len(F) == N
    F = np.array(F).T.tolist()
    d = len(F)
    assert len(F)   == d
    assert len(bfs) == d
    assert len(cs)  == d

    weights = np.zeros([d, N])
    betas   = np.zeros([d, 4])
    conds   = np.zeros([d, 1])
    print(f"Training RBF network. Num sources = {N}, dimensions = {d}")
    
    for i in range(d):
        f  = F[i]
        c  = cs[i]
        bf = bfs[i]
        if not all(v == 0 for v in f):
            weights[i], betas[i], conds[i] = train(S, f, bf, c)
    return weights, betas, conds

def predict_3d(L, B, S, T, bfs, c):
    """
        Uses this modules predict function to operate over a 3d set of prediction values.
        Makes the function slightly easier to use and apply. 
    """
    N = len(S)
    assert len(S)    == N
    assert len(L[0]) == N
    d = len(L)
    assert len(L)    == d
    assert len(S[0]) == d
    assert len(B)    == d
    assert len(B[0]) == 4 # cant currently andle different polynomials.
    t = len(T) 
    assert len(T) == t
    assert len(T[0]) == d

    print(f"Predicting from RBF network. Num sources = {len(S)} Num targets = {len(T)}")
    ds = np.zeros([d, t])
    for i in range(d):
        ds[i] = predict(L[i], B[i], S, T, bf=bfs[i], c=c[i])
    return ds.T.tolist()

def rbf_hyperparameter_optimise_condition(sources, known_values, clims=[0.01, 10], cnum=100, inc_local=False, inc_old=False):
    """
        Hyperparameter Optimisation for RBF interpolation of a dataset. This simply minimises the condition number for A. It can be 
        configured to include locally supported basis functions or just purely global. Following a "convergence" study, it was shown that 
        minimising the condition number to ensure a stable matrix is sufficient to pick a good basis function/shape parameter combo.

        Essentially, the condition number of A affects how much the f in Ax=f will change based on x changing. In this instance, it 
        determines how "wild" the interpolation of new points will be that were not in the training dataset.
        See "Matrix Condition Number" on wikipedia for full details.

        This method avoids the O(N^3) and O(M) training/prediction calculations from the full method.
        
        Attributes
        ----------
            sources      : list (Nxd) - 
            known_values : list (Nx1) - 
            clims        : list (2x1) - limits for the searchable range of c values
            cnum         : int        - number of c values to consider
            inc_local    : bool       - true if locally supported functions should be considered as viable options for optimisation.
            inc_local    : bool       - true if old implementations of globally supported functions should be considered as viable 
                                        options for optimisation.
        Returns
        ----------
            best_b         : func  - best basis function
            best_c         : float - best c value 
            best_cond      : float - lowest condition number for the best b and c
            best_eval_func : func  - best evaluation function (eg rbf.predict)
    """
    sources      = np.array(sources)
    known_values = np.array(known_values)
    
    print("Performing RBF hyperparameter optimisation based on condition number")
    locally_supported_bfs      = ["wendland_c_0", "wendland_c_2"]
    globally_supported_bfs     = ["multiquadric", "inverse_multiquadric", "inverse_quadric", "gaussian"]
    globally_supported_bfs_new = ["multiquadric_new", "inverse_multiquadric_new", "gaussian_new"]
    bfs = globally_supported_bfs_new
    if inc_old:
        bfs += globally_supported_bfs
    if inc_local:
        bfs += locally_supported_bfs
    cs   = [x for x in np.arange(start=clims[0], stop=clims[1], step=(clims[1]-clims[0])/cnum)]
    #cs = [0.1, 1.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    #r = radius(sources, known_values)
    #log.debg(f"r from radius functions is: {r}")
    #if r != 0.0:
    #    cs.append(r) 
    
    bnum = len(bfs)
    cnum = len(cs)
    total_num = bnum * cnum

    print(f"num_tests = {total_num}")
    best_b    = bfs[0]
    best_c    = 1.0
    best_cond  = 1000000000000000000000.0
    for bf in bfs:
        for c in cs:
            A, cond = constructA(len(sources), sources, bf, c, polynomial=False)
            if cond < best_cond and cond > 1000000000000:
                best_b = bf
                best_c = c
                best_cond = cond

    weights, betas, cond = train(sources, known_values, bf=best_b, c=best_c)
    print(f"Best condtition number was calculated as {best_cond} but training condition number is {cond}.")
    best_eval_func       = lambda x: predict(weights, betas, sources, [x], bf=best_b, c=best_c)
    return best_b, best_c, best_cond, best_eval_func

def rbf_hyperparameter_optimise_full(sources, known_values, clims=[0.01, 100], cnum=100, rmse_gt=True):
    """
        Hyperparameter Optimisation for RBF interpolation of a dataset. This is a brute force hyperparameter optimisation that 
        uses the classic leave-one-out method of minimising RMSE and maximising spearmans rank.

        This is very slow for large datasets and it is not recommended to be used. It is slow because of the O(N^3) matrix
        inversion and O(M) training calculation. These are both very lengthy.
        
        Attributes
        ----------
            sources : list - (Nxd)
            known_values      : list - (Nx1)
            clims         : list - (2x1)
            cnum          : int  -
            rmse_gt       : bool - true if rmse must be greater than 0.0. set to true to prevent overfitting 
        Returns
        ----------
            best_b         : func  - best basis function
            best_c         : flaot - best c value 
            best_k         : int   - optimal number of "leave one out" points used (1)
            min_rmse       : float - minimum rmse
            best_eval_func : func  - best evaluation function (eg rbf.predict)
            max_spear      : float - 
    """
    sources      = np.array(sources)
    known_values = np.array(known_values)
    print("Performing RBF hyperparameter optimisation")
    bfs = ["multiquadric", "multiquadric_new", "inverse_multiquadric", "inverse_multiquadric_new", "inverse_quadric", "gaussian", "gaussian_new", "wendland_c_0", "wendland_c_2"]
    #bfs = [wendland_c_0, wendland_c_2]
    #bfs = [multiquadric_new, multiquadric]
    
    cs   = [x for x in np.arange(start=clims[0], stop=clims[1], step=(clims[1]-clims[0])/cnum)]
    cs = [0.1, 1.0, 5.0, 10.0]#, 30.0, 50.0, 100.0]
    #r = radius(sources, known_values)
    #log.debg(f"r from radius functions is: {r}")
    #if r != 0.0:
    #    cs.append(r) 
    
    # TODO - implement bootstrapping
    ks = [1]
    
    bnum = len(bfs)
    cnum = len(cs)
    knum = len(ks)
    total_num = bnum * cnum * knum

    print(f"num_tests = {total_num}")
    count     = 0
    best_b    = bfs[0]
    best_c    = 1.0
    best_k    = 1
    min_rmse  = 100.0
    max_spear = 0.0
    max_cond  = 0.0
    min_cond  = 1000000000000000000000.0
    best_cond = 1000000000000000000000.0
    best_eval_func = None
    for bf in bfs:
        for c in cs:
            #log.debg(count)
            # TODO - handle bootstrapping here. For now, just use LOOC
            rmses = []
            spears = []
            conds = []
            for k in range(len(sources)):
                s = np.delete(sources, [k], axis=0)
                kv = np.delete(known_values, [k])
                t = np.array([sources[k]])
                predictions, eval_func, cond = rbf(s, kv, t, bf, c, log=False)
                avg_error_abs, avg_error_pc, sd_error_pc, spear, kendal, rmse = rbf_stats(sources, known_values, eval_func)
                spears.append(spear)
                rmses.append(rmse)
                conds.append(cond)
            spear = sum(spears)/len(spears)
            rmse  = sum(rmses)/len(rmses)
            cond  = sum(conds)/len(conds)
            count += 1
            print(f"(i - {count}): Spearmans Rank={spear}, RMSE={rmse}, BF={bf.__name__}, C={c}, Kappa={cond}")
            if  rmse < min_rmse:#abs(spear) >= abs(max_spear) and
                update = True
                if rmse_gt:
                    if rmse <= 0.0:
                        update = False
                if update:
                    print(f"Updating best fit (i - {count}): Spearmans Rank={spear}, RMSE={rmse}, BF={bf.__name__}, C={c}, Kappa={cond}")
                    best_b    = bf
                    best_c    = c
                    best_k    = k
                    min_rmse  = rmse
                    max_spear = spear
                    best_cond = cond
                    best_eval_func = eval_func
    return best_b, best_c, best_cond, min_rmse, best_eval_func, max_spear

def rbf(sources, known_results, targets, bf, c, log=False):
    weights, betas, cond = train(sources, known_results, bf=bf, c=c)
    eval_func   = lambda x: predict(weights, betas, sources, [x], bf=bf, c=c)
    predictions = np.array([float(eval_func(x)[0]) for x in targets])
    return predictions, eval_func, cond

def rbf_stats(sources, known_results, eval_func):
    predictions = [round(float(eval_func(s)[0]), 3) for s in sources]
    error_abs = [a - p for a, p in zip(known_results, predictions)]
    errors_pc = [(p - a) * 100 * (2.0 / (a + p)) for a, p in zip(known_results, predictions)]
    spear  = spearmanr(known_results, predictions).correlation
    kendal = kendalltau(known_results, predictions).correlation#, alternative='two-sided')
    rmse   = sqrt(sum([x**2 for x in error_abs])/len(error_abs))
    avg_error_abs = sum(error_abs)/len(error_abs)
    avg_error_pc = sum(errors_pc)/len(errors_pc)
    sqsum = 0.0
    mu = avg_error_pc
    for e in errors_pc:
        sqsum += (e - mu)**2
    sd_error_pc = sqrt(sqsum/len(errors_pc))
    return avg_error_abs, avg_error_pc, sd_error_pc, spear, kendal, rmse