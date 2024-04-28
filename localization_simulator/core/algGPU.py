from itertools import combinations
from time import perf_counter
import numpy as np
import cupy as cp


def brute(param):
    start = perf_counter()
    candidates = list(combinations(range(len(param.p)),param.k))

    inf_max = float('-inf')
    solution = set()

    for c in candidates:
        inf_j = fimGPU(param.x,param.p,param.d,c,param.iso_var,param.sensor_var)
        # print(f"Set:{c} and inf_j:{inf_j} ")

        if inf_j > inf_max:
            inf_max = inf_j
            solution = c
    
    stop = perf_counter()
    return [solution,inf_max,0,stop-start]

def log_det(m):
    return cp.log(cp.linalg.det(m))

def outer_grad(x, p, d, sol):
    """Compute the Gradient for range-only localization based on A3 for Mechtron 3X03.

    Args:
        x (numpy.ndarray[float]): x_i (a specific pose)
        p (numpy.ndarray[tuple[float]]): All anchors positions
        d (numpy.ndarray[float]): the distance measurements for x_i
        sol (set{int}): The specific anchor numbers in the candidate set

    Returns:
        (numpy.ndarray[float]): The gradient
    """
    n = len(x)
    gradient = cp.zeros((n,n))

    for j in sol:
        distance = cp.linalg.norm(x-p[j])
        # if distance == 0:
        #     print(f"x:{x} \t p[j]:{p[j]} \t distance:{distance}")
        if cp.isnan(d[j]):
            continue
        g = ((distance-d[j])*((x-p[j])/distance))
        gradient += cp.outer(g,g)
        
    return 2*gradient


def fimGPU(x, p, d, sol, isotropic, variance, hessian=False, trace=False):
    """_summary_

    Args:
        x (numpy.ndarray[float]): x (All poses)
        p (numpy.ndarray[tuple[float]]): All anchors positions
        d (numpy.ndarray[float]): the distance measurements for all x_i (all poses)
        sol (set{int}): The specific anchor numbers in the candidate set
        isotropic (bool): The isotropic covariance matrix
        variance (float): The variance for measurements
        hessian (bool, optional): Flag to use the hessian instead of the gradient. Defaults to False.

    Returns:
        (float): The log determinant of the information matrix
    """
    m = len(x)
    inf = cp.repeat(cp.linalg.inv(isotropic)[cp.newaxis, :, :], m, axis=0)
    emptyInf = sum([log_det(inf[i]) for i in range(m)])
    for _ in sol:
        for i in range(m):
            if hessian:
                pass
            else:
                grad = outer_grad(x[i],p,d[i],sol)
                grad = (1/(variance**2))*grad
                inf[i] += grad
    
    if trace:
        return sum([cp.trace(inf[i]) for i in range(m)])
    else:
        return sum([log_det(inf[i]) for i in range(m)]) - emptyInf
    

if __name__ == "__main__":
    pass