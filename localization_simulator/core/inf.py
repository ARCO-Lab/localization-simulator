import numpy as np
from scipy.stats import norm
import pandas as pd
from itertools import combinations

def isotropic(dim, variance):
    return np.identity(dim) * variance

def shiftedPos(pose, isotropic):
    return np.random.multivariate_normal(pose, isotropic)

# def gradient_range_localization(x, p, d, j):
#     """
#     Compute the Gradient for range-only localization based on A3 for Mechtron 3X03.
#     """
#     print(f"P is: {p}")
#     m = len(p)
#     gradient = np.zeros(np.shape(x))

#     # for i in range(m):
#     #     distance = np.linalg.norm(x-p[i])
#     #     print(f"distance is {distance}")
#     #     gradient += ((distance-d[i])*((x-p[i])/distance))

#     distance = np.linalg.norm(x-p[j])
#     gradient += ((distance-d[j])*((x-p[j])/distance))
#     return 2*gradient

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
    gradient = np.zeros((n,n))

    for j in sol:
        distance = np.linalg.norm(x-p[j])
        g = ((distance-d[j])*((x-p[j])/distance))
        gradient += np.outer(g,g)
        
    return 2*gradient


def fim(x, p, d, sol, isotropic, variance, hessian=False):
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
    inf = np.repeat(np.linalg.inv(isotropic)[np.newaxis, :, :], m, axis=0)
    for _ in sol:
        for i in range(m):
            grad = outer_grad(x[i],p,d[i],sol)
            grad = (1/(variance**2))*grad
            inf[i] += grad
    
    return sum([log_det(inf[i]) for i in range(m)])

def hessian_range_localization(x, p, d):
    """
    Compute the Hessian for range-only localization based on A3 for Mechtron 3X03.
    """
    m = len(d)
    n = np.prod(np.shape(p))
    hessian = np.zeros((n,n))

    distance = np.linalg.norm(x - p)
    direction = np.outer(x-p,x-p)

    term1 = direction / distance
    for i in range(m):
        term2 = ((distance - d[i])*direction / (distance ** 3))
        term3 = ((distance - d[i]) / distance) * np.identity(n)
        hessian += (term1 - term2 + term3)

    return 2*hessian

def log_det(m):
    return np.log(np.linalg.det(m))

if __name__ == "__main__":
    # isotropic_matrix = isotropic(2, 1)

    # x = np.array([(4,4), (5,5), (6,6), (7,7)])
    # p = np.array([(3,3), (14,14), (18,12)])
    # d = np.array([[1.41,2.31,3.45], [12.1,13.21,24.12], [16.1,15.13,19.4], [12,11.3,17.7]]) 
    # iso = isotropic(2,0.5)

    x = np.array([(1, 1)])
    p = np.array([(2, 1), (5, 1), (1, 5)])
    iso = isotropic(2, 2)
    variance = 1
    anc = [0,1,2]

    def addNoise(x, p, variance):
        d = np.array([[np.linalg.norm(i - j) for j in p] for i in x])
        noise = np.random.normal(0, np.sqrt(variance), size=np.shape(d))
        return d + noise

    def sanity(anc):
        ancComb = [set(x) for i in range(len(anc) + 1) for x in combinations(anc, i)]
        infTable = {str(comb): [] for comb in ancComb}
        for _ in range(10):
            d = addNoise(x,p,variance)
            for s in ancComb:
                i = fim(x,p,d,s,iso,variance)
                infTable[str(s)].append(i)

        df = pd.DataFrame(infTable)
        df.index += 1
        mean = df.mean()
        quartiles = df.quantile([0.25,0.5,0.75])
        summary = pd.DataFrame({
            'Mean': mean,
            '25th Percentile':quartiles.loc[0.25],
            '50th Percentile':quartiles.loc[0.5],
            '75th Percentile':quartiles.loc[0.75]
        })
        return df, summary

    print(sanity(anc)[0])
    print(sanity(anc)[1])

