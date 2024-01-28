import numpy as np

def isotropic(dim, variance):
    return np.identity(dim) * variance

def shiftedPos(pose, isotropic):
    return np.random.multivariate_normal(pose, isotropic)

import numpy as np

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

    x = np.array([(4,4)])
    p = np.array([(3,3), (14,14), (18,12), (13,13)]) 
    d = np.array([1.5, 14.1, 16.1, 12.4]) 

    hessian = hessian_range_localization(x, p, d)
    print("Hessian Matrix:")
    print(hessian)
    print(np.shape(hessian))
    print(log_det(hessian))