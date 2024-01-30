import numpy as np
from scipy.stats import norm

def isotropic(dim, variance):
    return np.identity(dim) * variance

def shiftedPos(pose, isotropic):
    return np.random.multivariate_normal(pose, isotropic)

def gradient_range_localization(x, p, d):
    """
    Compute the Gradient for range-only localization based on A3 for Mechtron 3X03.
    """
    m = len(p) if len(p)==len(d) else Exception("Same dimensionality and length between distance measurements and anchors")
    gradient = np.zeros(np.shape(x))

    for i in range(m):
        distance = np.linalg.norm(x-p[i])
        gradient += ((distance-d[i])*((x-p[i])/distance))
    
    return 2*gradient

def jointProbDist(pose, anchorPos, measurement,variance):
    jpd = 1.0
    for i in range(len(anchorPos)):
        mean = np.linalg.norm(pose-anchorPos[i])
        jpd *= norm.pdf(measurement[i], loc=mean, scale=np.sqrt(variance))
    return jpd

def fim(x, p, d, variance):
    gradient = gradient_range_localization(x,p,d)
    log_jpd = np.log(jointProbDist(x,p,d,variance))
    return np.outer(gradient*log_jpd,gradient*log_jpd)

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
    p = np.array([(3,3), (14,14), (18,12)]) 
    d = np.array([1.41, 12.1, 16.1]) 

    # print(gradient_range_localization(x,p,d))
    fish = fim(x,p,d,1)
    print(np.linalg.det(fish))
    print(log_det(fish))

    # hessian = hessian_range_localization(x, p, d)
    # print("Hessian Matrix:")
    # print(hessian)
    # print(np.shape(hessian))
    # print(log_det(hessian))
