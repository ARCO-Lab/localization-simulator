import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag

def isotropic(dim, variance):
    return np.identity(dim) * variance

def shiftedPos(pose, isotropic):
    return np.random.multivariate_normal(pose, isotropic)

def gradient_range_localization(x, p, d):
    """
    Compute the Gradient for range-only localization based on A3 for Mechtron 3X03.
    """
    m = len(p)
    gradient = np.zeros(np.shape(x))

    for i in range(m):
        distance = np.linalg.norm(x-p[i])
        gradient += ((distance-d[i])*((x-p[i])/distance))
    
    return 2*gradient


def fim(x, p, d, sol, isotropic, variance, hessian=False):
    m = len(x)
    inf = np.zeros((m, m, *isotropic.shape))
    for i in range(m):
        inf[i, i] = np.linalg.inv(isotropic)
    for j in sol:
        for i in range(m):
            grad = gradient_range_localization(x[i],p,d[i])
            mean = np.linalg.norm(x[i]-p[j])
            log_likelihood = norm.pdf(d[i][j],loc=mean,scale=np.sqrt(variance))
            g_ij = grad*np.log(log_likelihood)
            inf[i][i] += np.outer(g_ij,g_ij)
    
    return sum([log_det(inf[i][i]) for i in range(m)])


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

    x = np.array([(4,4), (5,5), (6,6), (7,7)])
    p = np.array([(3,3), (14,14), (18,12)])
    d = np.array([[1.41,2.31,3.45], [12.1,13.21,24.12], [16.1,15.13,19.4], [12,11.3,17.7]]) 
    iso = isotropic(2,0.5)

    # print(np.shape(x))
    # print(np.shape(p))
    # print(np.shape(d))

    # print(gradient_range_localization(x[0],p,d[0]))

    tt = fim(x,p,d,{1,2},iso,1)

    print(tt)
    # print(tt[0][0])

    # print(np.outer(gradient_range_localization(x[0],p,d[0]),gradient_range_localization(x[0],p,d[0])))


    # hessian = hessian_range_localization(x, p, d)
    # print("Hessian Matrix:")
    # print(hessian)
    # print(np.shape(hessian))
    # print(log_det(hessian))

