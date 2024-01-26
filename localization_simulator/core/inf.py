import numpy as np

def isotropic(dim, variance):
    return np.identity(dim) * variance

def newAnchorPos(locations, isotropic):
    return [np.random.multivariate_normal(i, isotropic) for i in locations]

if __name__ == "__main__":
    isotropic_matrix = isotropic(2, 1)
    for i in [newAnchorPos([(3, 3), (14, 14), (18, 12)], isotropic_matrix)]:
        print(i)
