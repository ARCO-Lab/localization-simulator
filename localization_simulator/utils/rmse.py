"""
A module for using Newtons method to approximate position at a given pose using the norm of the gradient.

Todo:
    * Potentially look at adding a dampening factor in case of instability or excessive oscillation.
    * Do more to decouple nls from maps.
"""

import autograd.numpy as np
from autograd import grad, hessian
import copy
from utils.nls import NLS
    
def rmse(pose, guess, distances, anchors, variance, isotropic, tolerance):
    nls = NLS(0,0,anchors,tolerance,0,0)
    counter = 0

    guess = np.array(guess)
    Real = copy.deepcopy(guess)

    pose = pose.astype(float)
    guess = guess.astype(float)
    guess = np.array(guess)

    while True:
        # print(Real)
        guess, grad, _ = nls.estimatePose(guess, distances, pose, variance, isotropic)
        # print(f"Aprrox {counter}:{guess}")
        counter += 1
        if grad <= tolerance:
            break
    return np.linalg.norm(guess-Real)**2

    
 
if __name__ == "__main__":
    pass