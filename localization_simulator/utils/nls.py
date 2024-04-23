"""
A module for using Newtons method to approximate position at a given pose using the norm of the gradient.

Todo:
    * Potentially look at adding a dampening factor in case of instability or excessive oscillation.
    * Do more to decouple nls from maps.
"""

import autograd.numpy as np
from autograd import grad, hessian
import copy
class NLS:
    """Newton solver for approximating the solution the solution at a given timestep.

    Attributes:
        points (list[list[numpy.ndarray[float]]]): All initial and subsequent approximations for each timestep.
        gradNorms (list[list[float]]): Gradient norms for the intial and subsequent approximations for each timestep.
        anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).
        variance (float): The variance to be used for adding noise to measurements.
        tolerance (float): the maximum error before Newtons method stops iterating.
    """
    def __init__(self,points,gradNorms,anchors,tolerance, distances = None,variance= None) -> None:
        """Init method

        Args:
            points (list[list[numpy.ndarray[float]]]): All initial and subsequent approximations for each timestep.
            gradNorms (list[list[float]]): Gradient norms for the intial and subsequent approximations for each timestep.
            anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).
            variance (float): The variance to be used for adding noise to measurements.
            tolerance (float): the maximum error before Newtons method stops iterating.
        """
        self.points = points
        self.gradNorms = gradNorms
        self.anchors = anchors
        self.distances = distances
        self.variance = variance
        self.tolerance = tolerance

    def addNoise(self, pose):
        """Add noise to distance measurements.

        Gaussian noise with a mean of 0 and a standard deviation determined by the specified variance is then added to
        these distances to simulate measurement noise.

        Args:
            pose (numpy.ndarray[float]): The current pose of the robot


        Returns:
            (numpy.ndarray[float]): Noisy distance measurements from the robot to each anchor.
        """
        x, y = pose
        distances = np.sqrt((self.anchors[:, 0] - x)**2 + (self.anchors[:, 1] - y)**2)
        noisy_distances = distances + np.random.normal(0, np.sqrt(self.variance), len(distances))
        return noisy_distances

    def eq(self, est_pose):
        """ 
        Calculates the Euclidean distances from anchor points to given pose.

        Used as part of the objective function.

        Args:
            est_pose (numpy.ndarray[float]s): The pose estimate

        Returns:
            (numpy.ndarray[float]):  Euclidean distances from the estimated pose to each anchor
        """
        est_x, est_y = est_pose
        est_distances = np.sqrt((self.anchors[:, 0] - est_x)**2 + (self.anchors[:, 1] - est_y)**2)
        return est_distances
    
    def eqMask(self, est_pose, mask):
        """ 
        Calculates the Euclidean distances from anchor points to given pose.

        Used as part of the objective function.

        Args:
            est_pose (numpy.ndarray[float]s): The pose estimate

        Returns:
            (numpy.ndarray[float]):  Euclidean distances from the estimated pose to each anchor
        """
        est_x, est_y = est_pose
        est_distances = np.sqrt((self.anchors[:, 0] - est_x)**2 + (self.anchors[:, 1] - est_y)**2)
        est_distances = est_distances[mask]
        return est_distances

    # Using Newtons method and autograd for gradient and hessian
    def estimatePose(self, est_pose, measurements, pose, variance, isotropic):
        """Estimate the pose using Newtons method. 

        Given an initial estimated pose and distance measurements, the method iteratively updates the estimated
        pose using Newton's method. The error function is defined as the squared sum of differences between the estimated
        distances and the actual measurements. Autograd is utilized to calculate the gradient and hessian of this error
        function with respect to the estimated pose.

        Args:
            est_pose (numpy.ndarray[float]): The pose estimate
            measurements (numpy.ndarray[float]): The noisy measurments from the pose to anchors

        Returns:
            (tuple[numpy.ndarray[float], float]): The updated estimated pose and the norm of the gradient.
        """
        est_x, est_y = est_pose

        measurements = np.array(measurements)
        mask = ~np.isnan(measurements)
        valid_measurements = measurements[mask]

        g = grad(lambda p: (np.dot(np.transpose(pose-p),np.linalg.inv(isotropic))@(pose-p)) + np.sum((1 / (variance)) * (self.eqMask(p, mask) - valid_measurements) ** 2))(est_pose)
        h = hessian(lambda p: (np.dot(np.transpose(pose-p),np.linalg.inv(isotropic))@(pose-p)) + np.sum((1 / (variance)) * (self.eqMask(p,mask) - valid_measurements) ** 2))(est_pose)

        # print(f"ACTUAL GRAD {g}")
        # print(h)

        h_inv = np.linalg.pinv(h)
        delta_pose = np.dot(h_inv, g) 
        
        est_x -= delta_pose[0]
        est_y -= delta_pose[1]
        
        return np.array([est_x, est_y]), np.linalg.norm(g), h

    def process(self, pose, guess, distances=None):
        """Perform Newtons method to refine an intial guess for the pose based on distance measurements until a tolerance is met.

        Args:
            pose (numpy.ndarray[float]): The actual pose
            guess (numpy.ndarray[float]): The intial guess for a given poses. A randomly determined point on the map.
        """
        p = []
        grads = []
        p.append(pose)
        p.append(guess)
        while True:
            measurements = self.addNoise(pose)
            guess, grad = self.estimatePose(guess, measurements)
            p.append(guess)
            grads.append(grad)
            if grad <= self.tolerance:
                break
        self.points.append(p)
        self.gradNorms.append(grads)
    
    def rmse(self, pose, guess, distances, anchors, variance, isotropic):
        self.anchors = anchors
        # print(f"Pose is {pose}\tguess is {guess}\tdistances is {distances}")
        counter = 0
        # guess = np.array(guess)

        guess = np.array(guess)

        Real = copy.deepcopy(guess)

        pose = pose.astype(float)
        guess = guess.astype(float)
        guess = np.array(guess)

        while True:
            # print(Real)
            guess, grad, _ = self.estimatePose(guess, distances, pose, variance, isotropic)
            # print(f"Aprrox {counter}:{guess}")
            counter += 1
            if grad <= self.tolerance:
                break
        return np.linalg.norm(guess-Real)**2

    
    def gradNorm(self, pose, guess, distances, anchors, variance, isotropic):
        sum = 0
        self.anchors = anchors
        # print(f"Pose is {pose}\tguess is {guess}\tdistances is {distances}")
        counter = 0
        # print("BUG____________________")
        while True:
            sum += np.linalg.norm(guess-pose)**2
            guess, grad, _ = self.estimatePose(guess, distances, pose, variance, isotropic)
            # print(f"Aprrox {counter}:{guess}")
            counter += 1
            if grad <= self.tolerance:
                break
        return grad

    def hesEig(self, pose, guess, distances, anchors, variance, isotropic):
        sum = 0
        self.anchors = anchors
        # print(f"Pose is {pose}\tguess is {guess}\tdistances is {distances}")
        counter = 0
        # print("BUG____________________")
        while True:
            sum += np.linalg.norm(guess-pose)**2
            guess, grad, hessian = self.estimatePose(guess, distances, pose, variance, isotropic)
            # print(f"Aprrox {counter}:{guess}")
            counter += 1
            if grad <= self.tolerance:
                break
        return np.linalg.eigvalsh(hessian)

if __name__ == "__main__":
    # a = NLS(None, np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 2.0]]), 0.01, 0.1)
    # a.process(np.array([4.0, 4.0]), np.array([1.0, 1.0]))
    import numpy as np

    # Example distance array with NaN values
    distances = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])

    # Mask NaN values
    valid_distances = distances[~np.isnan(distances)]
    # print(valid_distances)