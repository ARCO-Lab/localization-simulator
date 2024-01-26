"""
A module for using Newtons method to approximate position at a given pose using the norm of the gradient.

Todo:
    * Potentially look at adding a dampening factor in case of instability or excessive oscillation.
    * Do more to decouple nls from maps.
"""

import autograd.numpy as np
from autograd import grad, hessian
class NLS:
    """Newton solver for approximating the solution the solution at a given timestep.

    Attributes:
        points (list[list[numpy.ndarray[float]]]): All initial and subsequent approximations for each timestep.
        gradNorms (list[list[float]]): Gradient norms for the intial and subsequent approximations for each timestep.
        variance (float): The variance to be used for adding noise to measurements.
        tolerance (float): the maximum error before Newtons method stops iterating.
    """
    def __init__(self,points,gradNorms, variance, tolerance) -> None:
        """Init method

        Args:
            points (list[list[numpy.ndarray[float]]]): All initial and subsequent approximations for each timestep.
            gradNorms (list[list[float]]): Gradient norms for the intial and subsequent approximations for each timestep.
            variance (float): The variance to be used for adding noise to measurements.
            tolerance (float): the maximum error before Newtons method stops iterating.
        """
        self.points = points
        self.gradNorms = gradNorms
        self.variance = variance
        self.tolerance = tolerance

    def addNoise(self, pose, anchors):
        """Add noise to distance measurements.

        Gaussian noise with a mean of 0 and a standard deviation determined by the specified variance is then added to
        these distances to simulate measurement noise.

        Args:
            pose (numpy.ndarray[float]): The current pose of the robot
            anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).


        Returns:
            (numpy.ndarray[float]): Noisy distance measurements from the robot to each anchor.
        """
        x, y = pose
        distances = np.sqrt((anchors[:, 0] - x)**2 + (anchors[:, 1] - y)**2)
        noisy_distances = distances + np.random.normal(0, np.sqrt(self.variance), len(distances))
        return noisy_distances

    def eq(self, est_pose, anchors):
        """ 
        Calculates the Euclidean distances from anchor points to given pose.

        Used as part of the objective function.

        Args:
            est_pose (numpy.ndarray[float]s): The pose estimate
            anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).

        Returns:
            (numpy.ndarray[float]):  Euclidean distances from the estimated pose to each anchor
        """
        est_x, est_y = est_pose
        est_distances = np.sqrt((anchors[:, 0] - est_x)**2 + (anchors[:, 1] - est_y)**2)
        return est_distances

    # Using Newtons method and autograd for gradient and hessian
    def estimatePose(self, est_pose, measurements, anchors):
        """Estimate the pose using Newtons method. 

        Given an initial estimated pose and distance measurements, the method iteratively updates the estimated
        pose using Newton's method. The error function is defined as the squared sum of differences between the estimated
        distances and the actual measurements. Autograd is utilized to calculate the gradient and hessian of this error
        function with respect to the estimated pose.

        Args:
            est_pose (numpy.ndarray[float]): The pose estimate
            measurements (numpy.ndarray[float]): The noisy measurments from the pose to anchors
            anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).

        Returns:
            (tuple[numpy.ndarray[float], float]): The updated estimated pose and the norm of the gradient.
        """
        est_x, est_y = est_pose

        g = grad(lambda p, q: np.sum((self.eq(p, q) - measurements)**2))(est_pose, anchors)
        h = hessian(lambda p, q: np.sum((self.eq(p, q) - measurements)**2))(est_pose, anchors)

        h_inv = np.linalg.pinv(h)
        delta_pose = np.dot(h_inv, g)
        
        est_x -= delta_pose[0]
        est_y -= delta_pose[1]
        
        return np.array([est_x, est_y]), np.linalg.norm(g)

    def process(self, pose, guess, anchors):
        """Perform Newtons method to refine an intial guess for the pose based on distance measurements until a tolerance is met.

        Args:
            pose (numpy.ndarray[float]): The actual pose
            guess (numpy.ndarray[float]): The intial guess for a given poses. A randomly determined point on the map.
            anchors (list[tuple[float]]): Locations of each anchor (either 2D or 3D).
        """
        p = []
        grads = []
        p.append(pose)
        p.append(guess)
        while True:
            measurements = self.addNoise(pose, anchors)
            guess, grad = self.estimatePose(guess, measurements, anchors)
            p.append(guess)
            grads.append(grad)
            if grad <= self.tolerance:
                break
        self.points.append(p)
        self.gradNorms.append(grads)

if __name__ == "__main__":
    a = NLS(None, np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 2.0]]), 0.01, 0.1)
    a.process(np.array([4.0, 4.0]), np.array([1.0, 1.0]))