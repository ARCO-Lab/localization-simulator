import autograd.numpy as np
from autograd import grad, hessian
class NLS:

    def __init__(self,points,gradNorms, anchors, variance, max_error) -> None:
        self.points = points
        self.gradNorms = gradNorms
        self.anchors = anchors
        self.variance = variance
        self.max_error = max_error

    def addNoise(self, pose):
        x, y = pose
        distances = np.sqrt((self.anchors[:, 0] - x)**2 + (self.anchors[:, 1] - y)**2)
        noisy_distances = distances + np.random.normal(0, np.sqrt(self.variance), len(distances))
        return noisy_distances

    def eq(self, est_pose):
        est_x, est_y = est_pose
        est_distances = np.sqrt((self.anchors[:, 0] - est_x)**2 + (self.anchors[:, 1] - est_y)**2)
        return est_distances

    # Using Newtons method and autograd for gradient and hessian
    def estimatePose(self, est_pose, measurements):
        est_x, est_y = est_pose

        print(f"est pose: {est_pose}")
        print(f"measurement: {measurements}")

        g = grad(lambda p: np.sum((self.eq(p) - measurements)**2))(est_pose)
        h = hessian(lambda p: np.sum((self.eq(p) - measurements)**2))(est_pose)
        
        h_inv = np.linalg.pinv(h)
        delta_pose = np.dot(h_inv, g)
        
        est_x -= delta_pose[0]
        est_y -= delta_pose[1]
        
        return np.array([est_x, est_y]), np.linalg.norm(g)

    def process(self, pose, guess):
        p = []
        grads = []
        p.append(pose)
        p.append(guess)
        # print(f"The guess is: {guess}")
        # Add a max iteration and look at a comparison between norm of gradient less than a tolerance
        while True:
            measurements = self.addNoise(pose)
            guess, grad = self.estimatePose(guess, measurements)
            # print(f"Guess for Pose {pose}: {guess}")
            p.append(guess)
            grads.append(grad)
            if grad <= self.max_error:
                break
        self.points.append(p)
        self.gradNorms.append(grads)

if __name__ == "__main__":
    a = NLS(None, np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 2.0]]), 0.01, 0.1)
    a.process(np.array([4.0, 4.0]), np.array([1.0, 1.0]))