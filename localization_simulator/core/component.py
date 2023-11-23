import numpy as np
from .error import Error
class Anchor:

    def __init__(self,name, location, cutoff, error, clr) -> None:
        self.name = name
        self.location = location
        self.nDim = len(location)
        self.cutoff = cutoff
        self.error = Error(error[0],error[1])
        self.clr = clr

    def getDist(self, pose):
        return np.linalg.norm(pose-self.location)
        

class Robot:

    def __init__(self, trajectory) -> None:
        self.curPose = None
        self.trajectory = trajectory

        initialPose = trajectory[0]
        if all(len(x) == len(initialPose) for x in trajectory):
            self.nDim = len(initialPose)
        else:
            raise ValueError("Pose Dimensionality is not consistent")
        

if __name__ == "__main__":
    print("hello")