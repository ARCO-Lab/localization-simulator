
class Anchor:

    def __init__(self, location, cutoff, strength, noise) -> None:
        self.location = location
        self.nDim = len(location)
        self.cutoff = cutoff
        self.strength = strength
        self.noise = noise

class Robot:

    def __init__(self, trajectory) -> None:
        self.curPose = None
        self.trajectory = trajectory

        initialPose = trajectory[0]
        if all(len(x) == len(initialPose) for x in trajectory):
            self.nDim = len(initialPose)
        else:
            raise ValueError("Pose Dimensionality is not consistent")