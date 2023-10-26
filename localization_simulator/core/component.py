
class Anchor:

    def __init__(self, location, cutoff, strength, noise, clr) -> None:
        self.location = location
        self.nDim = len(location)
        self.cutoff = cutoff
        self.strength = strength
        self.noise = noise
        self.clr = clr

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