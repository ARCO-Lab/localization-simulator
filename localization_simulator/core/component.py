"""
A module for creating components that can be used within the simulation map such as anchors, robots, or other potential entities for the simulation environment.

Todo:
    * Fully implement and integrate robot component
    * Rework the trajectory module in utils as a robot component so multiple paths can be registered and other behaviour can be modelled in addition to trajectory
    * Add anchor cutoff
"""

import numpy as np

from .error import Error
class Anchor:
    """Component module to model anchor/sensor/beacon behaviour in a simulation environment.

    Attributes:
        name (str): Anchor name/id.
        location (tuple[float, float, optional(float)]): A tuple of length 2 (2D map) or 3 (3D map) to represent the location of the anchor.
        nDim (int): Anchor dimensionality (either 2 or 3).
        cutoff (float): Radius around the anchor, in which readings can only be made for objects within the cutoff.
        error (tuple[int]): Anchor error model.
        clr (str): Colour of anchor for visualization purposes.
    """
    def __init__(self,name, location, cutoff, error, clr) -> None:
        """Init method

        Args:
            name (str): Anchor name/id.
            location (tuple[float, float, optional(float)]): A tuple of length 2 (2D map) or 3 (3D map) to represent the location of the anchor.
            cutoff (float): Radius around the anchor, in which readings can only be made for objects within the cutoff.
            error (tuple[int]): Anchor error model.
            clr (str): Colour of anchor for visualization purposes.
        """
        self.name = name
        self.location = location
        self.nDim = len(location)
        self.cutoff = cutoff
        self.error = Error(error[0],error[1])
        self.clr = clr

    def getDist(self, pose):
        """Gets the euclidean distance between a pose and anchor.

        Args:
            pose (pandas.core.series.Series): The pose

        Returns:
            (float): The euclidean distance between the pose and anchor
        """
        return np.linalg.norm(pose-self.location)
        
class Robot:
    """Component module to model robot behaviour in a simulation environment (currently not used)

    Attributes:
        curPose (str): Current pose of the robot.
        trajectory (Trajectory): A designated robot trajectory.
        nDim (str): Robot dimensionality (either 2 or 3).
    """
    def __init__(self, trajectory) -> None:
        """Init Method

        Args:
            trajectory (Traj2D or Traj3D): A robot trajectory 

        Raises:
            ValueError: If the dimensionality in a trajectory ever changes.
        """
        self.curPose = None
        self.trajectory = trajectory

        initialPose = trajectory[0]
        if all(len(x) == len(initialPose) for x in trajectory):
            self.nDim = len(initialPose)
        else:
            raise ValueError("Pose Dimensionality is not consistent")
        
if __name__ == "__main__":
    pass