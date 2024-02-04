"""
A module for creating error models that can be integrated into individual anchors/beacons/sensors/robots (currently not being used).

Todo:
    * Research and finalize on error model for anchors
    * Fully implement and add to anchors
    * Add exception handling
    * Update chen
"""

from enum import Enum
import numpy as np
import scipy.stats as stats

def gonzalez(distance):
    """UWB bias model (Loss of sight conducted on 1-10m experiments) based on Gonzalez et al.

    Args:
        distance (float): The distance from the robot to the UWB sensor.

    Returns:
        (float): The distance bias

    Raises:

    """
    return 0.1*(1.01-np.exp(-0.17*distance))

def chen(envType,distance,exp,lognormrand):
    """UWB bias model (No Loss of sight on 100-1300 m) based on chen et al.

    Args:
        envType (str): The distance from the robot to the UWB sensor.
        distance (float): The distance from the robot to the UWB sensor.
        exp (float): An exponent used in the calculation (Update).
        lognormrand (float): A value used in the calculation (update).

    Returns:
        (float): The distance bias

    Raises:
        Exception: For any error it encounters (Change)
    """
    types = {
        "bad_urban": 1,
        "urban": 0.4,
        "suburban":0.3,
        "rural":0.1
    }

    try:
        T1 = types[envType]
    except:
        raise Exception
    
    trms = T1*(distance/1000**exp)*lognormrand
    t = 2
    return ((1/trms)*np.exp(-(t/trms)))*1000

class Bias(Enum):
    """Enum class for bias models
    """
    gonzalez = gonzalez
    chen = chen

class Variance(Enum):
    """Enum class for known sensor variance
    """
    uwb_skylab_sku603 = 10
    uwb_Zebra = 10/3


class Error:
    """Error module that can be used by components.

    Attributes:
        mean (float): The distribution mean
        std_dev (float): The distribution variance
    """
    def __init__(self, mean, variance) -> None:
        """Init Method

        Args:
            mean (float): The distance from the robot to the UWB sensor.
            std_dev (float): The distance from the robot to the UWB sensor.
        """
        self.mean = mean
        self.variance = variance
    
    def getPDF(self, dist, mean=None):
        return stats.norm.pdf(dist, loc=self.mean if mean is None else mean, scale=np.sqrt(self.variance))


if __name__ == "__main__":
    pass
    # mean = 10
    # std_dev = 2
    # x = 12

    # new_pdf = Error.getPDF(x,mean,std_dev)

    # print("new PDF:", new_pdf)

