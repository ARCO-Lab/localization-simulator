from enum import Enum
import numpy as np
import scipy.stats as stats

#LOS
# Model based on 1-10 m
def gonzalez(d):
    return 0.1*(1.01-np.exp(-0.17*d))

#NLOS
# Model based on 100-1300 m
def chen(envType,d,exp,lognormrand):

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
    


    trms = T1*(d/1000**exp)*lognormrand

    t = 2

    return ((1/trms)*np.exp(-(t/trms)))*1000



class Bias(Enum):
    gonzalez = gonzalez
    chen = chen

class Variance(Enum):
    uwb_skylab_sku603 = 10
    uwb_Zebra = 10/3


class Error:

    def __init__(self, mean, std_dev) -> None:
        self.mean = mean
        self.std_dev = std_dev
    
    def getPDF(self, dist, mean=None):
        return stats.norm.pdf(dist, loc=self.mean if mean is None else mean, scale=self.std_dev)


if __name__ == "__main__":

    mean = 10
    std_dev = 2
    x = 12

    new_pdf = Error.getPDF(x,mean,std_dev)

    print("new PDF:", new_pdf)