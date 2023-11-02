from enum import Enum
import numpy as np

# def gonzalez(d):
#     return 0.1*(1.01-np.exp(-0.17*d))

class Noise(Enum):
    gonzalez = lambda d : 0.1*(1.01-np.exp(-0.17*d))


if __name__ == "__main__":
    print(Noise.gonzalez(10))