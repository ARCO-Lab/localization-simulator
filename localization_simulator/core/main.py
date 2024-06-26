"""
The main script to run the simulator

Todo:
    * Improve the process of creating routines
"""
from .component import Anchor
from .map import Map
from .alg import *
from ..utils.plot import Plot, postPlot
from ..utils.helper import parseArgs
import numpy as np
import yaml
import os

def customRoutine(config, headless=False):
    """A custom routine as provided by a config yaml file

    Args:
        config (dict): The yaml file safe loaded into a dictionary format
    """
    anchorList = [Anchor(i["name"],i["location"],i["cutoff"],i["error"],i["colour"]) for i in config["anchor"]]
    m = Map(config["map"])
    m.setK(config["k"])
    m.setVariance(config["sensor_measurement_variance"])
    m.setIsotropic(config["isotropic"])
    m.placeAnchor(anchorList)
    m.loadTraj(config["pose"],6)
    if headless:
        m.headless()
    else:
        m.visualize2D() if m.nDim == 2 else m.visualize3D()
        postPlot(m.points, m.gradNorms)

def sampleRoutine2d():
    """A sample routine for a 2D simulation
    """
    anchorList2d = [Anchor("a",(3,3),0,(0,3),"red"),Anchor("b",(14,14),0,(0,3),"blue"),Anchor("c",(18,12),0,(0,3),"red")]
    m = Map((20,20))
    m.placeAnchor(anchorList2d)
    m.loadTraj([(0,0),(8,8),(6,6),(4,5),(10,4)],5)
    m.visualize2D()
    postPlot(m.points,m.gradNorms)

def sampleRoutine3d():
    """A sample routine for a 3D simulation
    """
    anchorList3d = [Anchor("a",(3,3,3),0,(0,3),"red"),Anchor("b",(14,14,14),0,(0,3),"blue"),Anchor("c",(18,12,13),0,(0,3),"red")]
    m = Map((20,20,20))
    m.placeAnchor(anchorList3d)
    m.loadTraj([(0,0,0),(8,8,8),(6,6,7),(4,5,4),(10,4,2)],12)
    m.visualize3D()

if __name__ == "__main__":
    # Used to parse arguments 
    use2d, use3d, config_file, headless, save = parseArgs()

    if config_file:
        with open(f"config/{config_file}.yaml","r") as file:
            y = yaml.safe_load(file)
            customRoutine(y,headless)
    else:
        if use2d == use3d:
            np.random.choice([sampleRoutine2d,sampleRoutine3d])()
        elif use2d:
            sampleRoutine2d()
        else:
            sampleRoutine3d()