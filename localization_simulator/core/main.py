"""
The main script to run the simulator

Todo:
    * Improve the process of creating routines
"""
from .component import Anchor
from .map import Map
from ..utils.plot import Plot, postPlot
from ..utils.helper import parseArgs
import numpy as np

def sampleRoutine2d():
    """A sample routine for a 2D simulation
    """
    anchorList2d = [Anchor("a",(3,3),0,(0,3),"red"),Anchor("b",(14,14),0,(0,3),"blue"),Anchor("c",(18,12),0,(0,3),"red")]
    m = Map((20,20))
    m.placeAnchor(anchorList2d)
    m.loadTraj([(0,0),(8,8),(6,6),(4,5),(10,4)],6)
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
    use2d, use3d = parseArgs()

    if use2d == use3d:
        np.random.choice([sampleRoutine2d,sampleRoutine3d])()
    elif use2d:
        sampleRoutine2d()
    else:
        sampleRoutine3d()