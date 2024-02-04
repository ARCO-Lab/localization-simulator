"""
A helper module for useful functions 

Todo:
    * N/A
"""
import argparse

def parseArgs():
     """Parses arguments for simulator. Currently used to select a defined 2D or 3D simulation as well as a custom routine defined by a config file.

     Returns:
         (list[bool]): The argument selections
     """
     retArgs = []
     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     parser.add_argument("--2", "-dim2", default = False, required = False, action="store_true", help="Flag to show 2D.")
     parser.add_argument("--3", "-dim3", default = False, required = False, action="store_true", help="Flag to show 3D.")
     parser.add_argument("--c", "-config", type = str, default = None, required = False, help="Flag to show 3D.")
     args = parser.parse_args()
     for arg in vars(args):
          retArgs.append(getattr(args, arg))
     return retArgs