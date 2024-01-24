import argparse

def parseArgs():
        retArgs = []
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--2", "-dim2", default = False, required = False, action="store_true", help="Flag to show 2D.")
        parser.add_argument("--3", "-dim3", default = False, required = False, action="store_true", help="Flag to show 3D.")
        args = parser.parse_args()
        for arg in vars(args):
             retArgs.append(getattr(args, arg))
        return retArgs