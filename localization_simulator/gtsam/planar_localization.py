"""
Based on example from https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/PlanarSLAMExample.py

Original example authors: Alex Cunningham (C++), Kevin Deng & Frank Dellaert (Python)
"""


from __future__ import print_function

import math
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X

# Create noise models

# taken from example "PlanarSLAMExample.py" should be modeled eventually
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
MEASUREMENT_NOISE_2 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1])) #testing the range only model


# calculate the change in x, y, theta based on wheel odometry measurements
# takes in wheel_changes = [del_sr, del_sl] -> right and left wheel distance travelled respectively

#currently calculates in the robot's reference frame
def odom_calculation(given_list, b):
    odoms = []
    path = [np.array([0, 0, 0])]

    for given in given_list:

        del_sr = given[0]
        del_sl = given[1]

        odoms.append([del_sr, del_sl])
    
        del_x = (del_sr + del_sl)/2 * math.cos((del_sr - del_sl)/(2*b))
        del_y = (del_sr + del_sl)/2 * math.sin((del_sr - del_sl)/(2*b))
        del_theta = (del_sr - del_sl)/b

        path.append(np.array([del_x, del_y, del_theta]))

    return odoms, path

# calculate odometry movements (del_sr, del_sl) from poses given
# takes list of poses as argument and returns list of odometry movements
# each pose in list consists of (x, y, theta)

def pose_to_odometry(poses):
    b = 0.30 #distance between wheels (in meters) this is a guess that I will update
    num_poses = len(poses)
    odoms = []

    # stores delta values for change in x, y, theta between each set of poses
    del_travel = []

    # calculate the delta vector
    for i in range(1, num_poses):
        del_travel.append(poses[i] - poses[i-1]) 

    #print(del_travel, "\n")
        
    # calculate the del_sr, del_sl from the delta vector (del_x, del_y, del_theta)
    for j in range(0, num_poses-1):
        del_x = del_travel[j][0]
        del_y = del_travel[j][1]
        del_theta = del_travel[j][2]

        # new strategy for calculation

        # need to adapt to the current state of the overleaf document

        d = math.sqrt(del_x**2 + del_y**2)
        
        
        if del_theta > 0:
            # turning ccw
            r = (d * math.sin((math.pi- abs(del_theta))/2) / math.sin(abs(del_theta))) - b/2
            del_sl = del_theta * r
            del_sr = del_theta * (r + b)

        elif del_theta == 0:
            del_sl  = d
            del_sr = d
        
        else:
            # turning cw
            r = (d * math.sin((math.pi- abs(del_theta))/2) / math.sin(abs(del_theta))) - b/2
            del_sl = abs(del_theta) * (r + b)
            del_sr = abs(del_theta) * r

        
        odoms.append([del_sr, del_sl])
        
    return odoms

def main(odom_meas, beacon_placement, beacon_meas):

    b = 0.3

    odom_move, rel_transforms = odom_calculation(odom_meas, b)

    num_poses = len(rel_transforms)
    num_beacons = len(beacon_placement)

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    pose_keys = {}
    landmark_keys = {}

    for i in range(num_poses):
        key_name = f"X{i}"
        key_value = X(i)
        pose_keys[key_name] = key_value

    for j in range(num_beacons):
        key_name = f"L{j}"
        key_value = L(j)
        landmark_keys[key_name] = key_value

    # Add a prior on pose X0 at the origin. A prior factor consists of a mean and a noise model
    graph.add(
        gtsam.PriorFactorPose2(pose_keys['X0'], gtsam.Pose2(rel_transforms[0]), PRIOR_NOISE))

    
    
    for i in range(num_poses -1):
        # calculate covar matrix at each step instead of using a blanket model because the magnitude of wheel changes directly influence amount of noise
        # need to understand which noise model to plug this matrix into to be an accurate noise measurement
        # noise_matrix = Sigma_yy(odom_move[0][0], odom_move[0][1], 0.3, k_r, k_l)

        graph.add(
            gtsam.BetweenFactorPose2(pose_keys[f"X{i}"], pose_keys[f"X{i+1}"], gtsam.Pose2(rel_transforms[i+1]), ODOMETRY_NOISE)) 
    

    # loop this to match the previous strategy but need to match poses to the measurements corresponding
    for i in range(len(beacon_meas)):
        
        graph.add(
            gtsam.RangeFactor2D(pose_keys[beacon_meas[i][0]], landmark_keys[beacon_meas[i][1]], beacon_meas[i][2], MEASUREMENT_NOISE_2)) 
    
    

    # Print graph
    print("Factor Graph:\n{}".format(graph))



    initial_estimate = gtsam.Values()

    full_estimate = [rel_transforms[0]]

    # dummy calculation of just adding together relative transformations to create pose estimates
    for i in range(num_poses - 1):
        full_estimate.append(rel_transforms[i] + rel_transforms[i + 1])

    # use calc above as the initial (noisy) estimate
    initial_estimate = gtsam.Values()
    for i in range(num_poses):
        initial_estimate.insert(pose_keys[f"X{i}"], gtsam.Pose2(full_estimate[i]))

    for i in range(num_beacons):
        initial_estimate.insert(landmark_keys[f"L{i}"], gtsam.Point2(beacon_placement[i]))
        print(beacon_placement[i])

    # Print
    print("Initial Estimate:\n{}".format(initial_estimate))

    # Optimize using Levenberg-Marquardt optimization. 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    # Calculate and print marginal covariances for all variables
    # next steps to make adaptable to diff number of keys
    marginals = gtsam.Marginals(graph, result)

    for key in pose_keys:
        print("{} covariance:\n{}\n".format(key, marginals.marginalCovariance(pose_keys[key])))

    for key in landmark_keys:
        print("{} covariance:\n{}\n".format(key, marginals.marginalCovariance(landmark_keys[key])))


    for key in pose_keys:         
        gtsam_plot.plot_pose2(0, result.atPose2(pose_keys[key]), 0.5,
                              marginals.marginalCovariance(pose_keys[key]))
        
    for key in landmark_keys:
        gtsam_plot.plot_point2(0, result.atPoint2(landmark_keys[key]), 0.5,
                              marginals.marginalCovariance(landmark_keys[key]))
  
    
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    
    poses = [np.array([0, 0, 0]), np.array([1.5, 0, 0]), np.array([1, -.5, -math.pi/4])]
    odoms = pose_to_odometry(poses)

    case_1_odom = [[2, 2], [2, 2]]
    case_1_beacons = [np.array([2, 2]), np.array([2, 4])]
    case_1_meas = [("X0", "L0", math.sqrt(8)), ("X1", "L0", 2), ("X1", "L1", math.sqrt(8)), ("X2", "L1", 2)]
    
    main(case_1_odom, case_1_beacons, case_1_meas)

    
