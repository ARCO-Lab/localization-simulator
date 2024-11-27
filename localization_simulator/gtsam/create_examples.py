
from __future__ import print_function

import math
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Create noise models

# taken from example "PlanarSLAMExample.py" should be modeled eventually
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

# used to create Jacobian G necessary for covar calculation below
def jacobian_G(del_sr, del_sl, b):

    row_1 = [(1/2) * math.cos((del_sr - del_sl)/(2*b)) - ((del_sr + del_sl)/(4*b)) * math.sin((del_sr - del_sl)/(2*b)), 
             (1/2) * math.cos((del_sr - del_sl)/(2*b)) + ((del_sr + del_sl)/(4*b)) * math.sin((del_sr - del_sl)/(2*b)), 
            ]
    row_2 = [(1/2) * math.sin((del_sr - del_sl)/(2*b)) + ((del_sr + del_sl)/(4*b)) * math.cos((del_sr - del_sl)/(2*b)), 
             (1/2) * math.sin((del_sr - del_sl)/(2*b)) - ((del_sr + del_sl)/(4*b)) * math.cos((del_sr - del_sl)/(2*b)), 
            ]
    row_3 = [ 1/b,
              -1/b
            ]
    G = np.array([row_1, row_2, row_3])
    return G


# used to create the odometry noise matrix
def Sigma_yy(del_sr, del_sl, b, k_r, k_l):

    Sigma_xx = np.array([[k_r * abs(del_sr), 0], [0, k_l * abs(del_sl)]])

    G = jacobian_G(del_sr, del_sl, b)

    interim = np.matmul(G, Sigma_xx)

    final = np.matmul(interim, np.transpose(G))

    return final

# pass in list of lists of [r, theta] to calc necesarry wheel movements
def path_from_r_theta(given_list, b):

    #start at origin every time for now
    #at each step choose an r and theta to make the geometry work
    odoms = []
    path = [np.array([0, 0, 0])]

    for given in given_list:
        if given[1] > 0:
            sl = abs(given[1]) * given[0]
            sr = abs(given[1]) * (given[0] + b)

        elif given[1] < 0:
            sl = abs(given[1]) * (given[0] + b)
            sr = abs(given[1]) * given[0]

        # if theta = 0 use r as a distance travelled in a straight line instead
        else:
            sl = given[0]
            sr = given[0]

        odoms.append([sr, sl])

        #position calculation
        x = (sr + sl)/2 * math.cos((sr-sl)/(2*b))
        y = (sr + sl)/2 * math.sin((sr-sl)/(2*b))
        theta = (sr - sl)/b 

        path.append(np.array([x, y, theta]))
    
    return odoms, path


# take in list of given r, theta measurments and gives factor graph

def factor_graph(r_theta_list):

    b = 0.3

    odom_move, rel_transforms = path_from_r_theta(r_theta_list, b)

    num_poses = len(rel_transforms)

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # pose dictionary
    pose_keys = {}

    # initialize pose variables
    for i in range(num_poses):

        key_name = f"X{i}"
        key_value = X(i)
        pose_keys[key_name] = key_value

    

    # Add a prior on pose X0
    
    # NOTE: need to decide how we're modelling the PRIOR_NOISE instead of what the example did
    graph.add(
        gtsam.PriorFactorPose2(pose_keys['X0'], gtsam.Pose2(rel_transforms[0]), PRIOR_NOISE))

    # choose k factors needed for covar matrix calculation
    k_l = 0.005
    k_r = 0.005

    # Add odometry factors between following poses
    for i in range(num_poses -1):
        # calculate covar matrix at each step instead of using a blanket model because the magnitude of wheel changes directly influence amount of noise
        # need to understand which noise model to plug this matrix into to be an accurate noise measurement
        noise_matrix = Sigma_yy(odom_move[0][0], odom_move[0][1], 0.3, k_r, k_l)

        graph.add(
            gtsam.BetweenFactorPose2(pose_keys[f"X{i}"], pose_keys[f"X{i+1}"], gtsam.Pose2(rel_transforms[i+1]), ODOMETRY_NOISE)) 
    

    # Print graph
    print("Factor Graph:\n{}".format(graph))

    # initialize the list of path poses
    full_estimate = [rel_transforms[0]]

    # dummy calculation of just adding together relative transformations to create pose estimates
    for i in range(num_poses - 1):
        full_estimate.append(rel_transforms[i] + rel_transforms[i + 1])

    # use calc above as the initial (noisy) estimate
    initial_estimate = gtsam.Values()
    for i in range(num_poses):
        initial_estimate.insert(pose_keys[f"X{i}"], gtsam.Pose2(full_estimate[i]))
        
  
    print("Initial Estimate:\n{}".format(initial_estimate))

    # Optimize using Levenberg-Marquardt optimization. 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    # Calculate and print marginal covariances for all variables
    marginals = gtsam.Marginals(graph, result)
   
    for key in pose_keys:
        print("{} covariance:\n{}\n".format(key, marginals.marginalCovariance(pose_keys[key])))


    for key in pose_keys:         
        gtsam_plot.plot_pose2(0, result.atPose2(pose_keys[key]), 0.5,
                              marginals.marginalCovariance(pose_keys[key]))
 
    
    plt.axis('equal')
    plt.show()


    return None





givens = [[2, -math.pi/4], [1, math.pi/4], [2, -math.pi/3], [1, 0], [2, math.pi/4]]

factor_graph(givens)

