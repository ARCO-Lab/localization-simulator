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
import matplotlib.patches as patches

# Create noise models

# taken from example "PlanarSLAMExample.py" should be modeled eventually
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
APRIL_TAG_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1])) #using initial diagonal estimate for now - need to put in noise transform

# constants used in the covariance noise matrix in odometry - dependant on the interaction between robot and environment - should be modelled eventually
K_R = 0.001
K_L = 0.001

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

# creates odometry measurements from list of given turn radius and theta
# b is distance between wheels
def odom_from_r_theta(given_list, b):

    #start at origin every time for now
    #at each step choose an r and theta to make the geometry work
    odoms = []

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

    
    return odoms

# given a list of odometry measurements (delta sr, sl), inject noise and calculate x, y, theta position
# calculation done in robot reference frame
def odom_calculation(given_list, b):
    odoms = []
    path = [np.array([0, 0, 0])]


    for given in given_list:

        std_dev_l = math.sqrt(K_L* abs(given[1]))
        std_dev_r = math.sqrt(K_R* abs(given[0]))

        noise_sr = std_dev_r * np.random.randn()
        noise_sl = std_dev_l * np.random.randn()

        del_sr = given[0] + noise_sr
        del_sl = given[1] + noise_sl

        odoms.append([del_sr, del_sl])
    
        del_x = (del_sr + del_sl)/2 * math.cos((del_sr - del_sl)/(2*b))
        del_y = (del_sr + del_sl)/2 * math.sin((del_sr - del_sl)/(2*b))
        del_theta = (del_sr - del_sl)/b

        path.append(np.array([del_x, del_y, del_theta]))
    
    return odoms, path

# computes estimate in world frame without noise for usage plotting ground truth
def no_noise_odom(given_list, b):

    path = [np.array([0, 0, 0])] 
    i = 0  

    for given in given_list:
       

        del_sr = given[0]
        del_sl = given[1]
    
        del_x = path[i][0] + (del_sr + del_sl)/2 * math.cos(path[i][2]+ (del_sr - del_sl)/(2*b))
        del_y = path[i][1] + (del_sr + del_sl)/2 * math.sin(path[i][2]+ (del_sr - del_sl)/(2*b))
        del_theta = path[i][2] + (del_sr - del_sl)/b

        path.append(np.array([del_x, del_y, del_theta]))

        i += 1

    return path

# given 3x3 transformation matrix, extract the 2D pose defined as x, y, theta
def pose_from_tranform(T):

    x = T[0,2]
    y = T[1, 2]
    theta = np.arctan2(T[1,0], T[0,0])

    pose = np.array([x, y, theta])

    return pose


# odom_meas = a list of odometry measurements that represent the path driven by robot
# april_meas = list of lists of format [pose, april tag, measurement (Tar)]
# april_loc = dictionary of april tags [april number, location (Twa)]
# b = distance between wheels on the robot
def main(odom_meas, april_meas, april_loc, b):

    odom_move, rel_transforms = odom_calculation(odom_meas, b)

    num_poses = len(rel_transforms)
    num_april_meas = len(april_meas)
   

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    pose_keys = {}
    

    for i in range(num_poses):
        key_name = f"X{i}"
        key_value = X(i)
        pose_keys[key_name] = key_value


    # Add a prior on pose X0 at the origin. A prior factor consists of a mean and a noise model
    graph.add(
        gtsam.PriorFactorPose2(pose_keys['X0'], gtsam.Pose2(rel_transforms[0]), PRIOR_NOISE))

    
    for i in range(num_poses -1):
        # calculate covar matrix at each step instead of using a blanket model because the magnitude of wheel changes directly influence amount of noise
        
        noise_matrix = Sigma_yy(odom_move[i][0], odom_move[i][1], b, K_R, K_L) + 10**(-6)*np.identity(3)

        graph.add(
            gtsam.BetweenFactorPose2(pose_keys[f"X{i}"], pose_keys[f"X{i+1}"], gtsam.Pose2(rel_transforms[i+1]), gtsam.noiseModel.Gaussian.Covariance(noise_matrix))) 
        
    # add april tag factors as priors on pose
    # ********* print statements are here to check usage but not sure if this is working correctly yet
    for i in range(num_april_meas):
        # matrix multiply Twa * Tar
        Twr = np.matmul(april_loc[april_meas[i][1]], april_meas[i][2])
        print(Twr)
        print(pose_from_tranform(Twr))
        graph.add(
            gtsam.PriorFactorPose2(pose_keys[april_meas[i][0]], gtsam.Pose2(pose_from_tranform(Twr)), APRIL_TAG_NOISE)
        )

    # Print graph
    print("Factor Graph:\n{}".format(graph))

    initial_estimate = gtsam.Values()

    # calculate ground truth for plotting with no noise
    full_estimate = no_noise_odom(odom_meas, 0.3)
    print("Initial values: \n")

    
    for i in range(num_poses):
        initial_estimate.insert(pose_keys[f"X{i}"], gtsam.Pose2(full_estimate[i]))

    # Print
    print("Initial Estimate:\n{}".format(initial_estimate))

    # Optimize using Levenberg-Marquardt optimization. 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result Full Noise:\n{}".format(result))


    # calculate marginals and plot poses and ground truth (full_estimate)
    marginals = gtsam.Marginals(graph, result)

    for key in pose_keys:
        print("{} covariance:\n{}\n".format(key, marginals.marginalCovariance(pose_keys[key])))

    for key in pose_keys:         
        gtsam_plot.plot_pose2(0, result.atPose2(pose_keys[key]), 0.5,
                              marginals.marginalCovariance(pose_keys[key]))
        
    for pose in full_estimate:
        gtsam_plot.plot_point2(0, pose[0:2], "c")

        
    plt.axis('equal')
    plt.show()

# defined this helper function to make an 'easy' test case for AprilTags
def create_T():

    Twa2 = np.array([[0, -1, 6], [1, 0, 1.5], [0, 0, 1]])
    Twa2_inv = np.linalg.inv(Twa2)

    Twr2 = np.array([[math.cos(math.pi/4), -math.sin(math.pi/4), 2.83445528], [math.sin(math.pi/4), math.cos(math.pi/4), 0.34564269], [0, 0, 1]])
    Twr3 = np.array([[math.cos(math.pi/4), -math.sin(math.pi/4), 4.95577562], [math.sin(math.pi/4), math.cos(math.pi/4), 2.46696304], [0, 0, 1]])

    Tar2 = np.matmul(Twa2_inv, Twr2)
    Tar3 = np.matmul(Twa2_inv, Twr3)

    return Tar2, Tar3

if __name__ == "__main__":
    
    move = [[2, 0], [1, math.pi/4], [3, 0]]
    odoms = odom_from_r_theta(move, 0.3)

    # hand designed april tag meas matrices for this test case
    Tar0 = np.array([[0, -1, 2], [1, 0, -1], [0, 0, 1]])
    Tar1 = np.array([[0, -1, 2], [1, 0, 1], [0, 0, 1]])
    Tar2 = np.array([[ 0.70710678,  0.70710678, -1.15435731], [-0.70710678,  0.70710678,  3.16554472], [ 0, 0, 1]])
    Tar3 = np.array([[ 0.70710678,  0.70710678,  0.96696304], [-0.70710678,  0.70710678,  1.04422438], [ 0,  0,  1]])

    April_1 = np.array([[0, 1, 1], [-1, 0, 2], [0, 0, 1]])
    April_2 = np.array([[0, -1, 6], [1, 0, 1.5], [0, 0, 1]])

    april_dic = {"A1": April_1, "A2": April_2}

    april_meas = [["X0", "A1", Tar0], ["X1", "A1", Tar1], ["X2", "A2", Tar2], ["X3", "A2", Tar3]]
    
    
    main(odoms, april_meas , april_dic, 0.3)
    

    

    
