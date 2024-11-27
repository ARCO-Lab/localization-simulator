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
# PRIOR_NOISE = gtsam.noiseModel.Gaussian.Covariance(np.array([[0, 0, 0], [0,0,0], [0,0,0]  ]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
# ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
MEASUREMENT_NOISE_2 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1])) #testing the range only model

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

# calculate the change in x, y, theta based on wheel odometry measurements
# takes in wheel_changes = [del_sr, del_sl] -> right and left wheel distance travelled respectively

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

#currently calculates in the robot's reference frame
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

        # new strategy for calulation

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



def plot_pose2_on_axes(axes, pose, axis_length=0.1, covariance=None, ellipse_color='b'):
    """
    Plot a 2D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        pose (gtsam.Pose2): The pose to be plotted.
        axis_length (float): The length of the camera axes.
        covariance (numpy.ndarray): Marginal covariance matrix to plot
            the uncertainty of the estimation.
        ellipse_color (str): Color of the covariance ellipse. Default is 'b' (blue).
    """
    # Get rotation and translation (center)
    gRp = pose.rotation().matrix()  # Rotation from pose to global
    t = pose.translation()
    origin = t

    # Draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'g-')

    if covariance is not None:
        pPp = covariance[0:2, 0:2]
        gPp = np.matmul(np.matmul(gRp, pPp), gRp.T)

        w, v = np.linalg.eig(gPp)

        # k = 2.296
        k = 5.0

        angle = np.arctan2(v[1, 0], v[0, 0])
        e1 = patches.Ellipse(origin, np.sqrt(w[0] * k), np.sqrt(w[1] * k),
                             np.rad2deg(angle), fill=False, edgecolor=ellipse_color)
        axes.add_patch(e1)

def plot_pose2(fignum, pose, axis_length=0.1, covariance=None,
               axis_labels=('X axis', 'Y axis'), ellipse_color='b'):
    """
    Plot a 2D pose on given figure with given `axis_length`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        pose (gtsam.Pose2): The pose to be plotted.
        axis_length (float): The length of the camera axes.
        covariance (numpy.ndarray): Marginal covariance matrix to plot
            the uncertainty of the estimation.
        axis_labels (iterable[string]): List of axis labels to set.
        ellipse_color (str): Color of the covariance ellipse. Default is 'b' (blue).
    """
    # Get figure object
    fig = plt.figure(fignum)
    axes = fig.gca()
    plot_pose2_on_axes(axes, pose, axis_length=axis_length,
                       covariance=covariance, ellipse_color=ellipse_color)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])

    return fig

def main(odom_meas):

    b = 0.3

    odom_move, rel_transforms = odom_calculation(odom_meas, b)

    num_poses = len(rel_transforms)
   

    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()
    graph_diag = gtsam.NonlinearFactorGraph()

    pose_keys = {}
    pose_keys_diag = {}
    

    for i in range(num_poses):
        key_name = f"X{i}"
        key_value = X(i)
        pose_keys[key_name] = key_value

        key_name = f"XD{i}"
        key_value = X(i + 20)
        pose_keys_diag[key_name] = key_value

    

    # Add a prior on pose X0 at the origin. A prior factor consists of a mean and a noise model
    graph.add(
        gtsam.PriorFactorPose2(pose_keys['X0'], gtsam.Pose2(rel_transforms[0]), PRIOR_NOISE))

    graph_diag.add(
        gtsam.PriorFactorPose2(pose_keys_diag['XD0'], gtsam.Pose2(rel_transforms[0]), PRIOR_NOISE))

    
    
    for i in range(num_poses -1):
        # calculate covar matrix at each step instead of using a blanket model because the magnitude of wheel changes directly influence amount of noise
        # need to understand which noise model to plug this matrix into to be an accurate noise measurement
        
        noise_matrix = Sigma_yy(odom_move[i][0], odom_move[i][1], b, K_R, K_L) + 10**(-6)*np.identity(3)
        noise_matrix_diag = Sigma_yy(odom_move[i][0], odom_move[i][1], b, K_R, K_L)

        graph.add(
            gtsam.BetweenFactorPose2(pose_keys[f"X{i}"], pose_keys[f"X{i+1}"], gtsam.Pose2(rel_transforms[i+1]), gtsam.noiseModel.Gaussian.Covariance(noise_matrix))) 
        graph_diag.add(
            gtsam.BetweenFactorPose2(pose_keys_diag[f"XD{i}"], pose_keys_diag[f"XD{i+1}"], gtsam.Pose2(rel_transforms[i+1]), gtsam.noiseModel.Diagonal.Sigmas(np.diag(noise_matrix_diag)))) 
    


    # Print graph
    print("Factor Graph Full Noise:\n{}".format(graph))

    print("Factor Graph Diagonal Noise\n{}".format(graph_diag))



    initial_estimate = gtsam.Values()
    initial_estimate_diag = gtsam.Values()

    full_estimate = no_noise_odom(odom_meas, 0.3)
    print("Initial values: \n")

    
    for i in range(num_poses):
        initial_estimate.insert(pose_keys[f"X{i}"], gtsam.Pose2(full_estimate[i]))
        initial_estimate_diag.insert(pose_keys_diag[f"XD{i}"], gtsam.Pose2(full_estimate[i]))

    # Print
    print("Initial Estimate:\n{}".format(initial_estimate))

    # Optimize using Levenberg-Marquardt optimization. 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("\nFinal Result Full Noise:\n{}".format(result))

    optimizer_diag = gtsam.LevenbergMarquardtOptimizer(graph_diag, initial_estimate_diag, params)
    result_diag = optimizer_diag.optimize()
    print("\nFinal Result Full Noise:\n{}".format(result_diag))

    # Calculate and print marginal covariances for all variables
    # next steps to make adaptable to diff number of keys
    marginals = gtsam.Marginals(graph, result)
    marginals_diag = gtsam.Marginals(graph_diag, result_diag)

    for key in pose_keys:
        print("{} covariance:\n{}\n".format(key, marginals.marginalCovariance(pose_keys[key])))

    for key in pose_keys_diag:
        print("{} covariance:\n{}\n".format(key, marginals_diag.marginalCovariance(pose_keys_diag[key])))


    for key in pose_keys:         
        gtsam_plot.plot_pose2(0, result.atPose2(pose_keys[key]), 0.5,
                              marginals.marginalCovariance(pose_keys[key]))
        
    for key in pose_keys_diag:         
        plot_pose2(0, result_diag.atPose2(pose_keys_diag[key]), 0.5,
                              marginals_diag.marginalCovariance(pose_keys_diag[key]))
        
    for pose in full_estimate:
        print(pose[0:2])
        gtsam_plot.plot_point2(0, pose[0:2], "c")

        
    
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    
    move = [[2, 0], [1, math.pi/4], [3, 0]]
    odoms = odom_from_r_theta(move, 0.3)
    
    
    main(odoms)

    
