"""
======================================================
Plot a confidence ellipse of a two-dimensional dataset
======================================================

This example shows how to plot a confidence ellipse of a
two-dimensional dataset, using its pearson correlation coefficient.

The approach that is used to obtain the correct geometry is
explained and proved here:

https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

The method avoids the use of an iterative eigen decomposition algorithm
and makes use of the fact that a normalized covariance matrix (composed of
pearson correlation coefficients and ones) is particularly easy to handle.
"""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import math

# calculate odometry from given r, theta values and b
def odom_from_r_theta(given, b):

    if given[1] > 0:
        sl = abs(given[1]) * given[0]
        sr = abs(given[1]) * (given[0] + b)

    else:
        sl = abs(given[1]) * (given[0] + b)
        sr = abs(given[1]) * given[0]

    return sr, sl

# calculate x and y position from given odometry measurments
def odom_calculation_xy(del_sr, del_sl, b):

    del_x = (del_sr + del_sl)/2 * math.cos((del_sr - del_sl)/(2*b))
    del_y = (del_sr + del_sl)/2 * math.sin((del_sr - del_sl)/(2*b))

    change = (del_x, del_y)

    return change

# caculate Jacobian G from given paramters
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


# caculate matrix Sigma yy from G * Sigma_xx * G^T
def Sigma_yy(del_sr, del_sl, b, k_r, k_l):

    Sigma_xx = np.array([[k_r * abs(del_sr), 0], [0, k_l * abs(del_sl)]])

    G = jacobian_G(del_sr, del_sl, b)

    interim = np.matmul(G, Sigma_xx)

    final = np.matmul(interim, np.transpose(G))

    return final

# %%
#
# The plotting function itself
# """"""""""""""""""""""""""""
#
# This function plots the confidence ellipse of the covariance of the given
# array-like variables x and y. The ellipse is plotted into the given
# Axes object *ax*.
#
# The radiuses of the ellipse can be controlled by n_std which is the number
# of standard deviations. The default value is 3 which makes the ellipse
# enclose 98.9% of the points if the data is normally distributed
# like in these examples (3 standard deviations in 1-D contain 99.7%
# of the data, which is 98.9% of the data in 2-D).


def confidence_ellipse(x, y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    cov : np matrix
    covariance matrix for given x, y data set

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.

    scale_x = np.sqrt(cov[0, 0]) * n_std    # this line he described to me so it's good they already did it
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# use these r, theta values to create a movement
givens = [2, math.pi/4]

# calculate the true odometry movements and poses without noise corruption
truth_sr, truth_sl = odom_from_r_theta(givens, 0.3)
truth_x, truth_y = odom_calculation_xy(truth_sr, truth_sl, 0.3)

# these values are used in the covariance matrix calculation - should eventually be replaced with real values from physical testing
k_l = 0.005
k_r = 0.005

# calculate standard deviation
std_dev_l = math.sqrt(k_l* abs(truth_sl))
std_dev_r = math.sqrt(k_r* abs(truth_sr))

sig_yy = Sigma_yy(truth_sr, truth_sl, 0.3, k_r, k_l)

# we only need the top left corner of the 3x3 matrix calculated
small_sig_yy = sig_yy[0:2, 0:2]


# store calculated samples
x_list = []
y_list = []

# create dataset by generating noise, adding to ground truth and then calculating the resulting values for x and y
for i in range(1000):
    noise_sr = std_dev_r * np.random.randn()
    noise_sl = std_dev_l * np.random.randn()

    x_samp, y_samp = odom_calculation_xy(truth_sr + noise_sr, truth_sl + noise_sl, 0.3)
    
    x_list.append(x_samp)
    y_list.append(y_samp)

x = np.array(x_list)
y = np.array(y_list)



# A plot with n_std = 3 (blue), 2 (purple) and 1 (red)

fig, ax_nstd = plt.subplots(figsize=(6, 6))

mu = truth_x, truth_y # set mean of the data set to the ground truth values
scale = 8, 5

ax_nstd.axvline(c='grey', lw=1)
ax_nstd.axhline(c='grey', lw=1)

ax_nstd.scatter(x, y, s=0.5) 

confidence_ellipse(x, y, small_sig_yy, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
confidence_ellipse(x, y, small_sig_yy, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
confidence_ellipse(x, y, small_sig_yy, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
ax_nstd.set_title('Different standard deviations')
ax_nstd.legend()
plt.show()

