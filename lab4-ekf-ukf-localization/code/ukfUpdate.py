#!/usr/bin/env python3
"""Module for the UKF portion of the landmark/localization lab

Defined Functions: 
UkfUpdate - the update function for the UKF
"""

from fieldSettings import field
import numpy as np
import helpers
from helpers import debugPrint
from scipy.linalg import block_diag


#################################################
# TODO: Implement the Motion Model for the robot
def g(u, mu):
    d_rot1 = u[0]
    d_trans = u[1]
    d_rot2 = u[2]

    angle = helpers.minimizedAngle(mu[2]+d_rot1)

    x_prime = mu[0] + d_trans*np.cos(angle)
    y_prime = mu[1] + d_trans*np.sin(angle)
    theta_prime = mu[2] + d_rot1+d_rot2

    return np.array([x_prime, y_prime, theta_prime]).T

######################################################
# TODO: Implement the Measurement Model
def h(mu_bar, markerPos): 
    landmark_x, landmark_y = markerPos
    z_hat = helpers.minimizedAngle(np.arctan2(landmark_y - mu_bar[:,1], landmark_x-mu_bar[:,0])- mu_bar[:,2])
    return z_hat

#######################################################
# TODO: Write a Function to Generate your sigma points
def makeSigmaPoints(mu, Sigma):
    k = 0
    beta = 0
    n = mu.shape[0]
    alpha = 1
    lamba = alpha**2 * (n+k) - n
    L = np.linalg.cholesky((n+lamba) * Sigma).T

    sigmaPoints = np.zeros((n, 2*n+1))
    sigmaPoints[:,0] = mu.flatten()
    w_m = np.zeros((2*n+1,))
    w_c = np.zeros((2*n+1,))
    w_m[0] = lamba/(n+lamba)
    w_c[0] = w_m[0] + (1 - alpha**2 + beta)

    for i in range(1, 2*n+1):
        if i < n:
            point = mu.flatten() - L[i-1]
        else:
            point = mu.flatten() + L[i-(n+1)]
        sigmaPoints[:,i] = point.flatten()
        w_m[i] = 1/(2*(n+lamba))
        w_c[i] = 1/(2*(n+lamba))

    return sigmaPoints, w_m, w_c

###########################################################
# TODO: Implement the UKF Update
#
# Make sure you use the unscented transform to
# propogate the uncertainty through both the prediction
# (Update Step) and measurement models (Correction Step).
#
# This function should return an updated mean
# and covariance for your state.
def ukfUpdate(mu, Sigma, u, M, z, Q, markerId):
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    #######################################################
    # Setup UKF steps 2-6 in Table 7.4
    #######################################################
    mu_augmented = np.vstack((mu.reshape(3,1), np.zeros((4,1))))
    sigma_augmented = block_diag(Sigma, M, Q)
    X_at1, w_m, w_c = makeSigmaPoints(mu_augmented, sigma_augmented)
    #######################################################
    # Prediction steps 7-9 in Table 7.4
    #######################################################
    # Unscented Transform Prediction of Propogated Mean and Covariance
    input_mu = X_at1[3:6, :]
    state_mu = X_at1[:3,:]
    Xbar_xt = g(u.reshape(3,1)+input_mu, state_mu)

    mu_bar = w_m.dot(Xbar_xt)
    sigma_bar = np.zeros((3,3))
    for i in range(15):
        diff = (Xbar_xt[i]-mu_bar).reshape(3,1)
        sigma_bar += w_c[i] * diff @ diff.T

    #######################################################
    # Correction step 10-18
    #######################################################
    # Unscented Transform Prediction of Measurement/Innovation Mean/Covariance
    # 10-13 in Table 7.4
    Z_bar = h(Xbar_xt, (landmark_x,landmark_y)) + X_at1[-1]
    z_hat = w_m.dot(Z_bar)

    S = 0
    for i in range(Z_bar.shape[0]):
        diff = (Z_bar[i]-z_hat)
        S += w_c[i] * diff * diff

    S = np.array([[S]])

    sigma_cross = 0
    for i in range(Z_bar.shape[0]):
        diff1 = (Xbar_xt[i]-mu_bar).reshape((3,1))
        diff2 = (Z_bar[i] - z_hat)
        sigma_cross += w_c[i] * diff1 * diff2


    # UKF Correction Equations 14-16 in Table 7.4
    K = sigma_cross @ np.linalg.inv(S)
    mu = mu_bar + (K * (z - z_hat)).flatten()
    mu[2] = helpers.minimizedAngle(mu[2])
    sigma = sigma_bar - (K @ S @ K.T)

    return mu, sigma
