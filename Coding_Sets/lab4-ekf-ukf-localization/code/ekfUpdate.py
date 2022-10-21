#!/usr/bin/env python3
"""Module for the EKF portion of the landmark/localization lab

Defined Functions: 
ekfUpdate - the update function for the EKF
"""

from fieldSettings import field
import helpers
import numpy as np

#################################################
# TODO: Implement the Motion Model for the robot
#
# This should output a vector with an updated pose
# and two Jacobians - one (R) w/respect to the motion
# commands and one (G) w/respect to the previous state.
def g(motion, mu):
    d_rot1 = motion[0]
    d_trans = motion[1]
    d_rot2 = motion[2]

    angle = helpers.minimizedAngle(mu.item(2)+d_rot1)

    x_prime = mu.item(0) + d_trans*np.cos(angle)
    y_prime = mu.item(1) + d_trans*np.sin(angle)
    theta_prime = mu.item(2) + d_rot1+d_rot2

    G = np.array([[1, 0, -d_trans*np.sin(angle)], 
                  [0, 1, d_trans*np.cos(angle)],
                  [0, 0, 1]])

    R = np.array([[d_trans*-np.sin(angle), np.cos(angle), 0], 
                  [d_trans*np.cos(angle), np.sin(angle), 0],
                  [1, 0, 1]])

    return np.array([x_prime, y_prime, theta_prime]), G, R

######################################################
# TODO: Implement the Measurement Model 
#
# This should return the predicted measurement and
# the jacobian (H) with respect to the previous state
def h(mu_bar, markerPos): 
    landmark_x, landmark_y = markerPos
    q = (landmark_x - mu_bar[0])**2 + (landmark_y - mu_bar[1])**2
    z_hat = helpers.minimizedAngle(np.arctan2(landmark_y - mu_bar[1], landmark_x-mu_bar[0])- mu_bar[2])
    H = np.array([[(landmark_y - mu_bar[1])/q],
                    [-(landmark_x-mu_bar[0])/q],
                    [-1]]).T
    return z_hat, H

########################################################
# TODO: Implement the ekfUpdate Equations
#
# This should return the new state and covariance after
# updating for motion and the recieved measurement
def ekfUpdate(mu_t1, Sigma_t1, u, M, z, Q, markerId):
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    #######################################################
    # Prediction step
    #######################################################

    # EKF prediction of mean and covariance
    mu_bar, G, R = g(u, mu_t1)
    sigma_bar = (G @ Sigma_t1 @ G.T) + (R @ M @ R.T)

    #######################################################
    # Correction step
    #######################################################

    # Compute expected observation and Jacobian
    z_hat, H = h(mu_bar, (landmark_x, landmark_y))
    # Innovation / residual covariance
    S = H @ sigma_bar @ H.T + Q
    # Kalman gain
    K = sigma_bar @ H.T @ np.linalg.inv(S)
    # Correction
    mu = mu_bar + (K @ (z - z_hat)).T
    sigma = (np.eye(3) - K @ H) @ sigma_bar

    mu[:,2] = helpers.minimizedAngle(mu[:,2])

    #######################################################
    return mu.flatten(), sigma