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

    return np.array([x, y, angle]), G, R

######################################################
# TODO: Implement the Measurement Model 
#
# This should return the predicted measurement and
# the jacobian (H) with respect to the previous state
def h(mu_bar, markerPos): 

    return np.array(z), H


########################################################
# TODO: Implement the ekfUpdate Equations
#
# This should return the new state and covariance after
# updating for motion and the recieved measurement
def ekfUpdate(mu, Sigma, u, M, z, Q, markerId):
    # NOTE: The header is not set in stone.  You may change it if you like.
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    stateDim=3
    motionDim=3
    observationDim=1

    #######################################################
    # Prediction step
    #######################################################
    # EKF prediction of mean and covariance


    #######################################################
    # Correction step
    #######################################################

    # Compute expected observation and Jacobian

    # Innovation / residual covariance

    # Kalman gain

    # Correction


    return mu_new[:stateDim], Sigma_new[0:stateDim, 0:stateDim]
