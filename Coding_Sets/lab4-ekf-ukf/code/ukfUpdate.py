#!/usr/bin/env python3
"""Module for the UKF portion of the landmark/localization lab

Defined Functions: 
UkfUpdate - the update function for the UKF
"""

from fieldSettings import field
import numpy as np
import helpers
from helpers import debugPrint

#################################################
# TODO: Implement the Motion Model for the robot
def g(mu, u):

    return np.array([x, y, t])

######################################################
# TODO: Implement the Measurement Model 
# NOTE: The header is not set in stone.  You may change it if you like.
def h(mu_bar, markerLocation): #Header Option 1

    return angle

#######################################################
# TODO: Write a Function to Generate your sigma points
def makeSigmaPoints(mu, Sigma):
    
    return sigmaPoints, weights

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
    # NOTE: The header is not set in stone.  You may change it if you like.
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    stateDim=3
    motionDim=3
    observationDim=1

    #######################################################
    # Setup UKF
    #######################################################

    #######################################################
    # Prediction step
    #######################################################
    
    # Unscented Transform Prediction of Propogated Mean and Covariance

    #######################################################
    # Correction step
    #######################################################

    #Unscented Transform Prediction of Measurement/Innovation Mean/Covariance
    
    # UKF Correction Equations

    return mu_new, Sigma_new
