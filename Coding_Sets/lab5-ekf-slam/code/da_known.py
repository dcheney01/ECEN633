import numpy as np
import helpers

####################################
# TODO: Implement Data Association
####################################
# Output:
#  - an "association" vector that specifies the landmark index of each measurement. (-1 means new landmark, -2 means ignore)
#  - Since MOST data associations (NN, JCBB) end up computing innovation and measurement jacobians,
#    it can save computation to pass them on from this function. 
#
# Inputs:
#  - measurements obtained at this time step
#  - R covariance matrix
#  - Curren state estimate (mu, Sigma)
#  - Landmark signatures (only available for use in da_known)
#  - Short circuit threshhold may be helpful for use in jcbb, not needed in the other algorithms
    # Short circuit threshhold is used in jcbb, not implemented here
def associateData(measurements, R, mu, Sigma, landmarkSignatures = [], shortCircuitThresh = 40.0**2):
    association = [] # Each index of this will hold the index of the landmark in the state.
    innovation = [] # Indexed the same as the above, this is used in the update step
    H = [] # Indexed the same as the above, this is used in the update step
    
    for z in measurements: # OK, for each measurement we have...
        dist, bearing, sig = z # Let's extract the components
        if sig in landmarkSignatures: # If we've seen this signature (markerID) before, it's in the state
            # Association
            ind = np.flatnonzero(np.asarray(landmarkSignatures == sig))[0] # We just need to find WHERE in the state
            association.append(ind) # And make sure we return that

            # Innovation + Jacobian
            range_hat, bearing_hat, Hi = h(mu, ind, len(landmarkSignatures))
            
            innovation.append(np.array([[dist - range_hat], [bearing-bearing_hat]]))
            H.append(Hi)

        else: # If the landmark signature (markerID) HASN'T been seen yet, we need to create a new landmark
            association.append(-1) # We say the association is "-1" to tell future code to add new landmarks
            innovation.append([0, 0]) # We won't use the innovation anyway, but we need to return SOMETHING
            H.append(np.zeros((2, len(mu)))) # Same for our H jacobian

    return np.array(association), np.array(H), np.array(innovation)

def h(mu, ind, N):

    x, y, theta = mu[:3]
    landmark_ind = 2*ind + 3
    landmark_x, landmark_y = mu[landmark_ind: landmark_ind+2]

    Delta = np.array([landmark_x-x, landmark_y-y])
    q = Delta.T @ Delta
    Sx = Delta[0]
    Sy = Delta[1]

    # range_hat and bearing_hat
    range_hat = np.sqrt(q)
    bearing_hat = helpers.minimizedAngle(np.arctan2(Sy - y, Sx - x) - theta)

    # Jacobian
    # Helper matrix that maps the low-dimensional matrix h_i
    first = np.block([[np.eye(3)], [np.zeros((2,3))]])
    second = np.block([[np.zeros((3,2))], [np.eye(2)]])
    F = np.block([first, np.zeros((5, 2*ind)), second, np.zeros((5, 2*N - 2*(ind+1)))]) 

    Hi = (1/q) * np.array([[-np.sqrt(q)*Sx, -np.sqrt(q)*Sy, 0,  np.sqrt(q)*Sx, np.sqrt(q)*Sy],
                            [Sy,             -Sx,           -q, -Sy,            Sx]]) @ F

    # print(f"F shape is: {F.shape}")
    # print(f"Hi shape is: {Hi.shape}")

    return range_hat, bearing_hat, Hi