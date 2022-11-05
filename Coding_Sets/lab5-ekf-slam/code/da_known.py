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
def associateData(measurements, R, mu, Sigma, landmarkSignatures = []):
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
            dist_hat, bearing_hat, Hi = h(mu, ind, len(landmarkSignatures))
            inno = np.array([dist - dist_hat, helpers.minimizedAngle(bearing-bearing_hat)])

            if len(innovation) < 1:
                innovation = inno
                H = Hi
            else:
                innovation = np.hstack((innovation, inno))
                H = np.vstack((H, Hi))

        else: # If the landmark signature (markerID) HASN'T been seen yet, we need to create a new landmark
            association.append(-1) # We say the association is "-1" to tell future code to add new landmarks
            if len(innovation) < 1:
                innovation = np.array([0, 0])
                H = np.zeros((2, len(mu)))
            else:
                innovation = np.hstack((innovation, np.array([0, 0]))) # We won't use the innovation anyway, but we need to return SOMETHING
                H = np.vstack((H, np.zeros((2, len(mu))))) # Same for our H jacobian

    return np.array(association), H, innovation

def h(mu, ind, N):
    x, y, theta = mu[:3]
    landmark_ind = 2*ind + 3
    landmark_x, landmark_y = mu[landmark_ind: landmark_ind+2]

    Delta = np.array([landmark_x-x, landmark_y-y])
    q = Delta.T @ Delta
    Sx = Delta[0]
    Sy = Delta[1]

    # range_hat and bearing_hat
    dist_hat = np.sqrt(q)
    bearing_hat = helpers.minimizedAngle(np.arctan2(Sy, Sx) - theta)

    # Jacobian
    # Helper matrix that maps the low-dimensional matrix h_i
    first = np.block([[np.eye(3)], [np.zeros((2,3))]])
    second = np.block([[np.zeros((3,2))], [np.eye(2)]])
    F = np.block([first, np.zeros((5, 2*ind)), second, np.zeros((5, 2*N - 2*(ind+1)))]) 

    Hi = (1/q) * np.array([[-np.sqrt(q)*Sx, -np.sqrt(q)*Sy,  0,  np.sqrt(q)*Sx, np.sqrt(q)*Sy],
                            [Sy,             -Sx,           -q, -Sy,            Sx]]) @ F

    # print(f"F shape is: {F.shape}")
    # print(f"Hi shape is: {Hi.shape}")

    return dist_hat, bearing_hat, Hi