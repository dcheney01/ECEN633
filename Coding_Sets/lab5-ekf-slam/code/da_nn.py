import numpy as np
import helpers

# Nearest neighbor is in terms of mahalanobis distance

# Lower than this, we associate.  Higher is a new landmark.
startingThresh = 15

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
#  - Current state estimate (mu, Sigma)
#  - Landmark signatures (only available for use in da_known)
#  - Short circuit threshhold may be helpful for use in jcbb, not needed in the other algorithms
def associateData(measurements, R, mu, Sigma, landmarkSignatures = []):
    association = [] # Each index of this will hold the index of the landmark in the state.
    innovation = [] # Indexed the same as the above, this is used in the update step
    H = [] # Indexed the same as the above, this is used in the update step
    N = (len(mu)-3)//2

    # If there are no previous landmarks
    if N < 1: # If there are no landmarks currently, add a new one
            association.append(-1) # We say the association is "-1" to tell future code to add new landmarks
            if len(innovation) < 1:
                innovation = np.array([0, 0])
                H = np.zeros((2, len(mu)))
            else:
                innovation = np.hstack((innovation, np.array([0, 0]))) # We won't use the innovation anyway, but we need to return SOMETHING
                H = np.vstack((H, np.zeros((2, len(mu)))))             # Same for our H jacobian
    # If there are previous landmarks
    else: 
        # Nearest Neighbor Algorithm
        for z in measurements:   # For each measurement
            dist, bearing, _ = z # Let's extract the components
            
            K = []
            intermediate_inno = []
            intermediate_H = []
            # Find K for each measurement -> landmark pair
            for landmark_ind in range(0, N):
                dist_hat, bearing_hat, Hi = h(mu, landmark_ind, N)
                inno = np.array([dist - dist_hat, helpers.minimizedAngle(bearing-bearing_hat)])
                intermediate_inno.append(inno)
                intermediate_H.append(Hi)
                K.append(K2(inno, Sigma, Hi))
    
            # And associate the measurement to the landmark that gives the smallest K 
            #   unless min_K > start_thresh, then we create a new landmark
            landmark_minK_ind = K.index(min(K))
            if K[landmark_minK_ind] <= startingThresh:
                # Association
                association.append(landmark_minK_ind) # And make sure we return that
                # Innovation + Jacobian
                if len(innovation) < 1:
                    innovation = intermediate_inno[landmark_minK_ind]
                    H = intermediate_H[landmark_minK_ind]
                else:
                    innovation = np.hstack((innovation, intermediate_inno[landmark_minK_ind]))
                    H = np.vstack((H, intermediate_H[landmark_minK_ind]))


            else: # If the landmark HASN'T been seen yet, we need to create a new landmark
                association.append(-1) # We say the association is "-1" to tell future code to add new landmarks
                if len(innovation) < 1:
                    innovation = np.array([0, 0])
                    H = np.zeros((2, len(mu)))
                else:
                    innovation = np.hstack((innovation, np.array([0, 0]))) # We won't use the innovation anyway, but we need to return SOMETHING
                    H = np.vstack((H, np.zeros((2, len(mu))))) # Same for our H jacobian

    return np.array(association), H, innovation

def K2(innovation, Sigma, Hi):
    return innovation.T @ np.linalg.inv(Hi @ Sigma @ Hi.T) @ innovation


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

    return dist_hat, bearing_hat, Hi