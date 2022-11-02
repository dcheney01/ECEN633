import numpy as np
from py import process
from generator import generate
import time
from fieldSettings import field
import helpers
import matplotlib

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# this might be necessary on MacOS for live plotting, dunno what it'll do on other systems
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
dpi = 147
plt.rcParams["figure.figsize"] = (4 * dpi/80, 4 * dpi/80) # size of plots, set to 8x8 inches (default 80ppi, so might be be the same on a given display)

# Can change this to true if you need some debugs in your code, I did it on specific timesteps where I had issues
helpers.printDebug = False 
from helpers import debugPrint

def run(numSteps, dataAssociation, updateMethod, pauseLen, makeGif):

    # Video/GIF generation
    printNumLength = len(str(numSteps)) # Used to name gif images properly
    if (makeGif): # If we're going to be making a gif/video of the output,
        from PIL import Image # we need PIL to do so

    # initialize and settings
    initialStateMean = np.array([180, 50, 0]) # initial state, [x, y, theta]
    maxObs = 2 # The maximum number of features that will be observed at any timestep
    alphas = np.array([0.05, 0.001, 0.05, 0.01])**2 # These have to do with the error in our motion controls
    betas = np.array([10, 10*np.pi/180]) # Error in observations
    R = np.diag(betas**2)
    deltaT = 0.1 # Time between each timestep, must be < 1.0s
    
    # NOTE: The below will pull a previously-generated dataset if one exists for the format you've selected.  If this is undesireable, call with "forceNew=True"
    data = generate(initialStateMean, numSteps, alphas, betas, deltaT, maxObs, forceNew=True) # there are other options in this function def for what landmarks are observed, we won't mess with them

    # For ease of the algorithm below, we'll just define our "dataAssociation" function now by importing
    if (dataAssociation == "known"): # Known is only for the simulator
        from da_known import associateData
    elif (dataAssociation == "nn"): # Nearest neighbor here is in terms of mahalanobis distance, not euclidean
        from da_nn import associateData
    elif (dataAssociation == "nndg"): # Nearest neighbor double gate adds a "no mans land" of ingored observations
        from da_nndg import associateData
    elif (dataAssociation == "jcbb"): # joint compatibility considers all the measurements at once
        from da_jcbb import associateData
    else:
        raise Exception('Unknown data association method.')

    #===================================================
    # Initialize State
    #===================================================
    realRobot = initialStateMean.copy() # This will be the variable we propagate
    realCov = 1e-03*np.eye(3) # start covariance low but non-zero
    landmarkSignatures = [] # This is to track the signature (markerID for known association) of landmarks.  Only used for da_known.
    # Track Landmark Associations for plotting later
    landmarkAssociationHistory = np.zeros((25, 26))

    muHist = np.zeros((numSteps, 2)) # Tracking the position of the robot over time for plotting

    plt.ion() # Interacting plotting so we can SEE it doing its thing
    for t in range(numSteps): # We'll run our algorithm for each step of data
        print("Running step ", t, " currently have ", int((len(realRobot)-3)/2), " landmarks ") # Not necessary, just nice to see.
            
        #=================================================
        # Data available to your filter at this time step
        #=================================================
        u = data[t, 0:3] # [d_rot_1, d_trans, d_rot_2]
        temp = data[t, 9:] # [id or -1 if no obs, noisy dist, noisy theta, noise-free dist, noise-free theta, repeat....]
        z = [] # We need to extract measurements.  Simulated data has a "-1" for id if there isn't an observation in that slot for this timestep.
        for i in range(int(len(temp)/5)): # So we need to check each
            if (temp[i*5] != -1): # and see if it's real or not
                z.append(temp[i*5:i*5+3]) # Then grab the first 3 values, since they have noise - last 2 are noise-free, not allowed in our simulator.
        z = np.array(z) # [[id, dist, theta], [id, dist theta], etc]
        order = np.array([1, 2, 0]) # But we want order "dist, theta, id" to match VP.
        z = ((z.T)[order]).T # Reorder, and now our measurements are ready to go.

        #=================================================
        # Update based on Robot Motion
        #=================================================
        d_rot1 = u[0]
        d_trans = u[1]
        d_rot2 = u[2]
        a1=alphas[0]
        a2=alphas[1]
        a3=alphas[2]
        a4=alphas[3]
        M = np.array([[a1*d_rot1**2 + a2 * d_trans**2,0,0],
                      [0, a3*d_trans**2 + a4 * (d_rot1**2 + d_rot2**2), 0],
                      [0, 0, a1*d_rot2**2 + a2*d_trans**2]])
        
        # Call your predict function
        # TODO: Implement the predict function defined below
        print(f"Robot State: {realRobot}")
        print(f"Command: {u}")
        realRobot_bar, realCov_bar = predict(realRobot, realCov, u, M)
        print(f"robot state after predict: {realRobot_bar}")

        #=================================================
        # Update based on Measurements
        #=================================================
        if (len(z) > 0): # If we HAVE any measurements...
            # Call the associateData function to determine which measurements correspond to which landmarks
            # TODO: Implement each of the associateData functions in the da_----.py files
            association, H, innovation = associateData(z, R, realRobot_bar, realCov_bar, landmarkSignatures) 
            
            # Save a history of associations so you can plot them later
            # print(f"association: {association}")
            for index in range(len(association)):
                trueId = int(z[index][2])
                associatedTo = int(association[index] + 1)
                landmarkAssociationHistory[trueId][associatedTo] = landmarkAssociationHistory[trueId][associatedTo] + 1

            #########################################################
            # TODO: Here you should call the update and augmentState functions
            # Commented out until implemented
            
            # Update your state for landmarks already observed
            realRobot, realCov = update(realRobot_bar, realCov_bar, association, H, R, innovation, z, updateMethod) 
            # print(f"robot state after update: {realRobot}")
            # print(f"robot covariance after update:\n{realCov}")

            # Augment our state with new landmarks that were not associated
            realRobot, realCov, landmarkSignatures = augmentState(association, z, realRobot, realCov, R, landmarkSignatures)
            # print()
            # print(f"robot state after augment: {realRobot}")
            # print(f"robot covariance after augment:\n{realCov}")

            # Comment this out after you implement and un comment the above two function calls
            # realRobot = realRobot_bar
            # realCov = realCov_bar
            
        #=================================================
        #TODO: plot and evaluate filter results here
        #=================================================
        muHist[t] = realRobot[:2] # Track the position of the robot over time
    
        plotsim(data, t, initialStateMean, realRobot, realCov, muHist); # Plot the state as it is after the timestep
        plt.legend()        
        plt.gcf().canvas.draw() # Tell the canvas to draw, interactive mode is weird
        if pauseLen > 0: # If we've got a pauselen, let's take a break so it doesn't blur past
            time.sleep(pauseLen)

        # Save GIV Data
        if (makeGif): # If we're saving to make a video, let's put the current frame into a saved image for later processing.
            imgData = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8) # Extra image data from the plot
            w, h = plt.gcf().canvas.get_width_height() # Determine the dimensions
            mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things, account for it.
            im = imgData.reshape((int(h*mod), int(w*mod), -1)) # Create our image array in the right shape
            Image.fromarray(im).save("outputGif/sim_" + str(t).zfill(printNumLength) + ".png") # And pass it to PIL to save it.

        # Make sure the canvas is ready to go for the next step
        plt.gcf().canvas.flush_events() 

        
        
    # END of Loop    
    plt.ioff() # Once we've done all time steps, turn off interactive mode
    print("Finished execution") # Nice to know when we're done since sometimes the data moves slowly
    np.savez("nnAssociation.npz", associationHistory=landmarkAssociationHistory)

    # GIF Plotting
    if (makeGif):
        import glob # Need this to load photos (better than holding them in RAM forever)
        # Create the frames
        frames = [] # This holds each image
        imgs = glob.glob("outputGif/sim_*.png") # load 'em in
        imgs.sort() # Make sure they're in the right order
        for i in imgs: # For each one, we'll open and append
            new_frame = Image.open(i)
            frames.append(new_frame)
        
        # Save into a GIF file that loops forever
        frames[0].save('outputGif/output.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=len(imgs)*0.05, loop=1)
    plt.show() # And just show the last image
    
#==========================================================================
# Pretty drawing stuff
#==========================================================================
def plotsim(data, t, initialStateMean, mu, Sigma, muHist):

    noiseFreePathColor = '#00FF00'
    noisyPathColor = '#0000FF'
    estimatedPathColor = '#FFBB00'

    noiseFreeBearingColor = '#00FFFF'
    observedBearingColor = '#FF0000'
        
    #=================================================
    # data *not* available to your filter, i.e., known
    # only by the simulator, useful for making error plots
    #=================================================
    # actual position (i.e., ground truth)
    x, y, theta = data[t, 3:6]
    
    # real observation
    z = data[t, 9:] # [id or -1 if no obs, noisy dist, noisy theta, noise-free dist, noise-free theta, repeat....]
    obs = [] # We need to gather together the observations that are non-null
    for i in range(int(len(z)/5)):
        if (not (z[i*5] == -1)):
            obs.append(z[i*5:i*5+5])
    obs = np.array(obs)

    #################################################
    # Graphics
    #################################################
    plt.clf() # clear the frame.
    helpers.plotField(obs[:,0]) # Plot the field with the observed landmarks highlighted

    # draw actual path and path that would result if there was no noise in
    # executing the motion command
    plt.plot(np.array([initialStateMean[0], *data[:t, 6]]), np.array([initialStateMean[1], *data[:t, 7]]), color=noiseFreePathColor, label='Noise Free Path')
    plt.plot(data[t, 6], data[t, 7], '*', color=noiseFreePathColor)

    # draw the path that has resulted from the movement with noise
    plt.plot(np.array([initialStateMean[0], *data[:t, 3]]), np.array([initialStateMean[1], *data[:t, 4]]), color=noisyPathColor, label='True Noisy Path')
    helpers.plotRobot(data[t, 3:6], "black", "#00FFFF40")

    # draw the path the estimated robot followed
    plt.plot(np.array([initialStateMean[0], *muHist[:t, 0]]), np.array([initialStateMean[1], *muHist[:t, 1]]), color=estimatedPathColor, label='EKF SLAM Est')
    plt.plot([mu[0]], [mu[1]], '*', color=estimatedPathColor)
    helpers.plotCov2D(mu[:2], Sigma[:2, :2], color=estimatedPathColor, nSigma=3)

    for observation in obs:
        # indicate observed angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + observation[2])*observation[1]]), np.array([y, y+np.sin(theta + observation[2])*observation[1]]), color=observedBearingColor)

        # indicate ideal noise-free angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + observation[4])*observation[3]]), np.array([y, y+np.sin(theta + observation[4])*observation[3]]), color=noiseFreeBearingColor)
    
    for i in range(int((len(mu) - 3)/2)):
        # We'll also plot a covariance ellipse for each landmark
        helpers.plotCov2D(mu[3+i*2:3+i*2+2], Sigma[3+i*2:3+i*2+2, 3+i*2:3+i*2+2], nSigma=3)

##################################################
# TODO: Implement the Motion Prediction Equations
##################################################
def predict(mu, Sigma, u, M):
    # print(f"inital mu: {mu}")
    # print(f"initial cov: \n{Sigma}")
    # print(f"initial u: {u}")

    # Propogate Jacobian
    N = (mu.shape[0]-3)//2 # Get number of known landmarks
    d_rot1 = u[0]
    d_trans = u[1]
    d_rot2 = u[2]
    angle = helpers.minimizedAngle(mu[2]+d_rot1)

    # Find Jacobians
    # F is a matrix that maps the 3-D state vector into a vector that is 3N+3 where N is the number of landmarks
    F = np.block([np.eye(3), np.zeros((3,2*N))])

    # G is the derivative of the motion model with respect to the previous state and current u
    G =  np.eye(3 + 2*N) + F.T @ np.array([[1, 0, -d_trans*np.sin(angle)], 
                                           [0, 1, d_trans*np.cos(angle)],
                                           [0, 0, 1]]) @ F

    # print(f"F: \n{F}\n")
    # print(f"G: \n{G}\n")
    # print(f"M: \n{M}\n")

    # Change in position
    x_prime = d_trans*np.cos(angle)
    y_prime = d_trans*np.sin(angle)
    theta_prime = d_rot1+d_rot2

    # Update Mean and Covariance (Handle both Robot State and Landmarks)
    mu_bar = mu + F.T @ np.array([x_prime, y_prime, theta_prime])
    Sigma_bar = (G @ Sigma @ G.T) + (F.T @ M @ F)

    # print()
    # print(f"final mu: {mu_bar}")
    # print(f"final cov: \n{Sigma_bar}")
    # assert False
    
    return mu_bar, Sigma_bar # And give them back to the calling function

###############################################
# TODO: Implement Measurement Update Equations
###############################################
def update(mu_bar, Sigma_bar, association, H, Q, innovation, z, updateMethod):
    ind = np.flatnonzero(np.asarray(association > -1)) # -1 is used as a keyword for "new landmark," -2 is used for "ignore"
    if (len(ind) == 0): # If we don't have any, short-circuit return
        # print("short-circuit return")
        return mu_bar, Sigma_bar

    if (updateMethod == "seq"): 
        print("seq method not implemented")
        for i in ind: 
            # TODO: Finish This to update incrementally for each measurement
            pass # To enable running until implemented
    elif (updateMethod == "batch"): 
        # TODO: Finish This to update all measurements of a time step at once
        H = np.vstack((H[0], H[1]))
        Q = np.block([[Q, np.zeros(Q.shape)],
                      [np.zeros(Q.shape), Q]])
        z_hat = innovation.squeeze()

        print(f"H shape is: {H.shape}")
        print(f"Q shape is: {Q.shape}")
        print(f"Sigma_bar shape is: {Sigma_bar.shape}")
        print(f"z_hat shape is: {z_hat.shape}")

        S = H @ Sigma_bar @ H.T + Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        print(f"S shape should be 4x4: {S.shape}")
        print(f"K shape should be {3+len(association)*2}x{2*z.shape[0]}: {K.shape}")

        mu = mu_bar + K @ (z_hat - z[:, :2])
        Sigma = (np.eye((3,3)) - K @ H) @ Sigma_bar
    else:
        raise Exception("Unknown update method, '" + updateMethod + "' - it must be 'seq' or 'batch'")
    
    return mu, Sigma

##################################

# takes measurement + robots pose and gives landmark position
def g(dist, bearing, x, y, theta):
    l_x = dist * np.cos(bearing + theta) + x
    l_y = dist * np.sin(bearing + theta) + y
    return np.array([l_x, l_y])

# TODO: Implement Augment State
##################################
def augmentState(association, measurements, mu, Sigma, Q, landmarkSignatures):
    indices = np.flatnonzero(np.asarray(association == -1)) # If our data association returned a "-1" for a measurement, it's a landmark to add to the state
    measurements = measurements[indices] # We want to filter out only the measurements we cared about
    x, y, theta = mu[:3] # We'll need the robot state
    Sigma_rr = Sigma[:3, :3] # And the main robot covariance, we won't change

    # For each measurement of a new landmark update your state
    for z in measurements:
        # print()
        # print("AUGMENTING STATE WITH ANOTHER LANDMARK")
        N = (mu.shape[0] - 3) // 2 # Number of landmarks currently in the state

        # Extract info from the measurement
        dist, bearing, sig = z
        # Update the signatures so we know what landmark index goes to what signature.  Only used for da_known
        landmarkSignatures = np.array([*landmarkSignatures, sig])

        # Update both the mean (mu_l) and 
        # covariance (Sigma_lr, Sigma_rl, Sigma_Ll, Sigma_lL, Sigma_ll) for the new landmark.
        mu = np.concatenate((mu, g(dist, bearing, x, y, theta)))

        G_r = np.array([[1, 0, -dist*np.sin(bearing + theta)], 
                        [0, 1,  dist*np.cos(bearing + theta)]])
        G_delta = np.array([[1, 0],
                            [0, 1]])

        Sigma_lr = G_r @ Sigma_rr
        Sigma_ll = G_r @ Sigma_rr @ G_r.T + G_delta @ Q @ G_delta.T
        # print(f"Sigma_lr shape should be 2x3: {Sigma_lr.shape}")
        # print(f"Sigma_ll shape should be 2x2: {Sigma_ll.shape}")

        if N < 1:
            # print(f"No Previous Landmarks, N = {N}")
            Sigma = np.block([[Sigma,      Sigma_lr.T],
                              [Sigma_lr,   Sigma_ll]])
            # print(f"Sigma shape should be {3+2*(N+1)}x{3+2*(N+1)}: {Sigma.shape}")

        else:
            Sigma_rL = Sigma[:3, 3:]
            Sigma_lL = G_r @ Sigma_rL

            # print(f"Sigma_rL shape should be 3x{2*(N)}: {Sigma_rL.shape}")
            # print(f"Sigma_lL shape should be 2x{2*(N)}: {Sigma_lL.shape}")

            # Sigma = np.block([[Sigma,      Sigma_rL, Sigma_lr.T],
            #                   [Sigma_rL.T, Sigma_LL, Sigma_lL.T],
            #                   [Sigma_lr,   Sigma_lL, Sigma_ll]])

            Sigma_new = np.zeros((3+2*(N+1), 3+2*(N+1)))
            Sigma_new[:(3+2*N), :(3+2*N)] = Sigma

            Sigma_new[-2:, :3] = Sigma_lr
            Sigma_new[:3, -2:] = Sigma_lr.T

            Sigma_new[-2:, 3:3+2*N] = Sigma_lL
            Sigma_new[3:3+2*N, -2:] = Sigma_lL.T

            Sigma_new[-2:, -2:] = Sigma_ll
            Sigma = Sigma_new

            # print(f"Sigma shape should be {3+2*(N+1)} x {3+2*(N+1)}: {Sigma.shape}")

    return mu, Sigma, landmarkSignatures
