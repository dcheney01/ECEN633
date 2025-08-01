import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import helpers
from helpers import debugPrint
from scipy.linalg import block_diag

import matplotlib
matplotlib.use("Qt5Agg") # this might be necessary on MacOS for live plotting, dunno what it'll do on other systems
import matplotlib.pyplot as plt
dpi = 147
plt.rcParams["figure.figsize"] = (4 * dpi/80, 4 * dpi/80) # size of plots, set to 8x8 inches (default 80ppi, so might be be the same on a given display)

from detectTrees import detectTrees

laserThresh = 25.0 # This is a laser threshold, it's used to discard some of the measurements - I found the algorithm was MUCH more successful when I did so.

def run(numSteps, dataAssociation, updateMethod, pauseLen, make_gif):
    if (make_gif): # If we're going to be making a gif/video of the output,
        from PIL import Image # we need PIL to do so

    #===================================================        
    # Load dataset
    #===================================================
    vpData = np.load("../data/vpData.npz")
    # gps - [time, la, lo]
    # laser - [time, readings]
    # control - [time, speed, steering]
    globals().update(vpData) # this turns vpData['control'] into a global labeled 'control', etc...

    # For ease of the algorithm below, we'll just define our "dataAssociation" function now by importing
    if (dataAssociation == "known"):
        from da_known import associateData
    elif (dataAssociation == "nn"):
        from da_nn import associateData
    elif (dataAssociation == "nndg"):
        from da_nndg import associateData
        
    else:
        raise Exception('Unknown data association method.')

    ####################################################
    # Initalize Params
    ####################################################
    # 2x2 process noise on control input
    Qu = np.diag(np.array([0.02, 2*np.pi/180])**2) # vc, then alpha # Default values
    
    # 3x3 process noise on model error
    Qf = np.diag(np.array([0.1, 0.1, 0.5*np.pi/180])**2) # [x (m), y (m), phi (deg/rad)] # Default values

    # 2x2 observation noise
    R = np.diag(np.array([0.06, 1.2*np.pi/180])**2) # [r (m), beta (deg/rad)] # Default values
    #===================================================
    
    # Initialize State
    #===================================================
    mu = np.array([*gps[2, 1:], 36*np.pi/180]) # We'll start the robot at a gps position from the data - and we know the bearing is roughly 36 degrees north of east.
    Sigma = np.zeros((3, 3)) # Our initial covariance is 0.  Could probably start at a small diagonal instead, but this works.
    
    controlIndex = 0 # control index - our data is structed differently than the simulator, we're stepping through times and comparing when we have measurements
    t = min(laser[0, 0], control[0, 0]) # our start time is not 0, so we start at the min of our control and our measurements

    toRun = min(numSteps, len(laser)) # The number of steps we're running depends on our control value, but if we want to run them all we pass "np.inf" and we don't want to run forever...
    robotPath = np.zeros((2, toRun+1)) # We'll track where the robot's been
    robotPath[:,0] = mu[:2] # and we know where we start

    plt.ion() # interactive plotting so we can see it as it goes.  Might make sense to add an option to turn this off for speed, but... I'm not gunna.  Wanna see if it diverges.
    gifFrames = []

    print("Running Victoria Park data for until:", toRun) # Just a nice printout so we know what to expect

    ##############################################
    # Main Iteration Loop
    ##############################################
    prediction_time = []
    update_time = []
    num_landmarks = []
    for k in range(toRun): # Now we begin running through the data
        # print("Running step ", k, " currently have ", int((len(mu)-3)/2), " landmarks ") # nice to know when watching, good for estimating calc duration
        didUpdate = False
        prediction_tmp = 0
        while (control[controlIndex, 0] < laser[k, 0]): # As long as the time for the "control" data is less than the next step of "observation" data...
            dt = control[controlIndex, 0] - t # find how long since the last control
            t = control[controlIndex, 0] # update the "last control" variable so we can do that again next time
            u = control[controlIndex, 1:] # and extract the control information

            ##################################
            # EKF Prediction Step
            ##################################
            # TODO: Implement this function below
            start = time.time()
            mu, Sigma = predict(mu, Sigma, u, dt, Qu, Qf) # predict our movement based on our model (function is in this file)
            end = time.time()
            prediction_tmp += end - start
            controlIndex += 1 # and update our "control index" so we know what data we've already looked at and processed
        
        prediction_time.append(prediction_tmp)

            
        robotPath[:,k+1] = mu[:2] # track our robot path based on the controls we've received to this point

        ###############################
        # Handle Measurements
        ###############################
        # If we're here, it means the laser measurement took place between the last control and the next, so we'll process it here.
        # Technically it didn't occur right at this moment, but it should be close enough.
        dt = laser[k, 0] - t # Find the time since the last command (motion command, probably, but not necissarily)
        t = laser[k, 0] # Set the current moment in time so we can figure out what to do next
        z = detectTrees(laser[k, 1:]) # And extract our data - detectTrees is a nighmare of a function.
        z[:,1] -= np.pi/2 # Relative to the simulator, a bearing of "0" from a measurement ACTUALLY means "-90 degrees relative to the bearing of the vehicle" rather than "0", so we need to correct.
        ind = np.flatnonzero(np.asarray(z[:,0] < laserThresh)) # We'll use ONLY the measurements where the range is sufficiently low - don't trust the farther ones because of nonlinearities
        z = z[ind] # Extract only those measurements

        ##########################
        # Call Data Association
        ##########################
        association, H, innovation = associateData(z, R, mu, Sigma) # Get our association - and data for the update
        # In the above, 'short circuit threshold' says 'don't bother trying to associate to landmarks further away from the robot than this' since they're very unlikely.

        ###########################
        # Measurement Update
        ###########################
        # TODO: Implement this function below and uncomment this line
        start = time.time()
        mu, Sigma = update(mu, Sigma, association, H, R, innovation, updateMethod) # Update based on our association - correction step
        end = time.time()
        update_time.append(end - start)
        num_landmarks.append((len(mu)-3)/2)
        # TODO: Implement this function below and uncomment this line
        mu, Sigma = augmentState(association, z, mu, Sigma, R) # and if we have any measurements that represent new landmarks, add those into the state


        ############################
        # Plotting and Graphics
        ############################
        graphics(mu, Sigma, z, robotPath[:, :k+1], boundingBox = np.array([-204.37032674363175, 82.57932226717286, -126.34121202791962, 206.2942328072021])) # Handle any graphics (keep minimal for speed, 7249 timesteps in the whole dataset)
        plt.gcf().canvas.draw() # interactive plotting is weird, but force it to execute here
        if pauseLen > 0: # If we have a pauselen (don't recommend)
            time.sleep(pauseLen) # we'll pause here so the frames don't blur by too fast (NOT LIKELY)
        if (make_gif):
            imgData = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            w, h = plt.gcf().canvas.get_width_height()
            mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things.
            im = imgData.reshape((int(h*mod), int(w*mod), -1))
            gifFrames.append(Image.fromarray(im))

        plt.gcf().canvas.flush_events() # Make sure the canvas is ready to go for the next step



    ######################
    # End of Data Loop
    ######################
    plt.ioff() # Once we've done all time steps, turn off interactive mode
    print("Finished execution") # Nice to know when we're done since sometimes the data moves slowly
    plt.show(block=False) # And just show the last image

    if (make_gif):
        # Save into a GIF file that loops forever
        gifFrames[0].save('video_task3.gif', format='GIF',
        append_images=gifFrames[1:],
        save_all=True,
        duration=numSteps*2*0.1, loop=1)

    fig, ax = plt.subplots()
    ax.scatter(num_landmarks, prediction_time, label="Prediction CPU Time")
    ax.scatter(num_landmarks, update_time, label="Update CPU Time")
    ax.legend(loc="upper right")

    plt.show()


###########################
# Graphics and Plotting
###########################
def graphics(mu, Sigma, z, robotPath, boundingBox = None):
    plt.clf() # clear the frame.
    # restrict view to a bounding box around the current pose
    if boundingBox is not None:
        boundingBox = np.array([boundingBox])
        boundingBox = boundingBox.reshape(-1)
        if (len(boundingBox) == 1):
            plt.axis([*([-boundingBox,boundingBox]+mu[0]), *([-boundingBox,boundingBox]+mu[1])])
        else:
            plt.axis(boundingBox)
    plt.gca().set_aspect('equal', adjustable='box') # Make sure the aspect ratio in the plot is 1:1
    
    # Put whatever graphics you want here for visualization
    # WARNING: this slows down your process time, so use sparingly when trying
    # to crunch the whole data set!

    estimatedPathColor = '#FFBB00'
    observedBearingColor = '#00FFFF'
    
    # project raw sensor detections in global frame using estimate pose
    xr, yr, tr = mu[:3]

    # plot the robot's 3-sigma covariance ellipsoid and path
    plt.plot(*robotPath, color=estimatedPathColor)
    helpers.plotCov2D(mu[:2], Sigma[:2, :2], color=estimatedPathColor, nSigma=3)

    for observation in z:
        # indicate observed angle relative to actual position
        plt.plot(np.array([xr, xr+np.cos(tr + observation[1])*observation[0]]), np.array([yr, yr+np.sin(tr + observation[1])*observation[0]]), color=observedBearingColor)
    
    for i in range(int((len(mu) - 3)/2)):
        # Plot a covariance ellipse for every landmark, too
        helpers.plotCov2D(mu[3+i*2:3+i*2+2], Sigma[3+i*2:3+i*2+2, 3+i*2:3+i*2+2], nSigma=3)
        # and an indicator 'cause those ellipses get SMALL (especially w.r.t. the size of the area we're mapping)
        plt.plot(*mu[3+i*2:3+i*2+2], "*", color="green")

    # plot the robot (so it's on top))
    helpers.plotRobot(mu[:3], "black", estimatedPathColor, r=0.5) # this is, uh... not what the vehicle really looks like.  But I'm lazy.


########################################
# TODO: Implement EKF Motion Prediction 
########################################
# https://www-personal.acfr.usyd.edu.au/nebot/experimental_data/modeling_info/Ute_modeling_info.htm
def predict(mu, Sigma, u, deltaT, Sigma_u, Q):
    # Vehicle Parameters
    a = 3.78; # [m] # used for the motion model/jacobians
    b = 0.50; # [m] # used for the motion model/jacobians
    L = 2.83; # [m] # used for the motion model/jacobians
    H = 0.76; # [m] # Not used - well, not here.  Technically used to translate the control vector, but that's already been done for us I think?
    N = (mu.shape[0]-3)//2 # Get number of known landmarks

    v_c, alpha = u # Extract useful data out of the controls
    x, y, theta = mu[:3] # Get robot data as useful variables

    # TODO: Implement the motion prediction and mean/covariance update equations
    phi = helpers.minimizedAngle(theta + alpha)
    mu_prime = np.array([deltaT*(v_c*np.cos(phi) - v_c/L*np.tan(alpha)*(a*np.sin(phi) + b*np.cos(phi))),
                         deltaT*(v_c*np.sin(phi) + v_c/L*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi))),
                         deltaT * (v_c / L) * np.tan(alpha)])
    
    # Remember - You have an update to the robot covariance, updates to the robot->landmark correlation, but the landmark covariance doesn't change.
    # We also have noise from the linearization (accounted for with G), noise in the controls (Sigma_u) AND additive noise (Q)
    F = np.block([np.eye(3), np.zeros((3,2*N))])

    G = F.T @ np.array([[1, 0, -deltaT*(v_c*np.sin(phi) + v_c/L*np.tan(alpha)*(a*np.cos(phi) - b*np.sin(phi)))],
                        [0, 1,  deltaT*(v_c*np.cos(phi) - v_c/L*np.tan(alpha)*(a*np.sin(phi) + b*np.cos(phi)))],
                        [0, 0, 1]]) @ F

    R = np.array([[-(a*np.sin(phi) + b*np.cos(phi))*np.tan(alpha)/L, -v_c*(a*np.sin(phi) + b*np.cos(phi))*(np.tan(alpha)**2 + 1)/L], 
                  [(a*np.cos(phi) - b*np.sin(phi))*np.tan(alpha)/L, v_c*(a*np.cos(phi) - b*np.sin(phi))*(np.tan(alpha)**2 + 1)/L], 
                  [deltaT*np.tan(alpha)/L, deltaT*v_c*(np.tan(alpha)**2 + 1)/L]])

    mu_bar = mu + F.T @ mu_prime
    Sigma_bar = (G @ Sigma @ G.T) + (F.T @ (R @ Sigma_u @ R.T) @ F) + (F.T @ Q @ F)

    return mu_bar, Sigma_bar

###############################################
# TODO: Implement Measurement Update Equations
###############################################
def update(mu_bar, Sigma_bar, association, H, Q, innovation, updateMethod):
    ind = np.flatnonzero(np.asarray(association > -1)) # Find the indices for landmarks we can update with.  -1 is new landmark, -2 is ignore.
    if (len(ind) == 0): # If there aren't any
        return mu_bar, Sigma_bar # Then we don't update.  Just short-circuit exit.
    if (updateMethod == "seq"): 
        for i in ind: 
            H_i = H[(2*i):(2*i)+2,:]
            innovation_i = innovation[2*i:2*i+2]
            # print(f"H_i shape: {H_i.shape}")
            # print(f"innovation_i shape is {innovation_i.shape}")

            S_i = H_i @ Sigma_bar @ H_i.T + Q
            K_i = Sigma_bar @ H_i.T @ np.linalg.inv(S_i)

            # print(f"K_i shape: {K_i.shape}")
            mu_bar += K_i @ innovation_i
            Sigma_bar = (np.eye(K_i.shape[0]) - K_i @ H_i) @ Sigma_bar
        mu_new = mu_bar
        Sigma_new = Sigma_bar
    elif (updateMethod == "batch"): 
        Q = block_diag(*[Q]*len(association))
        S = H @ Sigma_bar @ H.T + Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        mu_new = mu_bar + K @ (innovation)
        Sigma_new = (np.eye(K.shape[0]) - K @ H) @ Sigma_bar

    else:
        raise Exception("Unknown update method, '" + updateMethod + "' - it must be 'seq' or 'batch'")
    
    return mu_new, Sigma_new

##################################
# TODO: Implement Augment State
##################################
def augmentState(association, measurements, mu, Sigma, Q):
    indices = np.flatnonzero(np.asarray(association == -1)) # If our data association returned a "-1" for a measurement, it's a landmark to add to the state
    measurements = measurements[indices] # We want to filter out only the measurements we cared about
    x, y, theta = mu[:3] # We'll need the robot state
    Sigma_rr = Sigma[:3, :3] # And the main robot covariance, we won't change

    # For each measurement of a new landmark update your state
    for z in measurements:
        N = (mu.shape[0] - 3) // 2 # Number of landmarks currently in the state
        # Extract info for the measurement
        dist, bearing, sig = z
        alpha = helpers.minimizedAngle(bearing+theta)

        # Update both the mean (mu_l) and 
        # covariance (Sigma_lr, Sigma_rl, Sigma_Ll, Sigma_lL, Sigma_ll) for the new landmark.
        l_x = dist * np.cos(alpha) + x
        l_y = dist * np.sin(alpha) + y
        mu = np.concatenate((mu, [l_x, l_y]))

        G_r = np.array([[1, 0, -dist*np.sin(alpha)], 
                        [0, 1,  dist*np.cos(alpha)]])
        G_delta = np.array([[np.cos(alpha), -dist*np.sin(alpha)],
                            [np.sin(alpha), dist*np.cos(alpha)]])

        Sigma_lr = G_r @ Sigma_rr
        Sigma_ll = G_r @ Sigma_rr @ G_r.T + G_delta @ Q @ G_delta.T

        Sigma_rL = Sigma[:3, 3:2*N+3]
        Sigma_LL = Sigma[3:2*N+3, 3:2*N+3]
        Sigma_lL = G_r @ Sigma_rL

        Sigma = np.block([
            [Sigma_rr, Sigma_rL, Sigma_lr.T],
            [Sigma_rL.T, Sigma_LL, Sigma_lL.T],
            [Sigma_lr, Sigma_lL, Sigma_ll]
            ])

    return mu, Sigma
