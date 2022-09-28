import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plotcov2d(mean, cov, num_points= 500, nSigma=1):
    # % plot the ellipse
    x,y = np.random.multivariate_normal(mean, cov, num_points).T
    # covariance Ellipse Setup
    eig_val, eig_vec = np.linalg.eig(cov)
    alpha = np.arctan(eig_vec[0,1]/eig_vec[0,0]) * 180 / np.pi
    minor_axis_length = np.sqrt(eig_val[1]) * nSigma * 2
    major_axis_length = np.sqrt(eig_val[0]) * nSigma * 2

    # Plot the distribution
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=2)

    # Plot the covariance Ellipse
    e = Ellipse(mean, minor_axis_length, major_axis_length)
    e.set_edgecolor('r')
    e.set_linewidth(2)
    e.set_fill(False)
    e.set_angle(alpha)
    ax.add_artist(e)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (rad)")
    plt.title(f"{nSigma} Sigma Ellipse for a 2d gaussian distribution")
    plt.show()

def plot_distribution_sensor_frame(mean, cov, num_points= 500):
    x,y = np.random.multivariate_normal(mean, cov, num_points).T
    # Plot the distribution
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=2)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (rad)")
    plt.title(f"Gaussian Distribution of Beacon Position in Sensor Frame")
    plt.show()

def plot_distribution_cartesian_frame(mean, cov, num_points= 500):
    x,y = np.random.multivariate_normal(mean, cov, num_points).T
    # Convert to cartesian coordinates
    x_cart = x*np.cos(y)
    y_cart  = x*np.sin(y)

    # Plot the distribution
    fig, ax = plt.subplots()
    ax.scatter(x_cart, y_cart, s=2)
    plt.plot(0, 'ro')

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"Gaussian Distribution of Beacon Position in Cartesian Frame")
    plt.show()


# Linearized Covariance of the Beacon Position
def get_linearized_covariance_beacon(mean, cov, num_points= 500):
    x,y = np.random.multivariate_normal(mean, cov, num_points).T
 
    # Get the linearized distribution
    r0, th0 = mean
    J = np.array([[np.cos(th0), -r0*np.sin(th0)], [np.sin(th0), r0*np.cos(th0)]])
    fx0 = np.reshape(np.array([r0*np.cos(th0), r0*np.sin(th0)]), (2,1)) * np.ones((2, num_points))
    x_states = np.array([x,y])
    x0 = np.reshape(mean, (2,1)) * np.ones((2, num_points))
    y_lin = J @ x_states - J @ x0  + fx0

    # Get the non linear distribution
    x_cart = x*np.cos(y)
    y_cart  = x*np.sin(y)

    # Plot the distribution
    fig, ax = plt.subplots()
    linearized = ax.scatter(y_lin[0], y_lin[1], s=2)
    nonlinear = ax.scatter(x_cart,y_cart, s=2)
    # Plot the robot
    plt.plot(0, 'ro')

    plt.legend((linearized, nonlinear),
           ('Linearized', 'Non-Linear'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"Linearized vs Non-linear Transformation from Sensor Frame to Cartesian Frame")
    plt.show()



def plot_sigma_linearized(mean, cov, num_points= 500): 
    # % plot the ellipse
    x,y = np.random.multivariate_normal(mean, cov, num_points).T
    
    # Get the linearized distribution
    r0, th0 = mean
    J = np.array([[np.cos(th0), -r0*np.sin(th0)], [np.sin(th0), r0*np.cos(th0)]])
    fx0 = np.reshape(np.array([r0*np.cos(th0), r0*np.sin(th0)]), (2,1)) * np.ones((2, num_points))
    x_states = np.array([x,y])
    x0 = np.reshape(mean, (2,1)) * np.ones((2, num_points))
    y_lin = J @ x_states - J @ x0  + fx0

    # Get the linearized covariance
    cov_lin = J @ cov @ J.T
    

    # covariance for linearized Ellipse Setup
    eig_val, eig_vec = np.linalg.eig(cov_lin)
    alpha = np.arctan(eig_vec[0,0]/eig_vec[0,1]) * 180 / np.pi
    minor_axis_length = np.sqrt(eig_val[1]) * 2
    major_axis_length = np.sqrt(eig_val[0]) * 2

    # covariance ellipse for cartesian
    cart_eig_val, cart_eig_vec = np.linalg.eig(cov)
    cart_alpha = np.arctan(cart_eig_vec[0,0]/cart_eig_vec[0,1]) * 180 / np.pi
    cart_minor_axis_length = np.sqrt(cart_eig_val[1]) * 2
    cart_major_axis_length = np.sqrt(cart_eig_val[0]) * 2

    x_cart = x*np.cos(y)
    y_cart  = x*np.sin(y)

    # Plot the distribution
    fig, ax = plt.subplots()
    ax.scatter(x_cart, y_cart, s=2)

    # Plot the robot
    plt.plot(0, 'ro')

    for i in range(3):
        e = Ellipse(mean, minor_axis_length*(i+1), major_axis_length*(i+1))
        e.set_edgecolor('r')
        e.set_linewidth(2)
        e.set_fill(False)
        e.set_angle(alpha)
        ax.add_artist(e)

        e_cart = Ellipse(mean, cart_minor_axis_length*(i+1), cart_major_axis_length*(i+1))
        e_cart.set_edgecolor('b')
        e_cart.set_linewidth(2)
        e_cart.set_fill(False)
        e_cart.set_angle(cart_alpha)
        ax.add_artist(e_cart)
        # e.append(e_new)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"3 Sigma Covarariance Ellipse Comparison ")
    plt.show()


if __name__ == "__main__":

    mean = np.array([10, 0])
    cov = np.array([[.5**2, 0], [0, .25**2]])
    # plotcov2d(mean, cov, 10000, 1)
    # plot_distribution_sensor_frame(mean, cov, 5000)
    # plot_distribution_cartesian_frame(mean, cov, 500)
    # get_linearized_covariance_beacon(mean, cov, 500)
    plot_sigma_linearized(mean, cov, 5000)