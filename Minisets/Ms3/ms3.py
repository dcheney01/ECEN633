import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import distance

def sample_multivariate_gaussian(mean, cov, num_samples):
    # Returns x and y points in a Gaussian distribution
    return np.random.multivariate_normal(mean, cov, num_samples).T

def get_covariance_ellipses(fig, mean, cov, nSigma, edge_color='r'):
    # Returns list of covariance ellipses to be plotted
    eig_val, eig_vec = np.linalg.eig(cov)
    alpha = np.arctan(eig_vec[0,1]/eig_vec[0,0]) * 180 / np.pi
    minor_axis_length = np.sqrt(eig_val[1]) * 2
    major_axis_length = np.sqrt(eig_val[0]) * 2

    for n in range(nSigma):
        e = Ellipse(
            xy=(mean[0], mean[1]),
            width=major_axis_length * n,
            height=minor_axis_length * n,
            angle=np.degrees(alpha),
            fill=False,
            linewidth=1,
            color=edge_color,
        )
        fig.gca().add_patch(e)

def plot_nonlinear_sensor_frame_distribution(fig, mean, cov, label, num_samples=1000):
    # Plots Gaussian Distribution of Sensor in Sensor Fram
    x,y = sample_multivariate_gaussian(mean, cov, num_samples)
    ax = fig.gca()
    ax.scatter(x,y, s=2, label=label)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (rad)")
    plt.title(f"Gaussian Distribution of Beacon Position in Sensor Frame")
    return fig

def plot_nonlinear_cartesian_distribution(fig, mean, cov, color, label, num_samples=1000):
    # Plots the believed position of the beam in cartesian space with nonlinear coordinates
    x,y = sample_multivariate_gaussian(mean, cov, num_samples)
    ax = fig.gca()

    x_cart = x*np.cos(y)
    y_cart  = x*np.sin(y)

    return ax.scatter(x_cart, y_cart, s=2, color=color, label=label)

def plot_linearized_cartesian_distribution(fig, mean, cov, color, label, num_samples= 1000):
    # Plots the believed position of the beam in cartesian space with linearized coordinates
    x,y = sample_multivariate_gaussian(mean, cov, num_samples)
    x_states = np.array([x,y])

    r0, th0 = mean
    J = np.array([[np.cos(th0), -r0*np.sin(th0)], 
                  [np.sin(th0), r0*np.cos(th0)]])
    fx0 = np.array([[r0*np.cos(th0)], 
                    [r0*np.sin(th0)]])
    x0 = np.reshape(mean, (2,1)) 
    y_lin = (J @ (x_states - x0))  + fx0

    ax = fig.gca()
    return ax.scatter(y_lin[0], y_lin[1], s=2, label=label, color=color), fx0, J


if __name__ == "__main__":

    fig = plt.figure()

    # The following is for UP-Q1
    # mean = np.array([10, 0])
    # cov = np.array([[.5**2, 0], [0, .25**2]])
    # plot_sensor_distribution(fig, mean, cov, 10000)
    # plot_cartesian_distribution(fig, mean, cov, 1000)

    # The following is for UP-Q2 and UP-Q3
    mean = np.array([10, 0])
    cov = np.array([[.5**2, 0], [0, .25**2]])
    
    linearized, mean_lin, cov_lin = plot_linearized_cartesian_distribution(fig, mean, cov, 'r', "Linear", 1000)
    get_covariance_ellipses(fig, mean_lin, cov_lin, 3, edge_color='r')

    nonlinear = plot_nonlinear_cartesian_distribution(fig, mean, cov, 'b', "Nonlinear", 1000)
    get_covariance_ellipses(fig, mean, cov, 3, edge_color='b')

    plt.legend((linearized, nonlinear),
           ('Linearized', 'Non-Linear'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8) 
    plt.plot(0, 'go')                           # Plot the robot for scale
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"Linearized vs Non-linear Beacon Position in Cartesian Frame")
    
    

    plt.show()