import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import distance

def sample_multivariate_gaussian(mean, cov, num_samples):
    # Returns x and y points in a Gaussian distribution
    return np.random.multivariate_normal(mean, cov, num_samples).T

def get_covariance_ellipses(ax, mean, cov, nSigma, edge_color='r'):
    # Returns list of covariance ellipses to be plotted
    eig_val, eig_vec = np.linalg.eig(cov)
    alpha = np.arctan(eig_vec[0,1]/eig_vec[0,0]) * 180 / np.pi
    minor_axis_length = np.sqrt(eig_val[1]) * 2
    major_axis_length = np.sqrt(eig_val[0]) * 2

    for n in range(nSigma):
        e = Ellipse(
            xy=(mean[0], mean[1]),
            width=major_axis_length * (n+1),
            height=minor_axis_length * (n+1),
            angle=np.degrees(alpha),
            fill=False,
            linewidth=1,
            color=edge_color,
        )
        ax.add_patch(e)

def plot_nonlinear_sensor_frame_distribution(ax, mean, cov, label="", num_samples=1000):
    # Plots Gaussian Distribution of Sensor in Sensor Fram
    x,y = sample_multivariate_gaussian(mean, cov, num_samples)
    ax.scatter(x,y, s=2, label=label)

def get_cartesian_gaussian(mean, cov, num_samples):
    x,y = sample_multivariate_gaussian(mean, cov, num_samples)

    x_cart = x*np.cos(y)
    y_cart  = x*np.sin(y)
    return np.array([x_cart, y_cart]).T

def plot_nonlinear_cartesian_distribution(ax, mean, cov, color='b', label="", num_samples=1000):
    # Plots the believed position of the beam in cartesian space with nonlinear coordinates
    x_cart, y_cart = get_cartesian_gaussian(mean, cov, num_samples)

    mean_cart = np.array([[np.mean(x_cart)], np.mean(y_cart)])

    return ax.scatter(x_cart, y_cart, s=2, color=color, label=label), mean_cart, np.cov([x_cart,y_cart])

def plot_linearized_cartesian_distribution(ax, mean, cov, color='r', label="", num_samples= 1000):
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

    cov_lin = J @ cov @ J.T

    if ax is None:
        return None, fx0, cov_lin, J
    return ax.scatter(y_lin[0], y_lin[1], s=2, label=label, color=color), fx0, cov_lin, J

def plot_robot(ax):
    ax.plot(0, 'go') 

def get_ellipse_distances(cart_points):
    cov = np.cov(cart_points.T)
    mean = np.mean(cart_points, axis=0)  # mean along sample dimension
    distances = []
    for i in range(len(cart_points)):
        point = cart_points[i, :]  # ith sample
        q = distance.mahalanobis(point, mean, np.linalg.inv(cov))
        distances.append(q)
    return np.array(distances)

def UPQ1():
    # The following is for UP-Q1
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title('Sensor Frame Distribution', fontsize=12)
    axs[0].set_xlabel('Range (m)', fontsize=8)
    axs[0].set_ylabel('Bearing (rads)', fontsize=8)

    axs[1].set_title('Beacon Position in Cartesian Frame', fontsize=12)
    axs[1].set_xlabel('X (m)', fontsize=8)
    axs[1].set_ylabel('Y (m)', fontsize=8)
    
    fig.suptitle('Beacon Position in Sensor and Cartesian Frames', fontsize=16)

    mean = np.array([10, 0])
    cov = np.array([[.5**2, 0], [0, .25**2]])
    plot_nonlinear_sensor_frame_distribution(axs[0], mean, cov)
    plot_nonlinear_cartesian_distribution(axs[1], mean, cov)
    plot_robot(axs[1])

    plt.show()

def UPQ2():
    mean = np.array([10, 0])
    cov = np.array([[.5**2, 0], [0, .25**2]])
    
    _, _, cov_lin, J = plot_linearized_cartesian_distribution(None, mean, cov, num_samples=1000)
    print(f"Jacobian to transform Observation to Cartesian frame: \n{J}\n")
    print(f"Linearized Cartesian Covariance Matrix: \n{cov_lin}\n")

def UPQ3():
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title('Sensor Frame Distribution', fontsize=12)
    axs[0].set_xlabel('Range (m)', fontsize=8)
    axs[0].set_ylabel('Bearing (rads)', fontsize=8)

    axs[1].set_title('Beacon Position in Cartesian Frame', fontsize=12)
    axs[1].set_xlabel('X (m)', fontsize=8)
    axs[1].set_ylabel('Y (m)', fontsize=8)
    
    fig.suptitle('Beacon Position in Sensor and Cartesian Frames', fontsize=16)

    mean = np.array([10, 0])
    cov = np.array([[.5**2, 0], [0, .25**2]])

    plot_nonlinear_sensor_frame_distribution(axs[0], mean, cov)
    get_covariance_ellipses(axs[0], mean, cov, 3, edge_color='b') # Non-linearized ellipses are blue

    _, mean_cart, cov_cart = plot_nonlinear_cartesian_distribution(axs[1], mean, cov, label="Nonlinear")
    _, mean_lin, cov_lin, _ = plot_linearized_cartesian_distribution(axs[1], mean, cov, label="Linear", num_samples=1000)
    get_covariance_ellipses(axs[1], mean_cart, cov_cart, 3, edge_color='b') # Non-linearized ellipses are blue
    get_covariance_ellipses(axs[1], mean_lin, cov_lin, 3, edge_color='r') # Linearized Ellipses are red

    plot_robot(axs[1])

    plt.show()

def UPQ4():
    mean = np.array([10, 0])
    cov = np.array([[.01**2, 0], [0, .01**2]])

    cart_points = get_cartesian_gaussian(mean, cov, 1000)
    distances = get_ellipse_distances(cart_points)

    samples_less_than_one = np.sum(distances <= 1)
    samples_less_than_two = np.sum(distances <= 2)
    samples_less_than_three = np.sum(distances <= 3)
    print(f"Percentage of samples in 1-sigma: {samples_less_than_one/10}%")
    print(f"Percentage of samples in 2-sigma: {samples_less_than_two/10}%")
    print(f"Percentage of samples in 3-sigma: {samples_less_than_three/10}%")


if __name__ == "__main__":
    # UPQ1()

    # UPQ2()

    # UPQ3()

    UPQ4()
    
    

