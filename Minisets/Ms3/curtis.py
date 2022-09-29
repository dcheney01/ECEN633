
def plot_cartesian_point_cloud(fig, point_cloud, label):
    x, y = point_cloud.T
    ax = fig.gca()
    ax.scatter(x, y, alpha=0.5, label=label)
    ax.axis("equal")
    ax.scatter(0, 0, color="red")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    return fig

def count_number_samples_in_ellipses(point_cloud):
    cov = np.cov(point_cloud.T)
    mean = np.mean(point_cloud, axis=0)  # mean along sample dimension
    distances = []
    for i in range(len(point_cloud)):
        point = point_cloud[i, :]  # ith sample
        q = distance.mahalanobis(point, mean, np.linalg.inv(cov))
        distances.append(q)
    return np.array(distances)



means = np.array([10, 0])
covariance = np.array([[(0.5) ** 2, .1125], [0.1125, (0.25/5) ** 2]])
fig = plt.figure()
fig = plot_cartesian_point_cloud(
    fig, map_sensor_to_cartesian_uncertainty(means, covariance), "nonlinear"
)
linearized_positions = map_sensor_to_cartesian_uncertainty_linearized(means, covariance)
fig = plot_cartesian_point_cloud(fig, linearized_positions, "linear")
fig = plot_ellipses(fig, means, np.cov(linearized_positions.T))
distances = count_number_samples_in_ellipses(
    map_sensor_to_cartesian_uncertainty(means, covariance)
)
samples_less_than_one = np.sum(distances < 1)
samples_less_than_two = np.sum(distances < 2)
samples_less_than_three = np.sum(distances < 3)
print(f"Percentage of samples in 1-sigma: {samples_less_than_one/1000}")
print(f"Percentage of samples in 2-sigma: {samples_less_than_two/1000}")
print(f"Percentage of samples in 3-sigma: {samples_less_than_three/1000}")
fig = plt.figure()
fig = plot_sensor_point_cloud(fig, means, covariance)
fig = plot_ellipses(fig, means, covariance)
plt.show()