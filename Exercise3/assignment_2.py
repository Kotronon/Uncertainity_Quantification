from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between two sets of points."""

    dists = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return np.exp(-dists / scale)


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: npt.NDArray):
    """Computes the squared exponential covariance function between two sets of points."""

    dists = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    return np.exp(-(dists**2) / (2 * scale**2))


def get_xy_mesh(
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    x_mesh_size: int,
    y_mesh_size: int,
) -> npt.NDArray:
    """Creates a 2D mesh grid for the given limits and mesh sizes."""
    x_step = (x_lims[1] - x_lims[0]) / x_mesh_size
    y_step = (y_lims[1] - y_lims[0]) / y_mesh_size
    x_grid = np.arange(x_lims[0] + x_step / 2, x_lims[1], x_step)
    y_grid = np.arange(y_lims[0] + y_step / 2, y_lims[1], y_step)
    mesh = np.stack(np.meshgrid(x_grid, y_grid), axis=-1)
    return mesh


def sample(mesh, mean_fn, cov_fn, n_samples, rng, reg_scale=1e-7):
    """Samples from a Gaussian process defined by the mean and covariance functions."""

    # Flatten the 2D mesh into an array of coordinates: shape (N^2, 2)
    pts = mesh.reshape(-1, 2)
    n_points = pts.shape[0]
    
    # Evaluate the mean function
    if callable(mean_fn):
        m = mean_fn(pts)
    else:
        m = np.full(n_points, mean_fn)
        
    # Evaluate the covariance matrix: shape (N^2, N^2)
    K = cov_fn(pts, pts)
    
    # Add a small nugget term to the diagonal for numerical stability (strictly positive definite)
    K += reg_scale * np.eye(n_points)
    
    # Perform Cholesky decomposition: K = L @ L.T
    L = np.linalg.cholesky(K)
    
    # Sample standard normal vectors: shape (n_points, n_samples)
    psi = rng.normal(size=(n_points, n_samples))
    
    # Compute Gaussian field realizations: m + L @ psi
    samples_flat = m[:, None] + L @ psi
    
    # Reshape back to grid format: (n_samples, y_mesh_size, x_mesh_size)
    y_mesh_size, x_mesh_size = mesh.shape[0], mesh.shape[1]
    samples = samples_flat.T.reshape(n_samples, y_mesh_size, x_mesh_size)
    
    return samples


def plot_samples(samples, x_lims, y_lims):
    """Plots the samples from the Gaussian process."""
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample, cmap="coolwarm", origin="lower", extent=(*x_lims, *y_lims))
    return fig


if __name__ == "__main__":
    
    # Configuration setup
    x_lims, y_lims = [0.0, 1.0], [0.0, 1.0]
    x_mesh_size, y_mesh_size = 30, 30  # Results in an N^2 = 900 point grid
    scale = 0.2                       # Length-scale parameter 'l'
    mean = 0.1                        # Constant mean function
    seed = 42
    n_samples = 3
    rng = np.random.default_rng(seed)

    # Create the 2D mesh
    mesh = get_xy_mesh(x_lims, y_lims, x_mesh_size, y_mesh_size)

    # Sample from the Gaussian process with different kernels
    # Wrap covariance functions using a lambda to pass the length-scale 'l'
    samples_exp = sample(mesh, mean, lambda x, y: exp_cov_fn(x, y, scale), n_samples, rng)
    samples_sq_exp = sample(mesh, mean, lambda x, y: squared_exp_cov_fn(x, y, scale), n_samples, rng)

    # Plot the samples
    fig_exp = plot_samples(samples_exp, x_lims, y_lims)
    fig_exp.suptitle("Exponential Covariance Function Samples ($C_1$)")
    
    fig_sq_exp = plot_samples(samples_sq_exp, x_lims, y_lims)
    fig_sq_exp.suptitle("Squared Exponential Covariance Function Samples ($C_2$)")
    
    plt.show()
