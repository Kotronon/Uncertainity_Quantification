from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between two sets of points."""

    # TODO: compute the exponential covariance function.
    return np.zeros((x.shape[0], y.shape[0]))


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: npt.NDArray):
    """Computes the squared exponential covariance function between two sets of points."""

    # TODO: compute the squared exponential covariance function.
    return np.zeros((x.shape[0], y.shape[0]))


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

    # TODO: sample a Gaussian field suing the Cholesky decomposition.
    return np.zeros((n_samples, *mesh.shape[:-1]))


def plot_samples(samples, x_lims, y_lims):
    """Plots the samples from the Gaussian process."""
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample, cmap="coolwarm", origin="lower", extent=(*x_lims, *y_lims))
    return fig


if __name__ == "__main__":
    # TODO: set the condiguration.
    x_lims, y_lims = [None, None], [None, None]
    x_mesh_size, y_mesh_size = None, None
    scale = None
    mean = None
    seed = None
    n_samples = None
    rng = np.random.default_rng(seed)

    # TODO: create a 2D mesh.

    # TODO: sample from the Gaussian process with different kernels.

    # TODO: plot the samples.
