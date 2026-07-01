import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess


def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")
    axes[1].plot(t_grid, eigenfunctions(t_grid))
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    return fig


if __name__ == "__main__":
    # TODO: set the configuration.
    T = None
    n_points = None
    t_grid = np.linspace(0, T, n_points)
    Ms = [None]
    seed = None
    n_samples = None
    rng = np.random.default_rng(seed)

    # TODO: generate one realization of the Wiener process using the
    # standard definition.

    # TODO: generate approximations of the Wiener process using the KL expansion.

    # TODO: plot the approximation results.

    # TODO: visualize first eigenvalues and eigenfunctions.
