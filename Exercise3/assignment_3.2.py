import time
from functools import partial
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess


def generate_f_samples(
    mu: float,
    t_grid: npt.NDArray,
    n_samples: int,
    M: int | None,
    rng: np.random.Generator,
) -> list[Callable[[float], float]]:
    """Generates samples of the Wiener process."""

    # TODO: generate realizations of the Wiener process for f(t).
    # If M is None, we generate samples using the standard definition.
    # If M is not None, we generate samples using the KL expansion with M terms.
    # The samples are returned as a list of callable functions that
    # evaluate the Wiener process at a given time point.
    return [lambda t: 0.0] * n_samples


def simulate(
    t_grid: npt.NDArray,
    f_samples: list[Callable[[float], float]],
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""

    # TODO: simulate the oscillator model for each sample of f(t) and
    # return the trajectories as 2D array.
    return np.zeros(len(f_samples), len(t_grid))


def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""

    # TODO: compute the metrics.
    return np.zeros(solutions.shape[1]), np.zeros(solutions.shape[1])


def plot_solutions(
    t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # TODO: set parameters of the model.
    f_mean = None
    model_kwargs = {"c": None, "k": None, "omega": None}
    init_cond = {"y0": None, "y1": None}

    # TODO: set the time domain.
    T_max = None
    dt = None
    t_grid = np.arange(0, T_max + dt, dt)

    # TODO: set the number of Monte-Carlo samples and KL terms.
    N = None
    Ms = [None]
    seed = None
    rng = np.random.default_rng(seed)

    ###########################################################################

    # TODO: generate samples of the Wiener process for f using the stadard
    # generation and the KL expansion for different M.

    # TODO: simulate the oscillator model for each sample of f and record the
    # mean and standard deviation of the solutions at T_max.

    # TODO: optionally, plot the solutions for each sample of f.