import numpy as np
import numpy.typing as npt
import scipy.special as sp

from utils.oscillator import Oscillator


def generate_grid(
    bounds: tuple[float, float], n_nodes: int, grid_type: str = "uniform"
) -> npt.NDArray:
    # TODO: generate the interpoliation grid on [low, high] with n_nodes points.
    # You may reuse code from previous exercises.
    # ====================================================================
    grid = np.zeros(n_nodes)
    # ====================================================================
    return grid


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # You may reuse code from previous exercises.
    # ====================================================================
    mean_error, var_error = 0.0, 0.0
    # ====================================================================
    return mean_error, var_error


def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # You may reuse code from previous exercises.
    # ====================================================================
    mean, var = 0.0, 0.0
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_samples: npt.NDArray,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    sample_solutions = np.zeros_like(t_grid)
    # ====================================================================
    return sample_solutions
