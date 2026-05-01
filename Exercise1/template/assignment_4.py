from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator


def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================
    mean, var = 0, 1
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    sample_solutions = np.zeros((n_samples, len(t_grid)))
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    mean_error, var_error = 0, 0
    # ====================================================================
    return mean_error, var_error


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    pass
    # ====================================================================
