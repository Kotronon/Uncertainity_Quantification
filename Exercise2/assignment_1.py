import time

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.helpers import compute_errors, generate_grid, load_reference, simulate
from utils.interpolation import FirstBarycentricLagrange


def estimate_monte_carlo(
    omega_distr: cp.Distribution,
    n_samples: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # TODO: return the trajectories at the target time index for the given number of samples.
    # ====================================================================
    solutions = np.zeros(n_samples)
    # ====================================================================
    return solutions


def fit_lagrange(
    omega_bounds: tuple[float, float],
    n_nodes: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> FirstBarycentricLagrange:
    # TODO: Fit the Lagrange interpolator.
    # ====================================================================
    interpolator = None
    # ====================================================================
    return interpolator
    


def evaluate_pce(
    interpolator: FirstBarycentricLagrange, omega_distr: cp.Distribution, n_samples: int
) -> npt.NDArray:
    # TODO: compute the trajectories using the Lagrange surrogate.
    # ====================================================================
    solutions = np.zeros(n_samples)
    # ====================================================================
    return solutions


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: perform PCE computations and record the time taken.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: perform Monte Carlo sampling and record the time taken.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: compute relative errors.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results.
    # You may reuse code from previous exercises.
    # ====================================================================
    pass
    # ====================================================================
