import time

import chaospy as cp
import numpy as np

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol


def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    # TODO: create a joint distribution over the stochastic parameters.
    return cp.Uniform()


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.
    pass


if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = [None, None]
    k_lims = [None, None]
    f_lims = [None, None]
    y0_lims = [None, None]
    y1_lims = [None, None]

    # TODO: set the determinisic parameters.
    fixed_args = {"omega": None}

    # TODO: set the parameters of the methods.
    quadrature_degree = None
    pce_degree = None
    n_samples = None

    # TODO: set the time domain
    T_max = None
    dt = None
    t_grid = np.arange(0, T_max + dt, dt)

    ###########################################################################

    # TODO: define the distribution over the stochastic parameters.

    # TODO: run the pseudo-spectral method on full grid.

    # TODO: run the pseudo-spectral method on sparse grid.

    # TODO: run the Monte Carlo method.

    ###########################################################################
