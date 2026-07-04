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
    c_uniform = cp.Uniform(c_lims[0], c_lims[1])
    k_uniform = cp.Uniform(k_lims[0], k_lims[1])
    f_uniform = cp.Uniform(f_lims[0], f_lims[1])
    y0_uniform = cp.Uniform(y0_lims[0], y0_lims[1])
    y1_uniform = cp.Uniform(y1_lims[0], y1_lims[1])

    dists = [c_uniform, k_uniform, f_uniform, y0_uniform, y1_uniform]

    joint_uniform = cp.J(*dists)
    
    return joint_uniform


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.
    pass


if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = [8e-2, 12e-2]
    k_lims = [3e-2, 4e-2]
    f_lims = [8e-2, 12e-2]
    y0_lims = [45e-2, 55e-2]
    y1_lims = [-5e-2, 5e-2]

    # TODO: set the determinisic parameters.
    fixed_args = {"omega": 1.0}

    # TODO: set the parameters of the methods.
    quadrature_degree = [3, 4] # K
    pce_degree = [3, 4] # N
    n_samples = None

    # TODO: set the time domain
    T_min = 0
    T_max = 10
    dt = 0.01
    t_grid = np.arange(T_min, T_max + dt, dt)

    ###########################################################################

    # TODO: define the distribution over the stochastic parameters.
    joint_uniform = get_distribution(c_lims=c_lims, k_lims=k_lims, f_lims=f_lims, y0_lims=y0_lims, y1_lims=y1_lims)

    # TODO: run the pseudo-spectral method on full grid.
    for quad_deg in quadrature_degree:
        for pce_deg in pce_degree:
            pseudo_spectral_sobol(
                pce_degree=pce_deg, 
                quadrature_degree=quad_deg, 
                distribution=joint_uniform, 
                t_grid=t_grid, 
                fixed_args=fixed_args, 
                sparse=False,
            )
    # TODO: run the pseudo-spectral method on sparse grid.
    for quad_deg in quadrature_degree:
        for pce_deg in pce_degree:
            pseudo_spectral_sobol(
                pce_degree=pce_deg, 
                quadrature_degree=quad_deg, 
                distribution=joint_uniform, 
                t_grid=t_grid, 
                fixed_args=fixed_args, 
                sparse=True,
            )

    # TODO: run the Monte Carlo method.

    ###########################################################################
