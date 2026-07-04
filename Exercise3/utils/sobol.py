import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator


def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""

    # TODO: evaluate the oscillator model for each sample.
    return np.zeros((samples.shape[0], 1))


def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[float, float]:
    """Computes the Sobol' indices using Monte Carlo sampling."""
    
    # TODO: implement the algorithm from the paper.
    return 0, 0


def simulate_all(t_grid, nodes, fixed_args, method='odeint'):
    # dists = [c_uniform, k_uniform, f_uniform, y0_uniform, y1_uniform]
    solutions = np.zeros(nodes.shape[1])
    for i in range(nodes.shape[1]):
        c  = nodes[0, i]
        k  = nodes[1, i]
        f  = nodes[2, i]
        y0 = nodes[3, i]
        y1 = nodes[4, i]

        omega = fixed_args['omega']

        osc = Oscillator(c, k, f, omega)

        ys = osc.discretize(method=method, y0=y0, y1=y1, t_grid=t_grid)
        y_final = ys[-1]
        solutions[i] = y_final
    
    return solutions


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""
    
    # TODO: implement the pseduo-spectral method.
    # genreate poly basis
    expansion = cp.generate_expansion(pce_degree, distribution)

    # quad
    nodes, weights = cp.generate_quadrature(quadrature_degree, distribution)

    # eval
    ys_final = simulate_all(t_grid, nodes, fixed_args)

    pce = cp.fit_quadrature(expansion, nodes, weights, ys_final)

    first = cp.Sens_m(pce, distribution)
    total = cp.Sens_t(pce, distribution)

    return first, total
