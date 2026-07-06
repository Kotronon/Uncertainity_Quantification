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


def simulate_all(t_grid, nodes, fixed_args):
    # dists = [c_uniform, k_uniform, f_uniform, y0_uniform, y1_uniform]
    solutions = np.zeros(nodes.shape[1])
    for i in range(nodes.shape[1]):
        c  = nodes[0, i]
        k  = nodes[1, i]
        f  = nodes[2, i]
        y0 = nodes[3, i]
        y1 = nodes[4, i]

        omega  = fixed_args['omega']
        method = fixed_args['method']

        osc = Oscillator(c, k, f, omega)

        ys = osc.discretize(method=method, y0=y0, y1=y1, t_grid=t_grid)
        y_final = ys[-1]
        solutions[i] = y_final
    
    return solutions


def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[float, float]:
    """Computes the Sobol' indices using Monte Carlo sampling."""
    
    # TODO: implement the algorithm from the paper.
    def switch_parameter(A, B, parameter):
        A_new = A.copy()
        A_new[parameter, :] = B[parameter, :]
        return A_new

    A = distribution.sample(n_samples)
    B = distribution.sample(n_samples)

    n_args = A.shape[0]

    fA = simulate_all(t_grid=t_grid, nodes=A, fixed_args=fixed_args)
    # here the mean and variance is computed as the average output using sample matrix A (monte carlo)
    f0 = np.mean(fA) 
    V = np.var(fA, ddof=1) 

    # choosing method (a) from table 2 to compute S_i
    sobol = np.zeros(n_args)
    for i in range(n_args):
        B_Ai = switch_parameter(A=B, B=A, parameter=i)
        fB_Ai = simulate_all(t_grid=t_grid, nodes=B_Ai, fixed_args=fixed_args)
        sobol[i] = (np.mean(fA * fB_Ai) - f0**2) / V

    # choosing method (f) from table 2 to compute S_{Ti}
    jansen = np.zeros(n_args)
    for i in range(n_args):
        A_Bi = switch_parameter(A=A, B=B, parameter=i)
        fA_Bi = simulate_all(t_grid=t_grid, nodes=A_Bi, fixed_args=fixed_args)
        diff = fA - fA_Bi
        jansen[i] = np.mean(diff**2) / (2 * V)

    return sobol, jansen


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

    # print("n_samples: ", nodes.shape[1], "\n")

    # eval
    ys_final = simulate_all(t_grid, nodes, fixed_args)

    pce = cp.fit_quadrature(expansion, nodes, weights, ys_final)

    first = cp.Sens_m(pce, distribution)
    total = cp.Sens_t(pce, distribution)

    return first, total
