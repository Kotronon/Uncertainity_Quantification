import numpy as np
import numpy.typing as npt
import scipy.special as sp

from utils.oscillator import Oscillator


def generate_grid(
    bounds: tuple[float, float], n_nodes: int, grid_type: str = "uniform"
) -> npt.NDArray:
    # ====================================================================
    lower, upper = bounds
    if grid_type == "uniform":
        grid = np.linspace(lower, upper, n_nodes)
    elif grid_type == "chebyshev":
        roots, _ = sp.roots_chebyt(n_nodes)
        grid = 0.5 * (lower + upper) + 0.5 * (upper - lower) * roots
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")
    # ====================================================================
    return grid


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # ====================================================================
    mean_error = abs(1.0 - np.mean(samples) / mean_ref)
    var_error = abs(1.0 - np.var(samples, ddof=1) / var_ref)
    # ====================================================================
    return mean_error, var_error


def load_reference(filename: str) -> tuple[float, float]:
    # ====================================================================
    mean, var = np.loadtxt(filename)
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_samples: npt.NDArray,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # ====================================================================
    omega_samples = np.asarray(omega_samples).reshape(-1)
    sample_solutions = np.empty((omega_samples.size, t_grid.size))
    for index, omega in enumerate(omega_samples):
        oscillator = Oscillator(omega=float(omega), **model_kwargs)
        sample_solutions[index] = oscillator.discretize(
            method="odeint", t_grid=t_grid, **init_cond
        )
    # ====================================================================
    return sample_solutions
