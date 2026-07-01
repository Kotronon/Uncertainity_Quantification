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
    return 0, 0
