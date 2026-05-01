from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo

Function = Callable[[npt.NDArray], npt.NDArray]


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.zeros_like(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    return 0
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    samples = np.zeros_like(samples)
    # ====================================================================
    return samples


def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    # TODO: compute the integral with the Monta Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding RMSE.
    # ====================================================================
    integral, rmse = 0, 0
    # ====================================================================
    return integral, rmse


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    pass
    # ====================================================================
