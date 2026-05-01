import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.zeros_like(x)
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return 0
    # ====================================================================


def run_monte_carlo(Ns: tuple[int, ...], seed: int = 42) -> list[float]:
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    return [0] * len(Ns)
    # ====================================================================


def run_control_variates(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    return (0, 0, 0)
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    return (0, 0)
    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: run all the methods.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    pass
    # ====================================================================
