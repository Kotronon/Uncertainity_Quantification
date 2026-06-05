import chaospy as cp
import matplotlib.pyplot as plt
import numpoly
import numpy as np
import numpy.typing as npt

from utils.helpers import load_reference, simulate


def compute_coefficients(
    nodes: npt.NDArray,
    weights: npt.NDArray,
    polynomials: numpoly.ndpoly,
    target_t: float,
    model_kwargs: dict,
    init_cond: dict,
    mode: str,
) -> npt.NDArray:
    # TODO: compute the coefficients using the quadrature rule, either manually.
    # or using chaospy functionality.
    # ====================================================================
    coefficients = np.zeros(polynomials.shape[0])
    # ====================================================================
    return coefficients


def compute_moments(coefficients: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the target mean and variance from the PCE coefficients.
    # ====================================================================
    mean, variance = 0.0, 0.0
    # ====================================================================
    return mean, variance


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: compute pseudo-spectral coefficients.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: compute the momements and calcualte their errors.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results.
    # You may reuse code from previous exercises.
    # ====================================================================
    pass
    # ====================================================================
