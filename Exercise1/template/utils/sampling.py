from typing import Callable

import chaospy as cp
import numpy as np
import numpy.typing as npt

Function = Callable[[npt.NDArray], npt.NDArray]


def compute_rmse(values: npt.NDArray) -> npt.NDArray:
    return np.std(values, ddof=1, axis=1) / np.sqrt(values.shape[1])


def monte_carlo(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    transform: Function | None = None,
    rule: str = "random",
    seed: float = 42,
) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: implement the Monte Carlo method.
    # Return the mean approximation and the corresponding RMSE. Make sure
    # the function works for both 1-dimensional and n-dimensional
    # distributions (see include_axis_dim parameter of
    # cp.Distribution.sample).
    # ====================================================================
    samples = p.sample(n_samples, rule=rule, seed=seed)
    if transform is not None:
        samples = transform(samples)
    f_values = f(samples)
    mean = np.mean(f_values)
    rmse = np.std(f_values, ddof=1) / np.sqrt(n_samples)
    # ====================================================================
    return np.array([mean]), np.array([rmse])


def control_variates(
    p: cp.Distribution,
    n_samples: int,
    f: Function,
    phi: Function,
    control_mean: float,
    seed: float = 42,
) -> npt.NDArray:
    # TODO: implement the control variates method that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    # ====================================================================
    mean = np.zeros(1)
    # ====================================================================
    return mean


def importance_sampling(
    p: cp.Distribution,
    q: cp.Distribution,
    n_samples: int,
    f: Function,
    seed: float = 42,
) -> npt.NDArray:
    # TODO: implement the importance sampling that returns the mean
    # approximation. Make sure the function works for both 1-dimensional
    # and n-dimensional distributions.
    # ====================================================================
    mean = np.zeros(1)
    # ====================================================================
    return mean
