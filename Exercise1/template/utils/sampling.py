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
    samples = p.sample(n_samples, seed=seed)

    f_values = f(samples)
    phi_values = phi(samples)

    f_bar = np.mean(f_values)
    phi_bar = np.mean(phi_values)

    # np.cov returns a covariance matrix: [[Var(f), Cov(f, phi)], [Cov(phi, f), Var(phi)]]
    cov_matrix = np.cov(f_values, phi_values)
    covariance_f_phi = cov_matrix[0, 1]
    variance_phi = cov_matrix[1, 1]
    
    c_star = covariance_f_phi / variance_phi if variance_phi != 0 else 0

    mean_cv = f_bar - c_star * (phi_bar - control_mean)

    return np.atleast_1d(mean_cv)


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
    samples = q.sample(n_samples, seed=seed)

    f_values = f(samples)

    p_pdf = p.pdf(samples)
    q_pdf = q.pdf(samples)
    
    weights = p_pdf / q_pdf

    mean_is = np.mean(f_values * weights)

    return np.atleast_1d(mean_is)
