import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    samples = np.zeros((len(mu_target), n_samples))
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================
    mean = np.zeros(samples.shape[0])
    covariance = np.zeros((samples.shape[0], samples.shape[0]))
    # ====================================================================
    return mean, covariance


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    pass
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    pass
    # ====================================================================
