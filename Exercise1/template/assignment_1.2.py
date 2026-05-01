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
    samples = np.random.multivariate_normal(mu_target, V_target, size=n_samples).T
    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================
    # mean = np.mean(samples, axis=1)
    # covariance = np.cov(samples, ddof=1)

    mean = np.zeros(samples.shape[0])
    for i in range(samples.shape[1]):
        mean += np.array([samples[0][i], samples[1][i]])
    mean /= 2
    covariance = np.zeros((samples.shape[0], samples.shape[0]))

    # ====================================================================
    return mean, covariances


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    mu = np.array([-0.4, 1.1])
    V = np.array([[2, 0.4], [0.4, 1]])
    sample_sizes = [10, 100, 1000, 10000]
    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    vals = []
    for s in sample_sizes:
        sample = sample_normal(n_samples=s, mu_target=mu, V_target=V)
        mean, var = compute_moments(sample)


    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    pass
    # ====================================================================
