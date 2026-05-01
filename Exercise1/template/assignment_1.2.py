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
    np_mean = np.mean(samples, axis=1)
    np_covariances = np.cov(samples, ddof=1)

    py_mean = np.zeros(samples.shape[0])
    for i in range(samples.shape[1]):
        py_mean += np.array([samples[0][i], samples[1][i]])
    py_mean /= samples.shape[1]

    py_covariances = np.zeros((samples.shape[0], samples.shape[0]))
    for n in range(samples.shape[1]):
        add = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(samples.shape[0]):
            for j in range(samples.shape[0]):
                add[i][j] = (samples[i][n] - py_mean[i]) * (samples[j][n] - py_mean[j])
        py_covariances += add
    py_covariances /= samples.shape[1] - 1

    eps = 0.0001
    assert(np.isclose(np_mean, py_mean, atol=eps).all())
    assert(np.isclose(np_covariances, py_covariances, atol=eps).all())
    # ====================================================================
    return py_mean, py_covariances


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    mu = np.array([-0.4, 1.1])
    V = np.array([[2, 0.4], [0.4, 1]])
    sample_sizes = [10, 100, 1000, 10000]
    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================
    mean_error = []
    covariances_diagonal_error = []
    covariances_off_diagonal_error = []
    rmses = []

    for s in sample_sizes:
        sample = sample_normal(n_samples=s, mu_target=mu, V_target=V)
        mean, covariances = compute_moments(sample)

        absolute_error_mean = np.abs(mean - mu)
        absolute_error_covariances = np.abs(covariances - V)

        RMSE = np.zeros(mean.shape[0])
        for i in range(mean.shape[0]):
            RMSE[i] = np.sqrt(covariances[i][i]) / np.sqrt(s)
        
        mean_error.append(absolute_error_mean[0])
        covariances_diagonal_error.append(absolute_error_covariances[0, 0])
        covariances_off_diagonal_error.append(absolute_error_covariances[0, 1])
        rmses.append(RMSE[0])
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    plt.figure()

    plt.loglog(sample_sizes, mean_error, 'o-', label='abs mean error')
    plt.loglog(sample_sizes, covariances_diagonal_error, 'o-', label='abs diagonal covariances error')
    plt.loglog(sample_sizes, covariances_off_diagonal_error, 'o-', label='abs off-diagonal covariances error')
    plt.loglog(sample_sizes, rmses, 'o-', label='RMSE')

    plt.xlabel('Number of samples (N)')
    plt.ylabel('Error')
    plt.title('Error vs Sample Size (log-log)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.show()
    # ====================================================================
