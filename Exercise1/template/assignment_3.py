import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.exp(x)
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return np.e - 1
    # ====================================================================


def run_monte_carlo(Ns: tuple[int, ...], seed: int = 42) -> list[float]:
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    p = cp.Uniform(0, 1) 
    true_value = analytical_integral()
    errors = []
    
    for N in Ns:
        # monte_carlo returns (mean_array, rmse_array)
        estimate, _ = monte_carlo(p, N, f, seed=seed)
        # Convert estimate[0] to float to calculate error
        errors.append(float(abs(estimate[0] - true_value)))
        
    return errors
    # ====================================================================


def run_control_variates(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================

    p = cp.Uniform(0, 1)
    true_value = analytical_integral()

    # Control variates functions
    h1 = lambda x: x
    h2 = lambda x: 1 + x
    h3 = lambda x: 1 + x + x**2 / 2

    # Exact expectations
    Eh1, Eh2, Eh3 = 0.5, 1.5, (1 + 0.5 + 1/6)

    errors_h1, errors_h2, errors_h3 = [], [], []

    for N in Ns:
        # Match function signature: (p, n_samples, f, phi, control_mean, seed)
        est1 = control_variates(p, N, f, h1, Eh1, seed=seed)
        est2 = control_variates(p, N, f, h2, Eh2, seed=seed)
        est3 = control_variates(p, N, f, h3, Eh3, seed=seed)

        errors_h1.append(float(abs(est1[0] - true_value)))
        errors_h2.append(float(abs(est2[0] - true_value)))
        errors_h3.append(float(abs(est3[0] - true_value)))

    return errors_h1, errors_h2, errors_h3
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):

    p = cp.Uniform(0, 1) # The original target distribution
    true_value = analytical_integral()

    errors_beta_51 = []
    errors_beta_half = []

    dist1 = cp.Beta(5, 1)
    dist2 = cp.Beta(0.5, 0.5)

    for N in Ns:
        est1 = importance_sampling(p, dist1, N, f, seed=seed)
        est2 = importance_sampling(p, dist2, N, f, seed=seed)

        errors_beta_51.append(float(abs(est1[0] - true_value)))
        errors_beta_half.append(float(abs(est2[0] - true_value)))

    return errors_beta_51, errors_beta_half
    # ====================================================================


if __name__ == "__main__":
    # parameters
    Ns = (10, 100, 1000, 10000)
    seed = 42

    # run methods
    mc_errors = run_monte_carlo(Ns, seed)

    cv_h1_errors, cv_h2_errors, cv_h3_errors = (
        run_control_variates(Ns, seed)
    )

    is_beta_51_errors, is_beta_half_errors = (
        run_importance_sampling(Ns, seed)
    )

    plt.figure(figsize=(8, 6))

    plt.loglog(Ns, mc_errors, marker="o", label="Monte Carlo")

    plt.loglog(Ns, cv_h1_errors, marker="o", label="CV: h1=x")
    plt.loglog(Ns, cv_h2_errors, marker="o", label="CV: h2=1+x")
    plt.loglog(Ns, cv_h3_errors, marker="o", label="CV: h3=1+x+x^2/2")

    plt.loglog(
        Ns,
        is_beta_51_errors,
        marker="o",
        label="IS: Beta(5,1)"
    )

    plt.loglog(
        Ns,
        is_beta_half_errors,
        marker="o",
        label="IS: Beta(0.5,0.5)"
    )

    plt.xlabel("Number of samples N")
    plt.ylabel("Absolute error")
    plt.title("Monte Carlo Error Comparison")
    plt.grid(True)
    plt.legend()

    plt.show()
