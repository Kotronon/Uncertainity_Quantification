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
    np.random.seed(seed)
    true_value = analytical_integral()
    errors = []
    for N in Ns:
        estimate = monte_carlo(f, N)
        errors.append(abs(estimate - true_value))
    return errors
    # ====================================================================


def run_control_variates(
    Ns: tuple[int, ...], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    np.random.seed(seed)

    true_value = analytical_integral()

    # control variates
    h1 = lambda x: x
    h2 = lambda x: 1 + x
    h3 = lambda x: 1 + x + x**2 / 2

    # exact expectations on [0,1]
    Eh1 = 0.5
    Eh2 = 1.5
    Eh3 = 1 + 0.5 + 1 / 6

    errors_h1 = []
    errors_h2 = []
    errors_h3 = []

    for N in Ns:

        est1 = control_variates(f, h1, Eh1, N)
        est2 = control_variates(f, h2, Eh2, N)
        est3 = control_variates(f, h3, Eh3, N)

        errors_h1.append(abs(est1 - true_value))
        errors_h2.append(abs(est2 - true_value))
        errors_h3.append(abs(est3 - true_value))

    return errors_h1, errors_h2, errors_h3
    # ====================================================================


def run_importance_sampling(
    Ns: tuple[int, ...], seed: int = 42
):
    np.random.seed(seed)

    true_value = analytical_integral()

    errors_beta_51 = []
    errors_beta_half = []

    # Beta(5,1)
    dist1 = cp.Beta(5, 1)

    # Beta(0.5,0.5)
    dist2 = cp.Beta(0.5, 0.5)

    for N in Ns:

        est1 = importance_sampling(f, dist1, N)
        est2 = importance_sampling(f, dist2, N)

        errors_beta_51.append(abs(est1 - true_value))
        errors_beta_half.append(abs(est2 - true_value))

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
