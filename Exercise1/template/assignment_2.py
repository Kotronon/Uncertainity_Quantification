from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import monte_carlo

Function = Callable[[npt.NDArray], npt.NDArray]


def f(x: npt.NDArray) -> npt.NDArray:
    # TODO: define the target function.
    # ====================================================================
    return np.sin(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    if a > b:
        raise ValueError("a should be less than or equal to b.")
    elif a == b:
        return 0
    else:
        return -np.cos(b) + np.cos(a)
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    return a + (b - a) * samples
    # ====================================================================


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
    if with_transform:
        p = cp.Uniform(0, 1)
        trans = partial(transform, a=a, b=b)
    else:
        p = cp.Uniform(a, b)
        trans = None
    integral_arr, rmse_arr = monte_carlo(p, n_samples, f, trans, seed=seed)
    integral = (b - a) * integral_arr.item()
    # For RMSE, since Var(I) = (b-a)^2 * Var(mean_f), so RMSE = (b-a) * rmse_arr.item()
    rmse = (b - a) * rmse_arr.item()
    return integral, rmse
    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    N_list = [10, 100, 1000, 10000]
    seed = 42
    # ====================================================================

    # TODO: compute the integral and the errors.
    # ====================================================================
    # Assignment 2.1
    exact1 = analytical_integral(0, 1)
    print(f"Exact integral for [0,1]: {exact1}")
    errors1 = []
    rmses1 = []
    for N in N_list:
        I, rmse = integrate_mc(f, 0, 1, N, with_transform=False, seed=seed)
        error = abs(I - exact1)
        errors1.append(error)
        rmses1.append(rmse)
        print(f"N={N}: I={I}, error={error}, RMSE={rmse}")

    # Assignment 2.2
    exact2 = analytical_integral(2, 4)
    print(f"Exact integral for [2,4]: {exact2}")
    errors_direct = []
    rmses_direct = []
    errors_trans = []
    rmses_trans = []
    for N in N_list:
        I_d, rmse_d = integrate_mc(f, 2, 4, N, with_transform=False, seed=seed)
        error_d = abs(I_d - exact2)
        errors_direct.append(error_d)
        rmses_direct.append(rmse_d)
        print(f"Direct N={N}: I={I_d}, error={error_d}, RMSE={rmse_d}")

        I_t, rmse_t = integrate_mc(f, 2, 4, N, with_transform=True, seed=seed)
        error_t = abs(I_t - exact2)
        errors_trans.append(error_t)
        rmses_trans.append(rmse_t)
        print(f"Transform N={N}: I={I_t}, error={error_t}, RMSE={rmse_t}")
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    
    plt.figure(figsize=(10, 6))
    plt.loglog(N_list, errors1, 'o-', label='Exact Error [0,1]')
    plt.loglog(N_list, rmses1, 's-', label='RMSE [0,1]')
    plt.xlabel('Number of samples N')
    plt.ylabel('Error / RMSE')
    plt.title('Monte Carlo Integration Errors for [0,1]')
    plt.legend()
    plt.grid(True)
    plt.savefig("Exercise1/template/results/task2_monte_carlo_errors_01.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.loglog(N_list, errors_direct, '^-', label='Exact Error Direct [2,4]')
    plt.loglog(N_list, rmses_direct, 'v-', label='RMSE Direct [2,4]')
    plt.loglog(N_list, errors_trans, 'd-', label='Exact Error Transform [2,4]')
    plt.loglog(N_list, rmses_trans, 'p-', label='RMSE Transform [2,4]')
    plt.xlabel('Number of samples N')
    plt.ylabel('Error / RMSE')
    plt.title('Monte Carlo Integration Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig("Exercise1/template/results/task2_monte_carlo_errors24.png")
    plt.show()
    # ====================================================================
