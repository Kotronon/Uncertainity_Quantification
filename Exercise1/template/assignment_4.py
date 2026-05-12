from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator

from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import time

def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================
    with open(filename, "r") as file:
        lines = file.readlines()
        mean = float(lines[0].strip())
        var = float(lines[1].strip())
    # ====================================================================
    return mean, var


def worker(model_kwargs: dict[str, float], init_cond: dict[str, float], t_grid: npt.NDArray, omegas_chunk: npt.NDArray):
    c  = model_kwargs['c']
    k  = model_kwargs['k']
    f  = model_kwargs['f']
    dt = model_kwargs['dt']

    solutions_chunk = np.zeros((len(omegas_chunk), len(t_grid)))
    for i, omega in enumerate(omegas_chunk):
        y_t  = init_cond['y_0']
        dy_t = init_cond['y_1']
        for j, t in enumerate(t_grid):
            d2y_t = f * np.cos(omega * t) - c * dy_t - k * y_t
            dy_t = dy_t + dt * d2y_t
            y_t = y_t + dt * dy_t
            solutions_chunk[i, j] = y_t
    
    return solutions_chunk


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================
    implemented_rules = ["random", "halton"]
    if rule.lower() not in implemented_rules:
        raise NotImplementedError(
            f"The rule '{rule}' is not implemented. "
            f"Supported rules are: {', '.join(f"'{r}'" for r in implemented_rules)}."
        )

    if rule.lower() == "random":
        omegas = omega_distr.sample(size=n_samples, seed=seed)
    elif rule.lower() == "halton":
        halton_points = cp.create_halton_samples(n_samples, dim=1)
        omegas = omega_distr.inv(halton_points).flatten()

    sample_solutions = np.zeros((n_samples, len(t_grid)))    
    num_threads = min(n_samples, os.cpu_count() or 1)
    omegas_chunks = np.array_split(omegas, num_threads)

    # embarrassingly parallel problem
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        # ensure correct placement of chunk in sample_solutions for reproducibility
        start = 0
        for chunk in omegas_chunks:
            end = start + len(chunk)
            future = executor.submit(worker, model_kwargs, init_cond, t_grid, chunk)
            futures.append((future, start, end))
            start = end

        for future, start, end in futures:
            sample_solutions[start:end] = future.result()
    # ====================================================================
    
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)

    mean_error = abs(1 - mean / mean_ref)
    var_error = abs(1 - var / var_ref)
    # ====================================================================
    return mean_error, var_error


def compute_solutions(t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    N: npt.NDArray,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    seed=42
    ):
    mc_solutions     = []
    halton_solutions = []
    for n in N:
        # Monte Carlo
        mc = simulate(t_grid=t_grid, omega_distr=mc_dist_omega, n_samples=n, 
            model_kwargs=model_kwargs, init_cond=init_cond, 
            rule="random", seed=seed)
        # Quasi Monte Carlo
        halton = simulate(t_grid=t_grid, omega_distr=mc_dist_omega, n_samples=n, 
            model_kwargs=model_kwargs, init_cond=init_cond, 
            rule="halton", seed=seed)
        
        mc_solutions.append(mc)
        halton_solutions.append(halton)
    
    return mc_solutions, halton_solutions


if __name__ == "__main__":
    # TODO: define the parameters of the simulations.
    # ====================================================================
    c, k, f, y_0, y_1 = 0.5, 2.0, 0.5, 0.5, 0.0
    delta_t = 0.01
    t_grid = np.arange(0, 10 + delta_t, delta_t)

    model_kwargs = {}
    model_kwargs['c']  = c
    model_kwargs['k']  = k
    model_kwargs['f']  = f
    model_kwargs['dt'] = delta_t

    init_cond = {}
    init_cond['y_0'] = y_0
    init_cond['y_1'] = y_1

    mc_dist_omega = cp.Uniform(lower=0.95, upper=1.05)

    seed = 42
    recompute = True
    solutions_path = "./assignment_4_solutions.pkl"

    N = np.array([10**i for i in range(1, 5)])
    # ====================================================================

    # TODO: run the simulations.
    # ====================================================================
    if recompute:
        start = time.time()
        mc_solutions, halton_solutions = compute_solutions(t_grid=t_grid, 
            omega_distr=mc_dist_omega, N=N, model_kwargs=model_kwargs, 
            init_cond=init_cond, seed=seed)
        end = time.time()
        print(f"computation time: {end - start} seconds.")
        
        with open(solutions_path, 'wb') as f:
            pickle.dump({'mc_solutions': mc_solutions, 'halton_solutions': halton_solutions}, f)
    else:
        with open(solutions_path, 'rb') as f:
            data = pickle.load(f)
        mc_solutions = data['mc_solutions']
        halton_solutions = data['halton_solutions']
    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
    mean_ref, var_ref = load_reference(filename="./data/oscillator_ref.txt")
    result_mc = [arr[:, 1000] for arr in mc_solutions]
    result_halton = [arr[:, 1000] for arr in halton_solutions]

    errors_mc = [compute_errors(samples=s, mean_ref=mean_ref, var_ref=var_ref) for s in result_mc]
    errors_halton = [compute_errors(samples=s, mean_ref=mean_ref, var_ref=var_ref) for s in result_halton]
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    mean_error_mc, var_error_mc = zip(*errors_mc)
    mean_error_halton, var_error_halton = zip(*errors_halton)

    plt.loglog(N, mean_error_mc, 'o-', label='relative mean error Monte Carlo')
    plt.loglog(N, var_error_mc, 'o-', label='relative variance error Monte Carlo')
    plt.loglog(N, mean_error_halton, 'o-', label='relative mean error Quasi Monte Carlo (Halton)')
    plt.loglog(N, var_error_halton, 'o-', label='relative variance error Quasi Monte Carlo (Halton)')
    
    plt.xlabel("Number of Samples (N)", fontsize=20)
    plt.ylabel("Relative Error", fontsize=20)
    plt.title("Relative Errors Compared to Reference (loglog)", fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True, which="both")
    plt.show()
    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    for i in range(mc_solutions[0].shape[0]):
        plt.plot(t_grid, mc_solutions[0][i], label=f"Trajectory {i}")
    
    plt.xlabel("Time t", fontsize=20)
    plt.ylabel("Height y", fontsize=20)
    plt.title("10 Trajectories", fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=20)

    plt.show()
    # ====================================================================
