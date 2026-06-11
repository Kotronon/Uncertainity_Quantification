import time
from pathlib import Path

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.helpers import compute_errors, generate_grid, load_reference, simulate
from utils.interpolation import FirstBarycentricLagrange


def estimate_monte_carlo(
    omega_distr: cp.Distribution,
    n_samples: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    # ====================================================================
    omega_samples = omega_distr.sample(n_samples, seed=42)
    solutions = simulate(
        np.array([0.0, target_t]), omega_samples, model_kwargs, init_cond
    )[:, -1]
    # ====================================================================
    return solutions


def fit_lagrange(
    omega_bounds: tuple[float, float],
    n_nodes: int,
    target_t: float,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> FirstBarycentricLagrange:
    # ====================================================================
    nodes = generate_grid(omega_bounds, n_nodes, grid_type="chebyshev")
    values = simulate(
        np.array([0.0, target_t]), nodes, model_kwargs, init_cond
    )[:, -1]
    interpolator = FirstBarycentricLagrange(nodes=nodes, values=values)
    # ====================================================================
    return interpolator
    


def evaluate_pce(
    interpolator: FirstBarycentricLagrange, omega_distr: cp.Distribution, n_samples: int
) -> npt.NDArray:
    # ====================================================================
    omega_samples = np.asarray(omega_distr.sample(n_samples, seed=42)).reshape(-1)
    solutions = interpolator.evaluate(omega_samples)
    # ====================================================================
    return solutions


if __name__ == "__main__":
    # ====================================================================
    target_t = 10.0
    omega_bounds = (0.95, 1.05)
    omega_distr = cp.Uniform(*omega_bounds)
    model_kwargs = {"c": 0.5, "k": 2.0, "f": 0.5}
    init_cond = {"y0": 0.5, "y1": 0.0}
    node_counts = np.array([2, 5, 10, 20])
    sample_counts = np.array([10, 100, 1_000, 10_000])
    mean_ref, var_ref = load_reference(Path(__file__).with_name("oscillator_ref.txt"))
    # ====================================================================

    # ====================================================================
    pce_solutions = {}
    pce_fit_times = np.empty(node_counts.size)
    pce_evaluation_times = np.empty((node_counts.size, sample_counts.size))
    for node_index, n_nodes in enumerate(node_counts):
        start = time.perf_counter()
        interpolator = fit_lagrange(
            omega_bounds, n_nodes, target_t, model_kwargs, init_cond
        )
        pce_fit_times[node_index] = time.perf_counter() - start

        for sample_index, n_samples in enumerate(sample_counts):
            start = time.perf_counter()
            pce_solutions[n_nodes, n_samples] = evaluate_pce(
                interpolator, omega_distr, n_samples
            )
            pce_evaluation_times[node_index, sample_index] = (
                time.perf_counter() - start
            )
    pce_total_times = pce_fit_times[:, None] + pce_evaluation_times
    # ====================================================================

    # ====================================================================
    mc_solutions = {}
    mc_times = np.empty(sample_counts.size)
    for sample_index, n_samples in enumerate(sample_counts):
        start = time.perf_counter()
        mc_solutions[n_samples] = estimate_monte_carlo(
            omega_distr, n_samples, target_t, model_kwargs, init_cond
        )
        mc_times[sample_index] = time.perf_counter() - start
    # ====================================================================

    # ====================================================================
    pce_errors = np.empty((node_counts.size, sample_counts.size, 2))
    for node_index, n_nodes in enumerate(node_counts):
        for sample_index, n_samples in enumerate(sample_counts):
            pce_errors[node_index, sample_index] = compute_errors(
                pce_solutions[n_nodes, n_samples], mean_ref, var_ref
            )

    mc_errors = np.array(
        [
            compute_errors(mc_solutions[n_samples], mean_ref, var_ref)
            for n_samples in sample_counts
        ]
    )

    print("\nLagrange surrogate")
    print(
        f"{'N':>4} {'M':>7} {'mean':>13} {'variance':>13} "
        f"{'mean error':>13} {'var error':>13} {'fit [s]':>10} "
        f"{'eval [s]':>10} {'total [s]':>10}"
    )
    for node_index, n_nodes in enumerate(node_counts):
        for sample_index, n_samples in enumerate(sample_counts):
            samples = pce_solutions[n_nodes, n_samples]
            print(
                f"{n_nodes:4d} {n_samples:7d} "
                f"{np.mean(samples):13.6e} {np.var(samples, ddof=1):13.6e} "
                f"{pce_errors[node_index, sample_index, 0]:13.6e} "
                f"{pce_errors[node_index, sample_index, 1]:13.6e} "
                f"{pce_fit_times[node_index]:10.4e} "
                f"{pce_evaluation_times[node_index, sample_index]:10.4e} "
                f"{pce_total_times[node_index, sample_index]:10.4e}"
            )

    print("\nDirect Monte Carlo")
    print(
        f"{'M':>7} {'mean':>13} {'variance':>13} "
        f"{'mean error':>13} {'var error':>13} {'runtime [s]':>12}"
    )
    for sample_index, n_samples in enumerate(sample_counts):
        samples = mc_solutions[n_samples]
        print(
            f"{n_samples:7d} {np.mean(samples):13.6e} "
            f"{np.var(samples, ddof=1):13.6e} "
            f"{mc_errors[sample_index, 0]:13.6e} "
            f"{mc_errors[sample_index, 1]:13.6e} "
            f"{mc_times[sample_index]:12.4e}"
        )
    # ====================================================================

    # ====================================================================
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    figure.set_constrained_layout_pads(wspace=0.05)
    labels = ("Mean", "Variance")
    for error_index, label in enumerate(labels):
        axis = axes[error_index]
        for node_index, n_nodes in enumerate(node_counts):
            axis.loglog(
                sample_counts,
                pce_errors[node_index, :, error_index],
                "o-",
                label=f"Lagrange N={n_nodes}",
            )
        axis.loglog(
            sample_counts,
            mc_errors[:, error_index],
            "o--",
            color="black",
            label="Monte Carlo",
        )
        axis.set_xlabel("Number of Monte Carlo samples M")
        axis.set_ylabel(f"Relative {label.lower()} error")
        axis.grid(True, which="both")
        axis.legend()

    for node_index, n_nodes in enumerate(node_counts):
        axes[2].loglog(
            sample_counts,
            pce_total_times[node_index],
            "o-",
            label=f"Lagrange N={n_nodes} (fit + evaluation)",
        )
    axes[2].loglog(
        sample_counts, mc_times, "o--", color="black", label="Monte Carlo"
    )
    axes[2].set_xlabel("Number of Monte Carlo samples M")
    axes[2].set_ylabel("Runtime [s]")
    axes[2].grid(True, which="both")
    axes[2].legend()

    plt.savefig(
        Path(__file__).with_name("oscillator_errors.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    # ====================================================================
