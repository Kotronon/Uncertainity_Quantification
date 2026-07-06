import time
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess


def generate_f_samples(
    mu: float,
    t_grid: npt.NDArray,
    n_samples: int,
    M: int | None,
    rng: np.random.Generator,
) -> list[Callable[[float], float]]:
    """Generates samples of the Wiener process."""
    wiener = WienerProcess(mu=mu, t_grid=t_grid)
    if M is None:
        samples = wiener.generate(n_samples=n_samples, rng=rng)
    else:
        samples = wiener.approximate_kl(n_samples=n_samples, M=M, rng=rng)

    return [partial(np.interp, xp=t_grid, fp=sample) for sample in samples]


def simulate(
    t_grid: npt.NDArray,
    f_samples: list[Callable[[float], float]],
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""
    solutions = np.zeros((len(f_samples), len(t_grid)))
    for i, f_sample in enumerate(f_samples):
        model = Oscillator(f=f_sample, **model_kwargs)
        solutions[i] = model.discretize(
            "odeint", t_grid=t_grid, atol=atol, rtol=rtol, **init_cond
        )
    return solutions


def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""
    ddof = 1 if solutions.shape[0] > 1 else 0
    return np.mean(solutions, axis=0), np.std(solutions, axis=0, ddof=ddof)


def plot_solutions(
    t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # Parameters from the worksheet.
    f_mean = 0.5
    model_kwargs = {"c": 0.5, "k": 2.0, "omega": 1.0}
    init_cond = {"y0": 0.5, "y1": 0.0}

    # Time domain.
    T_max = 10.0
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    # Monte Carlo samples and KL terms.
    N = 1000
    Ms = [5, 10, 100]
    seed = 42
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    ###########################################################################

    # Generate f samples and simulate the oscillator for each sampler.
    sampler_terms = {"Wiener definition": None} | {f"KL M={M}": M for M in Ms}
    sampler_solutions = {}
    rows = []
    for name, M in sampler_terms.items():
        start_time = time.perf_counter()
        f_samples = generate_f_samples(
            mu=f_mean,
            t_grid=t_grid,
            n_samples=N,
            M=M,
            rng=np.random.default_rng(seed),
        )
        solutions = simulate(
            t_grid=t_grid,
            f_samples=f_samples,
            model_kwargs=model_kwargs,
            init_cond=init_cond,
        )
        elapsed_time = time.perf_counter() - start_time
        sampler_solutions[name] = solutions
        rows.append(
            (
                name,
                np.mean(solutions[:, -1]),
                np.var(solutions[:, -1], ddof=1),
                elapsed_time,
            )
        )

    print("method              mean y(10)      variance y(10)   time [s]")
    print("-" * 64)
    for name, mean_T, var_T, elapsed_time in rows:
        print(f"{name:<18} {mean_T:>12.6f} {var_T:>17.6f} {elapsed_time:>10.2f}")

    fig = plot_solutions(t_grid=t_grid, sampler_solutions=sampler_solutions)
    fig.tight_layout()
    fig.savefig(results_dir / "assignment_3.2_oscillator_solutions.png", dpi=200)
    print(f"Saved plot to {results_dir / 'assignment_3.2_oscillator_solutions.png'}")
