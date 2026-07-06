from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess


def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")
    axes[1].plot(t_grid, eigenfunctions(t_grid))
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    return fig


if __name__ == "__main__":
    # Configuration from the worksheet.
    T = 1.0
    n_points = 1000
    t_grid = np.linspace(0, T, n_points)
    Ms = [10, 100, 1000]
    seed = 42
    n_samples = 1
    rng = np.random.default_rng(seed)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    wiener = WienerProcess(mu=0.0, T=T, n_points=n_points)

    # Generate one realization of the Wiener process using the standard definition.
    standard_sample = wiener.generate(n_samples=n_samples, rng=rng)

    # Generate KL approximations. Reusing the same seed keeps the first KL
    # coefficients comparable for different truncation levels.
    kl_samples = {
        M: wiener.approximate_kl(
            n_samples=n_samples, M=M, rng=np.random.default_rng(seed)
        )
        for M in Ms
    }

    # Plot approximation results.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_grid, standard_sample[0], label="definition", c="black", lw=1.8)
    for M, samples in kl_samples.items():
        ax.plot(t_grid, samples[0], label=f"KL M={M}", lw=1.2)
    ax.set_title("Wiener process: definition vs. KL approximations")
    ax.set_xlabel("t")
    ax.set_ylabel("W(t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "assignment_3.1_wiener_kl_comparison.png", dpi=200)

    # Visualize eigenvalues for M=1000.
    eigenvalues = wiener.kl_eigenvalues(1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues)
    ax.set_yscale("log")
    ax.set_title("First 1000 KL eigenvalues")
    ax.set_xlabel("m")
    ax.set_ylabel("lambda_m")
    fig.tight_layout()
    fig.savefig(results_dir / "assignment_3.1_eigenvalues.png", dpi=200)

    # Visualize the first eigenfunctions without overplotting all 1000 modes.
    fig = plot_eigenpairs(wiener, n_terms=5, t_grid=t_grid)
    fig.tight_layout()
    fig.savefig(results_dir / "assignment_3.1_first_eigenpairs.png", dpi=200)

    print(f"Saved plots to {results_dir}")
