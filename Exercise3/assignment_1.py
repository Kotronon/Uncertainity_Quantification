import time

import chaospy as cp
import numpy as np

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol

import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    # TODO: create a joint distribution over the stochastic parameters.
    c_uniform = cp.Uniform(c_lims[0], c_lims[1])
    k_uniform = cp.Uniform(k_lims[0], k_lims[1])
    f_uniform = cp.Uniform(f_lims[0], f_lims[1])
    y0_uniform = cp.Uniform(y0_lims[0], y0_lims[1])
    y1_uniform = cp.Uniform(y1_lims[0], y1_lims[1])

    dists = [c_uniform, k_uniform, f_uniform, y0_uniform, y1_uniform]

    joint_uniform = cp.J(*dists)
    
    return joint_uniform


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.
    if method == "pseudo_spectral":
        return pseudo_spectral_sobol(**kwargs)
    elif method == "monte_carlo":
        return monte_carlo_sobol(**kwargs)
    else:
        raise NotImplementedError(f"The passed method: {method} is not supported.")


if __name__ == "__main__":
    recompute = False
    # TODO: set the stochastic parameters.
    c_lims = [8e-2, 12e-2]
    k_lims = [3e-2, 4e-2]
    f_lims = [8e-2, 12e-2]
    y0_lims = [45e-2, 55e-2]
    y1_lims = [-5e-2, 5e-2]

    # TODO: set the determinisic parameters.
    fixed_args = {'omega': 1.0, 'method': 'odeint'}

    # TODO: set the parameters of the methods.
    quadrature_degree = [3, 4] # K
    pce_degree = [3, 4] # N
    n_samples = [1024, 3125]

    # TODO: set the time domain
    T_min = 0
    T_max = 10
    dt = 0.01
    t_grid = np.arange(T_min, T_max + dt, dt)

    ###########################################################################
    
    results = {}

    if recompute:
        # TODO: define the distribution over the stochastic parameters.
        joint_uniform = get_distribution(c_lims=c_lims, k_lims=k_lims, f_lims=f_lims, y0_lims=y0_lims, y1_lims=y1_lims)

        # TODO: run the pseudo-spectral method on full grid.
        for quad_deg in quadrature_degree:
            for pce_deg in pce_degree:
                print(f"Running Pseudo Spectral Sobol with:\nQuadrature Degree = {quad_deg}\nPCE Degree = {pce_deg}\nSparse = {False}\n")
                results[f'Pseudo Spectral K={quad_deg} N={pce_deg} sparse={False}'] = run_method(
                    method="pseudo_spectral", 
                    pce_degree=pce_deg, 
                    quadrature_degree=quad_deg, 
                    distribution=joint_uniform, 
                    t_grid=t_grid, 
                    fixed_args=fixed_args, 
                    sparse=False,
                )
        # TODO: run the pseudo-spectral method on sparse grid.
        for quad_deg in quadrature_degree:
            for pce_deg in pce_degree:
                print(f"Running Pseudo Spectral Sobol with:\nQuadrature Degree = {quad_deg}\nPCE Degree = {pce_deg}\nSparse = {True}\n")
                results[f'Pseudo Spectral K={quad_deg} N={pce_deg} sparse={True}'] = run_method(
                    method="pseudo_spectral", 
                    pce_degree=pce_deg, 
                    quadrature_degree=quad_deg, 
                    distribution=joint_uniform, 
                    t_grid=t_grid, 
                    fixed_args=fixed_args, 
                    sparse=True,
                )

        # TODO: run the Monte Carlo method.
        for n in n_samples:
            print(f"Running Monte Carlo Sobol with: n_samples = {n}\n")
            results[f'Monte Carlo N={n}'] = run_method(
                method="monte_carlo",
                n_samples=n,
                distribution=joint_uniform,
                t_grid=t_grid,
                fixed_args=fixed_args,
            )
        
        # store results
        with open("./cache/assignment1.pkl", "wb") as file:
            pickle.dump(results, file)

    else:
        with open("./cache/assignment1.pkl", "rb") as file:
            results = pickle.load(file)
    
    print(results)

    # using pseudo spectral as reference for error
    reference = results["Pseudo Spectral K=4 N=4 sparse=False"]

    names = ['c', 'k', 'f', 'y0', 'y1']

    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.5*len(results)), sharex=True)

    for ax, (label, (first, total)) in zip(axes, results.items()):
        x = np.arange(len(names))
        w = 0.35

        ax.bar(x - w / 2, first, w, label="First")
        ax.bar(x + w / 2, total, w, label="Total")

        ax.set_title(label)
        ax.set_ylim(0, 1)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names)
    axes[0].legend()

    plt.tight_layout()

    fig.savefig("./cache/sobol_indices.png", dpi=300, bbox_inches="tight")
    
    errors_first = []
    errors_total = []
    labels = []

    for label, (first, total) in results.items():
        labels.append(label)
        errors_first.append(np.linalg.norm(first - reference[0]))
        errors_total.append(np.linalg.norm(total - reference[1]))

    
    x = np.arange(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))

    bars1 = ax.bar(x - w / 2, errors_first, w, label="First")
    bars2 = ax.bar(x + w / 2, errors_total, w, label="Total")


    def add_labels(bars):
        for b in bars:
            height = b.get_height()
            ax.annotate(
                f"{height:.2e}",
                xy=(b.get_x() + b.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )

    add_labels(bars1)
    add_labels(bars2)

    y_max = max(
        max(errors_first),
        max(errors_total)
    )

    ax.set_ylim(0, y_max * 1.55)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"$L_2$ error")
    ax.legend()

    fig.tight_layout()
    fig.savefig("./cache/errors.png", dpi=300, bbox_inches="tight")

    ###########################################################################
