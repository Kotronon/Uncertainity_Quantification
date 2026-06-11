import chaospy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import numpy.typing as npt

def calculate_polynoms_product(distr: cp.Distribution, n: int) -> npt.NDArray:
    # TODO: generate the first n orthonormal polynomials w.r.t. the given
    # distribution and compute the expected value of their inner products.
    # ====================================================================
    arr = cp.generate_expansion(order=n, dist=distr, normed=True)
    results = np.array([[float(cp.E(phi_i * phi_j, distr)) for phi_j in arr] for phi_i in arr])
    # ====================================================================
    return results


def visualize(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(
        results,
        cmap="inferno",
        norm=SymLogNorm(
            linthresh=1e-12,
            vmin=results.min(),
            vmax=results.max(),
        )
    )

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            value = abs(results[i, j])

            if value == 0:
                label = "0"
            else:
                exponent = int(np.floor(np.log10(value)))
                label = rf"$10^{{{exponent}}}$"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white"
            )

    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================
    lower = -1.0
    upper = 1.0
    rho_1 = cp.Uniform(lower=lower, upper=upper)

    mu = 5
    sigma = 1
    rho_2 = cp.Normal(mu=mu, sigma=sigma)

    N = 10
    I_N = np.eye(N + 1)

    eps = 1e-2
    # ====================================================================

    # Compute the inner products.
    # ====================================================================
    products_U = calculate_polynoms_product(distr=rho_1, n=N) - I_N
    products_N = calculate_polynoms_product(distr=rho_2, n=N) - I_N

    assert np.all(np.abs(products_U) < eps)
    assert np.all(np.abs(products_N) < eps)
    # ====================================================================

    # Visualize the results.
    # ====================================================================
    visualize(products_U)
    visualize(products_N)
    # ====================================================================
