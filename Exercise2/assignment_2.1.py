import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm



if __name__ == "__main__":
    mu = 0
    sigma = 1
    order = 8
    normal = cp.Normal(mu=mu, sigma=sigma)
    arr = cp.generate_expansion(order=order, dist=normal, normed=True)
    results = np.array([[float(cp.E(phi_i * phi_j, normal)) for phi_j in arr] for phi_i in arr])


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
            ax.text(
                j,
                i,
                f"{results[i,j]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.show()
