from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: float | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.T is None and self.n_points is None:
            self.T = self.t_grid[-1]
            self.n_points = len(self.t_grid)
        if self.t_grid is None:
            self.t_grid = np.linspace(0, self.T, self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):

        # TODO: generate n_samples realizations of the Wiener process
        # using the standard definition.
        return np.zeros((n_samples, self.n_points))

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):

        # TODO: generate n_samples realizations of the Wiener process
        # using the Karhunen-Loève expansion with M terms.
        return np.zeros((n_samples, self.n_points))


    def kl_eigenvalues(self, M: int):

        # TODO: compute the first M eigenvalues of the Wiener process.
        return np.zeros(M)

    def kl_eigenfunctions(self, M: int):

        # TODO: compute the first M eigenfunctions of the Wiener process.
        # It might be more conveniet to return a callable function that
        # returns evaluations of the first M eigenfunctions for the provided
        # time points.
        return lambda _: np.zeros(M)

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions
