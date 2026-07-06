from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: int | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.t_grid is None:
            if self.T is None or self.n_points is None:
                raise ValueError("Either t_grid or both T and n_points must be provided.")
            self.t_grid = np.linspace(0, self.T, self.n_points)
        else:
            self.t_grid = np.asarray(self.t_grid, dtype=float)
            self.T = float(self.t_grid[-1]) if self.T is None else self.T
            self.n_points = len(self.t_grid)

        self.T = float(self.T)
        self.n_points = int(self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):
        """Generates Wiener process realizations using independent increments."""
        dt = np.diff(self.t_grid, prepend=0.0)
        if np.any(dt < 0):
            raise ValueError("t_grid must be sorted in ascending order.")

        increments = rng.normal(
            loc=0.0, scale=np.sqrt(dt), size=(n_samples, self.n_points)
        )
        increments[:, 0] = 0.0
        return self.mu + np.cumsum(increments, axis=1)

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):
        """Generates Wiener process realizations using M KL terms."""
        eigenvalues, eigenfunctions = self.kl_eigenpairs(M)
        xi = rng.normal(loc=0.0, scale=1.0, size=(n_samples, M))
        basis = eigenfunctions(self.t_grid).T
        coefficients = xi * np.sqrt(eigenvalues)
        return self.mu + np.einsum("sm,mt->st", coefficients, basis, optimize=True)

    def kl_eigenvalues(self, M: int):
        """Computes the first M eigenvalues of the Wiener process covariance."""
        modes = np.arange(M) + 0.5
        return self.T**2 / (modes**2 * np.pi**2)

    def kl_eigenfunctions(self, M: int):
        """Returns a callable that evaluates the first M eigenfunctions."""
        modes = np.arange(M) + 0.5

        def eigenfunctions(t: npt.NDArray | float) -> npt.NDArray:
            t = np.asarray(t, dtype=float)
            values = np.sqrt(2.0 / self.T) * np.sin(
                np.outer(t.ravel(), modes * np.pi / self.T)
            )
            if t.ndim == 0:
                return values[0]
            return values.reshape(*t.shape, M)

        return eigenfunctions

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions
