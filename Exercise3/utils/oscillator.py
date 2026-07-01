from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import odeint


@dataclass
class Oscillator:
    """Model to simulate a dumped oscillator.

    Attributes
    ----------
    c: float
        Dumping coefficient.
    k: float
        Spring coefficient.
    f_t: float
        Time-dependent forcing amplitude. One can pass a constant to set the
        constant amplitde over time.
    omega: float
        Frequency.
    """

    c: float
    k: float
    f: Callable[[float], float] | float
    omega: float

    def __post_init__(self):
        if isinstance(self.f, float):
            # If f_t is a constant, we convert it to a callable.
            f = self.f
            self.f = lambda _: f

    def discretize(
        self,
        method: str,
        y0: float,
        y1: float,
        t_grid: npt.NDArray[np.float64],
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ) -> npt.NDArray[np.float64]:
        """Discretizes the oscillator according to provided initial conditions.

        Arguments
        ---------
        method: str, "euler" or "odeint"
            Method for discretizing the model.y0: float
            Initial position.
        y1: float
            Initial velocity.
        t_grid: npt.NDArray[np.float64]
            A grid of time points to evaluate the solution at.
        atol: float
            Absolute tolerance for discretization.
        rtol: float
            Relative tolerance for discretization.

        Returns
        -------
        positions: npt.NDArray[np.float64]
            An array of positions at discretized time points.
        """
        match method:
            case "odeint":
                return self._discretize_odeint(y0, y1, t_grid, atol, rtol)
            case "euler":
                return self._discretize_euler(y0, y1, t_grid, atol, rtol)
            case _:
                raise ValueError(f"Unknown method {method} for discretizing the model.")

    def _discretize_odeint(
        self,
        y0: float,
        y1: float,
        t_grid: npt.NDArray[np.float64],
        atol: float,
        rtol: float,
    ) -> npt.NDArray[np.float64]:
        solution = odeint(self._model, (y0, y1), t_grid, atol=atol, rtol=rtol)
        # We only return positions and not velocities.
        return solution[:, 0]

    def _discretize_euler(
        self,
        y0: float,
        y1: float,
        t_grid: npt.NDArray[np.float64],
        _: float,
        __: float,
    ) -> npt.NDArray[np.float64]:
        n_steps = len(t_grid)

        # Initialize the solution vectors.
        positions = np.zeros(n_steps)
        positions[0] = y0
        velocities = np.zeros(n_steps)
        velocities[0] = y1

        # Run the Euler scheme for the provided grid.
        for i in range(0, n_steps - 1):
            dt = t_grid[i + 1] - t_grid[i]
            d_position, d_velocity = self._model(
                (positions[i], velocities[i]), t_grid[i]
            )
            positions[i + 1] = positions[i] + dt * d_position
            velocities[i + 1] = velocities[i] + dt * d_velocity

        return positions

    def _model(self, state: tuple[float, float], t: float) -> tuple[float, float]:
        position, velocity = state
        d_position = velocity
        d_velocity = (
            self.f(t) * np.cos(self.omega * t) - self.k * position - self.c * velocity
        )
        return d_position, d_velocity
