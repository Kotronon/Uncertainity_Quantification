import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True)
class Interpolator(abc.ABC):
    nodes: npt.NDArray
    values: npt.NDArray

    def __post_init__(self):
        self.fit()

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        pass


class StandardLagrange(Interpolator):
    def fit(self):
        pass

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        # TODO: implement the standard Lagrange interpolation.
        # ================================================================
        n_nodes = self.nodes.shape[0]
        # Compute the differences in the numerator and denominator of
        # the Lagrange polynomials.
        num_diffs = x[:, None, None] - self.nodes[None, None, :]
        num_diffs = np.tile(num_diffs, (1, n_nodes, 1))
        denom_diffs = self.nodes[None, :, None] - self.nodes[None, None, :]
        denom_diffs = np.tile(denom_diffs, (x.shape[0], 1, 1))
        # Set the diagonal elements (j=i) to one to ''skip'' these terms.
        diag_i, diag_j = np.diag_indices(n_nodes)
        num_diffs[:, diag_i, diag_j] = 1
        denom_diffs[:, diag_i, diag_j] = 1
        # Compute the Lagrange polynomial and the final interpolation.
        lagrange_poly = np.prod(num_diffs / denom_diffs, axis=2)
        output = np.sum(lagrange_poly * self.values, axis=1)
        # ================================================================
        return output


class FirstBarycentricLagrange(Interpolator):
    weights: npt.NDArray

    def fit(self):
        # TODO: implement the barycentric Lagrange interpolation.
        # ================================================================
        diffs = self.nodes[:, None] - self.nodes[None, :]
        diffs[np.diag_indices_from(diffs)] = 1
        self.weights = 1 / np.prod(diffs, axis=1)
        # ================================================================

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        # TODO: implement the barycentric Lagrange interpolation.
        # ================================================================
        nodes_mask, nodes_idxs = self._get_filter_mask(x)
        output = np.zeros_like(x)
        # If one of the evaluation points is close to the nodes,
        # we need to set the output to the value of this node.
        output[nodes_mask] = self.values[nodes_idxs[nodes_mask]]

        # In the following, we can rely on the fact that the evaluation points
        # are not close to the nodes.
        x_clean = x[~nodes_mask]
        diffs = x_clean[:, None] - self.nodes[None, :]
        L = np.prod(diffs, axis=1)
        sum_term = self.values[None, :] * self.weights[None, :] / diffs
        output[~nodes_mask] = L * np.sum(sum_term, axis=1)
        # ================================================================
        return output

    def _get_filter_mask(
        self, x: npt.NDArray, eps: float = 1e-10
    ) -> tuple[npt.NDArray, npt.NDArray]:
        # Get a mask of the evaluation points that are close to the nodes.
        diffs = np.abs(x[:, None] - self.nodes[None, :]) < eps
        mask = np.any(diffs, axis=1)
        idxs = np.argmax(diffs, axis=1)
        return mask, idxs
