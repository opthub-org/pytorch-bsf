"""One weighted, constrained, augmented least-squares problem for both backends."""

from __future__ import annotations

from typing import Any

import torch

from torch_bsf.core import BezierSpace, FixedControls, Points, QuadraticPenalty, Samples


def scipy_sparse() -> Any:
    """Import the optional backend only on execution; preserve broken-install errors."""
    try:
        import scipy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise
        raise ImportError('Sparse least squares requires pip install "pytorch-bsf[sparse]".') from error
    from scipy import sparse

    return sparse


class LeastSquaresProblem:
    """Linear operator for normalized observations and the exact penalty rows."""

    def __init__(self, space: BezierSpace, samples: Samples, penalty: QuadraticPenalty, fixed: FixedControls | None):
        self.space, self.samples, self.penalty = space, samples, penalty
        self.basis = space.basis(Points(samples.points.parameters.detach(), samples.points.cell_ids).to())
        self.values = samples.values.detach().cpu().double()
        self.scale = (samples.weights.detach() / samples.weights.detach().sum()).sqrt()
        self.fixed = torch.zeros(space.n_controls, self.values.shape[1], dtype=torch.float64)
        self.free = torch.ones(space.n_controls, dtype=torch.bool)
        for row, value in (fixed.values if fixed is not None else {}).items():
            if row >= space.n_controls or value.shape != (self.values.shape[1],):
                raise ValueError("fixed control index or output dimension does not match the problem.")
            self.fixed[row] = value.detach()
            self.free[row] = False
        self.n_free = int(self.free.sum())
        self.edges = space.adjacency()
        # Penalties are stored as COO entries, not dense incidence/identity matrices.
        rows, columns, values = [], [], []
        offset = 0
        if penalty.smoothness:
            n = len(self.edges)
            rows.extend([torch.arange(n), torch.arange(n)])
            columns.extend([self.edges[:, 0], self.edges[:, 1]])
            values.extend(
                [
                    torch.full((n,), penalty.smoothness**0.5, dtype=torch.float64),
                    torch.full((n,), -(penalty.smoothness**0.5), dtype=torch.float64),
                ]
            )
            offset += n
        if penalty.ridge:
            rows.append(torch.arange(space.n_controls) + offset)
            columns.append(torch.arange(space.n_controls))
            values.append(torch.full((space.n_controls,), penalty.ridge**0.5, dtype=torch.float64))
            offset += space.n_controls
        self.reg_rows = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
        self.reg_columns = torch.cat(columns) if columns else torch.empty(0, dtype=torch.long)
        self.reg_values = torch.cat(values).double() if values else torch.empty(0, dtype=torch.float64)
        self.n_regularizers = offset
        self.shape = (len(samples) + offset, self.n_free)
        target = torch.cat([self.values * self.scale[:, None], torch.zeros(offset, self.values.shape[1], dtype=torch.float64)])
        self.rhs = target - self.full_matmat(self.fixed)

    def full_matmat(self, coefficients: torch.Tensor) -> torch.Tensor:
        data = self.basis.matmat(coefficients) * self.scale[:, None]
        reg = coefficients.new_zeros(self.n_regularizers, coefficients.shape[1])
        reg = reg.index_add(0, self.reg_rows, self.reg_values[:, None] * coefficients[self.reg_columns])
        return torch.cat([data, reg])

    def matmat(self, free_coefficients: torch.Tensor) -> torch.Tensor:
        full = free_coefficients.new_zeros(self.space.n_controls, free_coefficients.shape[1])
        full[self.free] = free_coefficients
        return self.full_matmat(full)

    def dense(self) -> torch.Tensor:
        """Materialize only when a dense solver explicitly requests it."""
        return self.matmat(torch.eye(self.n_free, dtype=torch.float64))

    def sparse(self) -> Any:
        """Assemble local basis and penalty entries directly in CSR, summing duplicate columns."""
        sparse = scipy_sparse()
        row_chunks, column_chunks, value_chunks = [], [], []
        for rows, columns, basis in self.basis.blocks:
            row_chunks.append(rows[:, None].expand(-1, len(columns)).reshape(-1))
            column_chunks.append(columns[None, :].expand(len(rows), -1).reshape(-1))
            value_chunks.append((basis * self.scale[rows, None]).reshape(-1))
        row_chunks.append(self.reg_rows + len(self.samples))
        column_chunks.append(self.reg_columns)
        value_chunks.append(self.reg_values)
        rows, columns, values = torch.cat(row_chunks), torch.cat(column_chunks), torch.cat(value_chunks)
        column_map = torch.full((self.space.n_controls,), -1, dtype=torch.long)
        column_map[self.free] = torch.arange(self.n_free)
        keep = self.free[columns] & (values != 0)
        return sparse.coo_matrix(
            (values[keep].numpy(), (rows[keep].numpy(), column_map[columns[keep]].numpy())), shape=self.shape
        ).tocsr()

    def coefficients(self, solution: torch.Tensor) -> torch.Tensor:
        full = self.fixed.clone()
        full[self.free] = solution
        return full

    def metrics(self, coefficients: torch.Tensor) -> dict[str, float]:
        residual = self.basis.matmat(coefficients) - self.values
        data_loss = (residual.square() * self.scale[:, None].square()).sum().item()
        ridge = self.penalty.ridge * coefficients.square().sum().item()
        smoothness = (
            self.penalty.smoothness * (coefficients[self.edges[:, 0]] - coefficients[self.edges[:, 1]]).square().sum().item()
        )
        return {
            "data_mse": data_loss / self.values.shape[1],
            "data_loss": data_loss,
            "ridge_penalty": ridge,
            "smoothness_penalty": smoothness,
            "objective": data_loss + ridge + smoothness,
        }
