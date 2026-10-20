r"""Fitting Bézier simploids and simploidal Bézier spaces to data.

Both models are linear in a single parameter matrix (the control points, or the proper control points
of a space): ``model(*inputs) == model.design_matrix(*inputs) @ parameter``.  The solvers below rely
on this interface only, so every model is fitted by the same code.
"""

from typing import Any, Literal, Sequence, TypeAlias, cast

import lightning.pytorch as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_bsf.simploidal.bezier import BezierSimploid
from torch_bsf.simploidal.combinatorics import _check_degree, _check_type, simploid_n_params
from torch_bsf.simploidal.sets import Simploid, SimploidalSet
from torch_bsf.simploidal.space import DegreeAssignment, SimploidalBezierSpace

Solver = Literal["lstsq", "trainer"]


def _solve_least_squares(a: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    r"""Minimizes :math:`\|A X - Y\|^2 + \mathrm{ridge} \|X\|^2` in double precision (minimum-norm if singular)."""
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, but {ridge} is given.")
    a64, y64 = a.detach().cpu().double(), y.detach().cpu().double()
    if ridge > 0:
        lhs = a64.T @ a64 + ridge * torch.eye(a64.shape[1], dtype=torch.float64)
        return cast(torch.Tensor, torch.linalg.solve(lhs, a64.T @ y64))
    return cast(torch.Tensor, torch.linalg.lstsq(a64, y64, driver="gelsd").solution)


#: A model whose output is linear in its only parameter: ``model(*inputs) == model.design_matrix(*inputs) @ parameter``.
_LinearBezierModel: TypeAlias = BezierSimploid | SimploidalBezierSpace


def _parameter(model: _LinearBezierModel) -> nn.Parameter:
    (parameter,) = model.parameters()
    return parameter


class _LightningWrapper(L.LightningModule):
    """Trains a linear Bézier model with :class:`lightning.pytorch.Trainer` by minimizing the mean squared error.

    A batch is ``(*inputs, values)``, where ``inputs`` are the arguments of the model's ``forward``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _predict_batch(self, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        *inputs, y = batch
        return self.model(*inputs), y

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("train_mse", mse, sync_dist=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx) -> None:
        y_hat, y = self._predict_batch(batch)
        self.log("val_mse", F.mse_loss(y_hat, y), sync_dist=True)
        self.log("val_mae", F.l1_loss(y_hat, y), sync_dist=True)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", F.l1_loss(y_hat, y), sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.model.parameters())


def _check_solver(solver: str, ridge: float) -> None:
    if solver not in ("lstsq", "trainer"):
        raise ValueError(f"Unknown solver {solver!r}.")
    if ridge and solver != "lstsq":
        raise ValueError("ridge is supported only by the 'lstsq' solver.")


def _fit(
    model: _LinearBezierModel,
    inputs: Sequence[torch.Tensor],
    values: torch.Tensor,
    solver: Solver,
    ridge: float,
    batch_size: int | None,
    kwargs: dict[str, Any],
) -> None:
    """Fits the parameter of a linear Bézier model to ``values`` at ``inputs`` in place."""
    _check_solver(solver, ridge)
    parameter = _parameter(model)
    if solver == "lstsq":
        # Build the design matrix on the CPU in double precision, whatever the dtype of the model;
        # move the inputs first, since some devices (e.g., MPS) do not support float64.
        design = model.design_matrix(*(x.cpu().double() if x.is_floating_point() else x.cpu() for x in inputs))
        solution = _solve_least_squares(design, values, ridge)
        with torch.no_grad():
            parameter.copy_(solution.to(parameter))
        return
    dataset = TensorDataset(*inputs, values.to(parameter.dtype))
    dl = DataLoader(dataset, batch_size=batch_size or len(dataset))
    L.Trainer(**kwargs).fit(_LightningWrapper(model), dl)


def _floating_dtype(tensors: Sequence[torch.Tensor]) -> torch.dtype:
    """The promoted floating dtype of ``tensors``, or the default dtype if none is floating."""
    dtypes = [t.dtype for t in tensors if t.is_floating_point()]
    if not dtypes:
        return torch.get_default_dtype()
    dtype = dtypes[0]
    for d in dtypes[1:]:
        dtype = torch.promote_types(dtype, d)
    return dtype


def _check_values(params: torch.Tensor, values: torch.Tensor, n_params: int) -> None:
    if params.dim() != 2 or params.shape[1] != n_params:
        raise ValueError(f"params must have shape (n, {n_params}), but {tuple(params.shape)} is given.")
    if values.dim() != 2 or values.shape[0] != params.shape[0]:
        raise ValueError(f"values must have shape ({params.shape[0]}, n_values), but {tuple(values.shape)} is given.")


def fit_simploid(
    params: torch.Tensor,
    values: torch.Tensor,
    simploid_type: Sequence[int],
    degree: int | Sequence[int],
    *,
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> BezierSimploid:
    r"""Fits a Bézier simploid to data.

    The control points minimize the squared error
    :math:`\sum_k \|B(\mathbf t_k) - \mathbf x_k\|^2`.

    Parameters
    ----------
    params
        Parameters of shape ``(n, sum(a_i + 1))``.
    values
        Values of shape ``(n, n_values)``.
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer` from random control points.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` (``"lstsq"`` only).
        Without it, under-determined problems get the minimum-norm solution.
    batch_size
        Minibatch size for ``"trainer"`` (full batch by default).
    seed
        Random seed passed to :func:`lightning.pytorch.seed_everything` (``"trainer"`` only).
    kwargs
        Arguments for :class:`lightning.pytorch.Trainer` (``"trainer"`` only).

    Returns
    -------
    The fitted Bézier simploid.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.simploidal import fit_simploid, sample_simploid
    >>> t = sample_simploid((2, 1), 100, seed=0)
    >>> x = t[:, :3] * t[:, 3:4]  # a function on a prism
    >>> bs = fit_simploid(t, x, (2, 1), degree=(1, 1))
    >>> torch.allclose(bs(t), x, atol=1e-5)
    True
    """
    st = _check_type(simploid_type)
    ds = _check_degree(degree, len(st))
    _check_values(params, values, simploid_n_params(st))
    _check_solver(solver, ridge)
    if solver == "trainer" and seed is not None:
        L.seed_everything(seed)
    n_values = int(values.shape[1])
    model = BezierSimploid.randn(st, ds, n_values) if solver == "trainer" else BezierSimploid.zeros(st, ds, n_values)
    dtype = _floating_dtype([values] if values.is_floating_point() else [params])
    model = model.to(dtype)
    _fit(model, [params], values, solver, ridge, batch_size, kwargs)
    return model


def _normalize_data(
    space: SimploidalBezierSpace, data: Sequence[tuple[Simploid, torch.Tensor, torch.Tensor]]
) -> list[tuple[Simploid, torch.Tensor, torch.Tensor]]:
    out = []
    for simploid, params, values in data:
        space.simploid_id(simploid)
        params, values = torch.as_tensor(params), torch.as_tensor(values)
        _check_values(params, values, simploid_n_params(simploid.simploid_type))
        if values.shape[1] != space.n_values:
            raise ValueError(f"All values must have {space.n_values} columns, but {values.shape[1]} is given.")
        out.append((simploid, params, values))
    return out


def fit_simploidal_set(
    simploidal_set: SimploidalSet,
    data: Sequence[tuple[Simploid, torch.Tensor, torch.Tensor]],
    degree: DegreeAssignment,
    *,
    default_degree: int | None = None,
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> SimploidalBezierSpace:
    r"""Fits a simploidal Bézier space to data on its simploids.

    Each sample is attached to a non-degenerate simploid (of any dimension)
    and given in that simploid's parameters.  Since simploids sharing a face
    share its control points, all pieces are fitted jointly and the result is
    continuous across shared faces.  The proper control points minimize the
    total squared error over all samples.

    Parameters
    ----------
    simploidal_set
        The simploidal set.
    data
        A sequence of ``(simploid, params, values)``, where ``params`` has shape
        ``(n, sum(a_i + 1))`` for the type of ``simploid`` and ``values`` has shape ``(n, n_values)``.
        A simploid may appear several times.
    degree
        The degree of every factor of every simploid (at least 1), or a mapping from
        non-degenerate simploids to their degrees; see :class:`SimploidalBezierSpace`.
    default_degree
        The degree of the factors left undetermined by a ``degree`` mapping.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer` from zero control points.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` on the proper control points
        (``"lstsq"`` only).  Without it, under-determined problems get the minimum-norm solution.
    batch_size
        Minibatch size for ``"trainer"`` (full batch by default).
    seed
        Random seed passed to :func:`lightning.pytorch.seed_everything` (``"trainer"`` only).
    kwargs
        Arguments for :class:`lightning.pytorch.Trainer` (``"trainer"`` only).

    Returns
    -------
    The fitted simploidal Bézier space.

    Examples
    --------
    Fit a piecewise model on a triangle glued to a square, with data that is
    continuous across the shared edge:

    >>> import torch
    >>> from torch_bsf.simploidal import SimploidalSet, fit_simploidal_set, sample_simploid
    >>> S = SimploidalSet()
    >>> tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    >>> _ = S.identify(tri.face(0, 2), sq.face(1, 1))
    >>> t1, t2 = sample_simploid((2,), 30, seed=0), sample_simploid((1, 1), 30, seed=1)
    >>> x1 = t1[:, 0:1] - t1[:, 1:2]  # the shared edge is (w0, w1, 0) on the triangle
    >>> x2 = (t2[:, 0:1] - t2[:, 1:2]) * t2[:, 2:3]  # and (w0, w1, 1, 0) on the square
    >>> space = fit_simploidal_set(S, [(tri, t1, x1), (sq, t2, x2)], degree=2)
    >>> torch.allclose(space.evaluate(tri, t1), x1, atol=1e-5)
    True

    The same with a quadratic triangle and a square that is cubic across the
    shared edge (the degree along the shared edge is common to both pieces):

    >>> space = fit_simploidal_set(S, [(tri, t1, x1), (sq, t2, x2)], degree={tri: 2, sq: (2, 3)})
    >>> space.simploid_degree(sq)
    (2, 3)
    """
    if not data:
        raise ValueError("data must not be empty.")
    n_values = int(torch.as_tensor(data[0][2]).shape[1])
    space = SimploidalBezierSpace(simploidal_set, degree, n_values, default_degree=default_degree)
    samples = _normalize_data(space, data)
    dtype = _floating_dtype([v for _, _, v in samples])
    space = space.to(dtype)
    _check_solver(solver, ridge)
    if solver == "trainer" and seed is not None:
        L.seed_everything(seed)
    width = max(p.shape[1] for _, p, _ in samples)
    ids = torch.cat([torch.full((len(p),), space.simploid_id(s), dtype=torch.long) for s, p, _ in samples])
    params_dtype = _floating_dtype([p for _, p, _ in samples])
    params = torch.cat([F.pad(p.to(params_dtype), (0, width - p.shape[1])) for _, p, _ in samples])
    values = torch.cat([v.to(dtype) for _, _, v in samples])
    _fit(space, [params, ids], values, solver, ridge, batch_size, kwargs)
    return space
