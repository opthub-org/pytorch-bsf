r"""Fitting Bézier simploids and simploidal Bézier spaces to data.

Both models are linear in a single parameter matrix :math:`P` (the control points, or the proper control
points of a space): ``model(*inputs) == model.design_matrix(*inputs) @ P``, and both list the pairs of
adjacent rows of :math:`P` with ``model.adjacency()``.  The solvers below rely on this interface only,
so every model is fitted by the same code.

Given ``n`` samples with ``m`` values each, the design matrix :math:`A` and the targets :math:`Y`, both
solvers minimize the mean squared error plus the smoothness penalty

.. math::

   \frac{1}{nm} \|A P - Y\|^2 + w \sum_{(p, q)} \|P_p - P_q\|^2

over the pairs :math:`(p, q)` of adjacent rows, with the frozen rows of :math:`P` held fixed.
Writing the sum as :math:`\|D P\|^2` with the incidence matrix :math:`D` of the adjacency graph, whose
Laplacian is :math:`L = D^\top D`, the ``"lstsq"`` solver minimizes :math:`nm` times this objective
(plus :math:`\mathrm{ridge}\,\|P\|^2`) exactly, i.e., it solves

.. math::

   (A^\top A + nm\,w\,L + \mathrm{ridge}\,I)\,P = A^\top Y

for the free rows, as the stacked least-squares problem with matrix :math:`[A; \sqrt{nm\,w}\,D]`.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import lightning.pytorch as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_bsf.preprocessing import NormalizeType, make_scaler
from torch_bsf.simploidal.bezier import BezierSimploid, ControlPointData
from torch_bsf.simploidal.combinatorics import SimploidIndex, SimploidType, _check_degree, _check_type, simploid_n_params
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


def _smoothness_penalty(parameter: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    r"""The sum :math:`\sum_{(p, q)} \|P_p - P_q\|^2` over the pairs of adjacent rows."""
    return (parameter[adjacency[:, 0]] - parameter[adjacency[:, 1]]).pow(2).sum()


class _LightningWrapper(L.LightningModule):
    """Trains a linear Bézier model with :class:`lightning.pytorch.Trainer`.

    The training loss is the mean squared error plus ``smoothness_weight`` times the smoothness penalty.
    A batch is ``(*inputs, values)``, where ``inputs`` are the arguments of the model's ``forward``.
    """

    def __init__(self, model: nn.Module, smoothness_weight: float = 0.0):
        super().__init__()
        self.model = model
        self.smoothness_weight = smoothness_weight
        if smoothness_weight > 0:
            self.register_buffer("adjacency", cast(_LinearBezierModel, model).adjacency(), persistent=False)

    def _predict_batch(self, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        *inputs, y = batch
        return self.model(*inputs), y

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        loss = mse
        if self.smoothness_weight > 0:
            parameter = _parameter(cast(_LinearBezierModel, self.model))
            loss = mse + self.smoothness_weight * _smoothness_penalty(parameter, cast(torch.Tensor, self.adjacency))
        self.log("train_mse", mse, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> None:
        y_hat, y = self._predict_batch(batch)
        self.log("val_mse", F.mse_loss(y_hat, y), sync_dist=True)
        self.log("val_mae", F.l1_loss(y_hat, y), sync_dist=True)

    def test_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", F.l1_loss(y_hat, y), sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.model.parameters())


def _check_solver(
    solver: str, ridge: float, smoothness_weight: float, batch_size: int | None = None, kwargs: Mapping[str, Any] = {}
) -> None:
    if solver not in ("lstsq", "trainer"):
        raise ValueError(f"Unknown solver {solver!r}.")
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, but {ridge} is given.")
    if ridge and solver != "lstsq":
        raise ValueError("ridge is supported only by the 'lstsq' solver.")
    if solver == "lstsq" and (batch_size is not None or kwargs):
        options = (["batch_size"] if batch_size is not None else []) + sorted(kwargs)
        verb = "is" if len(options) == 1 else "are"
        raise ValueError(f"{', '.join(options)} {verb} supported only by the 'trainer' solver (pass solver='trainer').")
    if smoothness_weight < 0:
        raise ValueError(f"smoothness_weight must be non-negative, but {smoothness_weight} is given.")


def _incidence(adjacency: torch.Tensor, n_rows: int) -> torch.Tensor:
    """The incidence matrix ``D`` (``+1`` at ``p``, ``-1`` at ``q`` for each pair), in double precision."""
    d = torch.zeros(len(adjacency), n_rows, dtype=torch.float64)
    edges = torch.arange(len(adjacency))
    d[edges, adjacency[:, 0].cpu()] = 1.0
    d[edges, adjacency[:, 1].cpu()] = -1.0
    return d


def _solve(
    model: _LinearBezierModel,
    inputs: Sequence[torch.Tensor],
    values: torch.Tensor,
    ridge: float,
    smoothness_weight: float,
    frozen: torch.Tensor,
) -> torch.Tensor:
    """The exact minimizer of the objective (see the module docstring) over the rows not in ``frozen``."""
    # Build the design matrix on the CPU in double precision, whatever the device and dtype of the model;
    # move the inputs first, since some devices (e.g., MPS) do not support float64.
    if _parameter(model).device.type != "cpu":
        model = copy.deepcopy(model).cpu()
    current = _parameter(model).detach().double()
    free = torch.ones(len(current), dtype=torch.bool)
    free[frozen] = False
    a = model.design_matrix(*(x.cpu().double() if x.is_floating_point() else x.cpu() for x in inputs)).detach()
    y = values.detach().cpu().double() - a[:, ~free] @ current[~free]
    a = a[:, free]
    if smoothness_weight > 0:
        d = _incidence(model.adjacency(), len(current)) * (values.numel() * smoothness_weight) ** 0.5
        a = torch.cat([a, d[:, free]])
        y = torch.cat([y, -d[:, ~free] @ current[~free]])
    solution = current.clone()
    if free.any():
        solution[free] = _solve_least_squares(a, y, ridge)
    return solution


def _fit(
    model: _LinearBezierModel,
    inputs: Sequence[torch.Tensor],
    values: torch.Tensor,
    *,
    solver: Solver,
    ridge: float,
    smoothness_weight: float,
    frozen: Sequence[int],
    batch_size: int | None,
    kwargs: dict[str, Any],
) -> None:
    """Fits the parameter of a linear Bézier model to ``values`` at ``inputs`` in place.

    The rows in ``frozen`` keep their current values.
    """
    _check_solver(solver, ridge, smoothness_weight)
    parameter = _parameter(model)
    rows = torch.tensor(sorted(set(frozen)), dtype=torch.long)
    if solver == "lstsq":
        solution = _solve(model, inputs, values, ridge, smoothness_weight, rows)
        with torch.no_grad():
            parameter.copy_(solution.to(parameter))
        return
    dataset = TensorDataset(*inputs, values.to(parameter.dtype))
    dl = DataLoader(dataset, batch_size=batch_size or len(dataset))
    mask = torch.zeros(len(parameter), 1, dtype=torch.bool)
    mask[rows] = True
    # Zeroing the gradient of the frozen rows keeps them fixed under LBFGS, whose steps are built from gradients.
    # The trainer may move the model to another device, so the mask follows the gradient.
    handle = parameter.register_hook(lambda grad: grad.masked_fill(mask.to(grad.device), 0.0)) if len(rows) else None
    try:
        L.Trainer(**kwargs).fit(_LightningWrapper(cast(nn.Module, model), smoothness_weight), dl)
    finally:
        if handle is not None:
            handle.remove()


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


def _resolve_shape(
    params: torch.Tensor,
    simploid_type: Sequence[int] | None,
    degree: int | Sequence[int] | None,
    init: object,
) -> tuple[SimploidType, tuple[int, ...]]:
    """The type and degree to fit: given explicitly, taken from a Bézier simploid ``init``, or inferred.

    Without a type, ``params`` are taken as the barycentric coordinates of a simplex.
    """
    if simploid_type is None:
        if isinstance(init, BezierSimploid):
            simploid_type = init.simploid_type
        else:
            n_params = int(params.shape[1]) if params.dim() == 2 else 0
            if n_params < 2:
                raise ValueError(
                    f"params must have at least 2 columns to infer a simplex, but {tuple(params.shape)} is given; "
                    "pass simploid_type (e.g. simploid_type=() for a constant)."
                )
            simploid_type = (n_params - 1,)
    st = _check_type(simploid_type)
    if degree is None:
        if not isinstance(init, BezierSimploid):
            raise ValueError("degree is required unless init is a Bézier simploid.")
        degree = init.degree
    return st, _check_degree(degree, len(st))


def fit(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: int | Sequence[int] | None = None,
    *,
    simploid_type: Sequence[int] | None = None,
    init: BezierSimploid | ControlPointData | None = None,
    freeze: Iterable[SimploidIndex | Sequence[int]] | None = None,
    smoothness_weight: float = 0.0,
    normalize: NormalizeType = "none",
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> BezierSimploid:
    r"""Fits a Bézier simplex, or more generally a Bézier simploid, to data.

    Without ``simploid_type``, ``params`` are the barycentric coordinates of a simplex, and
    the result is a Bézier simplex: the Bézier simploid of type ``(n_params - 1,)``.

    The control points minimize the mean squared error
    :math:`\frac{1}{nm} \sum_k \|B(\mathbf t_k) - \mathbf x_k\|^2` plus ``smoothness_weight`` times
    the sum of :math:`\|P_p - P_q\|^2` over the adjacent control points (see
    :meth:`BezierSimploid.adjacency`), where ``n`` is the number of samples and ``m`` that of values.
    Both solvers minimize this objective; ``"lstsq"`` does so exactly (see :mod:`torch_bsf.simploidal.fitting`).

    Parameters
    ----------
    params
        Parameters of shape ``(n, n_params)``: the barycentric coordinates of a simplex, or
        of every factor of a simploid of type :math:`(a_1, \ldots, a_n)`, concatenated
        (``n_params == sum(a_i + 1)``).
    values
        Values of shape ``(n, n_values)``.
    degree
        The degree of every factor, or one degree per factor.
        Optional if ``init`` is a Bézier simploid, whose degree is then used.
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.  Defaults to the type of ``init`` if it is a
        Bézier simploid, and to the simplex ``(n_params - 1,)`` otherwise.
    init
        Initial control points: a Bézier simploid of the same type, degree and number of values,
        or control points as accepted by :class:`BezierSimploid`, in the scale of ``values``.
        ``"trainer"`` starts from them (from random control points if ``None``); ``"lstsq"`` uses
        them only for the frozen rows.
    freeze
        Indices of control points that keep their values in ``init``, which is then required.
        A Bézier simplex also accepts its multi-indices alone, e.g. ``(3, 0, 0)``.
    smoothness_weight
        The weight :math:`w \ge 0` of the smoothness penalty.
    normalize
        How the values are normalized while fitting: ``"max"``, ``"std"``, ``"quantile"`` or ``"none"``
        (see :mod:`torch_bsf.preprocessing`).  The normalization is undone on the control points,
        so the returned model predicts in the scale of ``values`` either way; it changes only
        how the columns of ``values`` are weighted in the objective.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer`.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` on the free control points,
        added to :math:`nm` times the objective (``"lstsq"`` only).
        Without it, under-determined problems get the minimum-norm solution.
        With ``normalize``, the penalty applies to the normalized control points, so it shrinks them
        toward the offset of the normalization (e.g., the mean of ``values`` for ``"std"``) rather than zero,
        like an unpenalized intercept.
    batch_size
        Minibatch size (``"trainer"`` only; full batch by default).
    seed
        Random seed passed to :func:`lightning.pytorch.seed_everything` (``"trainer"`` only).
    kwargs
        Arguments for :class:`lightning.pytorch.Trainer` (``"trainer"`` only).

    Returns
    -------
    The fitted Bézier simploid, predicting in the scale of ``values``.

    Raises
    ------
    ValueError
        If the shapes do not match, there are no samples, ``degree`` is missing, ``freeze`` is given
        without ``init``, or an option is not supported by ``solver``.

    Examples
    --------
    Fit a Bézier triangle to data on a triangle:

    >>> import torch
    >>> import torch_bsf
    >>> from torch_bsf.sampling import simplex_grid
    >>> ts = simplex_grid(n_params=3, degree=3)  # parameters on a triangle
    >>> xs = 1 - ts * ts  # values corresponding to the parameters
    >>> bs = torch_bsf.fit(ts, xs, degree=2)
    >>> bs.simploid_type
    (2,)
    >>> torch.allclose(bs(ts), xs, atol=1e-5)
    True

    Fit a Bézier simploid on a prism, the product of a triangle and an edge:

    >>> from torch_bsf.simploidal import sample_simploid
    >>> t = sample_simploid((2, 1), 100, seed=0)
    >>> x = t[:, :3] * t[:, 3:4]
    >>> bs = torch_bsf.fit(t, x, degree=(1, 1), simploid_type=(2, 1))
    >>> torch.allclose(bs(t), x, atol=1e-5)
    True

    Keep the control point at a vertex and smooth the others:

    >>> init = torch_bsf.fit(t, x, degree=2, simploid_type=(2, 1))
    >>> vertex = ((2, 0, 0), (2, 0))
    >>> with torch.no_grad():
    ...     _ = init[vertex].fill_(1.0)
    >>> bs = torch_bsf.fit(t, x, init=init, freeze=[vertex], smoothness_weight=0.01)
    >>> bs[vertex].tolist()
    [1.0, 1.0, 1.0]
    """
    params, values = torch.as_tensor(params), torch.as_tensor(values)
    st, ds = _resolve_shape(params, simploid_type, degree, init)
    _check_values(params, values, simploid_n_params(st))
    if len(params) == 0:
        raise ValueError("At least one sample is required.")
    _check_solver(solver, ridge, smoothness_weight, batch_size, kwargs)
    if solver == "trainer" and seed is not None:
        L.seed_everything(seed)
    n_values = int(values.shape[1])
    if init is not None:
        model = _init_simploid(init, st, ds, n_values)
    elif freeze is not None:
        raise ValueError("init is required when freeze is given.")
    else:
        model = BezierSimploid.randn(st, ds, n_values) if solver == "trainer" else BezierSimploid.zeros(st, ds, n_values)
    frozen = sorted({model.row(index) for index in freeze or ()})
    dtype = _floating_dtype([values] if values.is_floating_point() else [params])
    model = model.to(dtype)
    scaler = make_scaler(normalize)
    values = scaler.fit_transform(values.to(model.control_points.device, dtype))
    original = model.control_points.detach().clone()
    with torch.no_grad():
        model.control_points.copy_(scaler.transform(original))
    _fit(
        model,
        [params],
        values,
        solver=solver,
        ridge=ridge,
        smoothness_weight=smoothness_weight,
        frozen=frozen,
        batch_size=batch_size,
        kwargs=kwargs,
    )
    with torch.no_grad():
        # The scalers are affine and the Bernstein basis sums to one, so this undoes the normalization of the model.
        restored = scaler.inverse_transform(model.control_points.detach())
        # Copy the frozen rows back exactly, free of the rounding of the round trip.
        restored[frozen] = original[frozen].to(restored)
        model.control_points.copy_(restored)
    return model


def _fold_indices(n_samples: int, n_folds: int, shuffle: bool, seed: int | None) -> list[torch.Tensor]:
    """The sample indices of ``min(n_folds, n_samples)`` nearly equal folds."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, but {n_folds} is given.")
    if n_samples < 2:
        raise ValueError(f"At least 2 samples are required for k-fold cross-validation, but {n_samples} is given.")
    if shuffle:
        generator = torch.Generator()
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        order = torch.randperm(n_samples, generator=generator)
    else:
        order = torch.arange(n_samples)
    return list(torch.tensor_split(order, min(n_folds, n_samples)))


def fit_kfold(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: int | Sequence[int] | None = None,
    *,
    n_folds: int = 5,
    shuffle: bool = True,
    seed: int | None = None,
    **kwargs: Any,
) -> nn.ModuleList:
    r"""Fits an ensemble of Bézier simploids by k-fold cross-validation.

    Splits the samples into ``n_folds`` folds and fits one model per fold, with :func:`fit`,
    to the samples of the other folds.  The resulting :class:`~torch.nn.ModuleList` can be
    passed directly to :func:`torch_bsf.active_learning.suggest_next_points` to drive a
    Query-By-Committee active learning loop.

    If ``len(params) < n_folds``, the number of folds is capped at ``len(params)``.

    Parameters
    ----------
    params
        Parameters of shape ``(n, n_params)``, as in :func:`fit`.
    values
        Values of shape ``(n, n_values)``.
    degree
        The degree, as in :func:`fit`.
    n_folds
        The number of folds (committee size).
    shuffle
        Whether to shuffle the samples before splitting them into folds.
        Otherwise, every fold is a contiguous block of samples.
    seed
        Random seed for shuffling, also passed to :func:`fit`.
    kwargs
        Other arguments for :func:`fit`, e.g. ``simploid_type``, ``smoothness_weight``, ``solver``
        and :class:`lightning.pytorch.Trainer` arguments.

    Returns
    -------
    A :class:`~torch.nn.ModuleList` of ``min(n_folds, len(params))`` fitted :class:`BezierSimploid`
    models; model ``k`` is fitted to all samples but those of fold ``k``.

    Raises
    ------
    ValueError
        If ``n_folds < 2``, ``len(params) < 2`` or ``len(params) != len(values)``, or from :func:`fit`.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf
    >>> from torch_bsf.active_learning import suggest_next_points
    >>> from torch_bsf.sampling import simplex_grid

    Prepare training data

    >>> params = simplex_grid(n_params=3, degree=3)
    >>> values = params.pow(2).sum(dim=1, keepdim=True)

    Build a 5-fold ensemble and suggest the 2 most uncertain points

    >>> models = torch_bsf.fit_kfold(params, values, degree=2, seed=0)
    >>> len(models)
    5
    >>> suggestions = suggest_next_points(models, n_suggestions=2, method="qbc")
    >>> suggestions.shape
    torch.Size([2, 3])

    See Also
    --------
    fit : Fit a single model.
    torch_bsf.active_learning.suggest_next_points : Use the ensemble for active learning.
    """
    params, values = torch.as_tensor(params), torch.as_tensor(values)
    folds = _fold_indices(len(params), n_folds, shuffle, seed)
    return _fit_folds(params, values, degree, folds, seed=seed, **kwargs)


def _check_n_samples(params: torch.Tensor, values: torch.Tensor) -> None:
    if len(params) != len(values):
        raise ValueError(f"params has {len(params)} samples, but values has {len(values)} samples.")


def _fit_folds(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: int | Sequence[int] | None,
    folds: Sequence[torch.Tensor],
    **kwargs: Any,
) -> nn.ModuleList:
    """Fits one model per fold, with :func:`fit`, to the samples of the other folds."""
    _check_n_samples(params, values)
    if kwargs.get("freeze") is not None:
        kwargs["freeze"] = list(kwargs["freeze"])  # an iterator would be consumed by the first fold
    models = nn.ModuleList()
    for k in range(len(folds)):
        train = torch.cat([fold for i, fold in enumerate(folds) if i != k])
        models.append(fit(params[train], values[train], degree, **kwargs))
    return models


def _init_simploid(
    init: BezierSimploid | ControlPointData,
    simploid_type: tuple[int, ...],
    degree: tuple[int, ...],
    n_values: int,
) -> BezierSimploid:
    """A copy of the initial control points as a Bézier simploid, checked against the fitted shape."""
    if isinstance(init, BezierSimploid):
        if (init.simploid_type, init.degree) != (simploid_type, degree):
            raise ValueError(
                f"init has type {init.simploid_type} and degree {init.degree}, "
                f"but type {simploid_type} and degree {degree} are fitted."
            )
        init = init.control_points.detach()
    model = BezierSimploid(simploid_type, degree, init)
    if model.n_values != n_values:
        raise ValueError(f"init has {model.n_values} values, but values have {n_values} columns.")
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
    init: SimploidalBezierSpace | torch.Tensor | None = None,
    freeze: Iterable[tuple[Simploid, SimploidIndex]] | None = None,
    smoothness_weight: float = 0.0,
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> SimploidalBezierSpace:
    r"""Fits a simploidal Bézier space to data on its simploids.

    Each sample is attached to a non-degenerate simploid (of any dimension)
    and given in that simploid's parameters.  Since simploids sharing a face
    share its control points, all pieces are fitted jointly and the result is
    continuous across shared faces.  The proper control points minimize the
    mean squared error over all samples plus ``smoothness_weight`` times the
    sum of :math:`\|P_p - P_q\|^2` over the adjacent stored control points
    (see :meth:`SimploidalBezierSpace.adjacency`), as in :func:`fit`.

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
    init
        Initial proper control points: a space with the same number of them, or a tensor of shape
        ``(n_proper_control_points, n_values)``.  ``"trainer"`` starts from them (from zero control
        points if ``None``); ``"lstsq"`` uses them only for the frozen rows.
    freeze
        Control points that keep their values in ``init``, which is then required, given as pairs
        ``(simploid, index)`` of any simploid of the set and a control point index of its Bézier simploid.
    smoothness_weight
        The weight :math:`w \ge 0` of the smoothness penalty.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer`.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` on the free proper control points,
        added to :math:`nm` times the objective (``"lstsq"`` only).
        Without it, under-determined problems get the minimum-norm solution.
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
    # Convert the space before copying ``init`` into it, so that the initial (and frozen) points are not rounded.
    space = space.to(dtype)
    _check_solver(solver, ridge, smoothness_weight, batch_size, kwargs)
    if init is not None:
        _init_space(space, init)
    elif freeze is not None:
        raise ValueError("init is required when freeze is given.")
    frozen = [space.row(simploid, index) for simploid, index in freeze or ()]
    if solver == "trainer" and seed is not None:
        L.seed_everything(seed)
    width = max(p.shape[1] for _, p, _ in samples)
    ids = torch.cat([torch.full((len(p),), space.simploid_id(s), dtype=torch.long) for s, p, _ in samples])
    params_dtype = _floating_dtype([p for _, p, _ in samples])
    params = torch.cat([F.pad(p.to(params_dtype), (0, width - p.shape[1])) for _, p, _ in samples])
    values = torch.cat([v.to(dtype) for _, _, v in samples])
    _fit(
        space,
        [params, ids],
        values,
        solver=solver,
        ridge=ridge,
        smoothness_weight=smoothness_weight,
        frozen=frozen,
        batch_size=batch_size,
        kwargs=kwargs,
    )
    return space


def _init_space(space: SimploidalBezierSpace, init: SimploidalBezierSpace | torch.Tensor) -> None:
    """Copies the initial proper control points into ``space``, checking their shape."""
    points = init.proper_control_points if isinstance(init, SimploidalBezierSpace) else torch.as_tensor(init)
    if tuple(points.shape) != tuple(space.proper_control_points.shape):
        raise ValueError(
            f"init must have shape {tuple(space.proper_control_points.shape)}, but {tuple(points.shape)} is given."
        )
    with torch.no_grad():
        space.proper_control_points.copy_(points.detach())
