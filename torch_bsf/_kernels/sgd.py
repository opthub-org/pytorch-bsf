r"""Fitting Bézier simploids and simploidal Bézier spaces to Pareto sets by stochastic gradient descent.

This module implements the preconditioned stochastic gradient descent of :cite:t:`hikima2025stochastic`,
generalized from Bézier simplices to every model of :mod:`torch_bsf._kernels`.

Instead of fitting a model to data, :func:`fit_sgd` minimizes the expected loss

.. math::

   \mathcal L(P) = \mathbb E_{\mathbf u}\bigl[\ell(B(\mathbf u \mid P), \mathbf u)\bigr]

over the control points :math:`P`, where :math:`\mathbf u` is uniform on the domain of the model.  For a
Bézier simplex on :math:`\Delta^{M-1}` and objectives :math:`f: \mathbb R^L \to \mathbb R^M`, the loss
:math:`\ell(\mathbf x, \mathbf t) = \mathbf t^\top f(\mathbf x)` (see :func:`linear_scalarization`) makes
the minimizer approximate the Pareto set of :math:`f`, parametrized by the weight :math:`\mathbf t`.

The expectation is approximated by the mean over ``n`` fixed samples, and the control points are updated
on minibatches :math:`B` of them as

.. math::

   P^{(k+1)} = P^{(k)} - \alpha_k \Sigma^{-1} \nabla \mathcal L_B(P^{(k)}),
   \qquad \Sigma = \mathbb E_{\mathbf u}\bigl[\mathbf z(\mathbf u) \mathbf z(\mathbf u)^\top\bigr],

where :math:`\mathbf z(\mathbf u)` is the row of the design matrix at :math:`\mathbf u`, so that
:math:`B(\mathbf u \mid P) = P^\top \mathbf z(\mathbf u)`.  The step minimizes the linearized loss plus
:math:`\frac{1}{2\alpha_k} \mathbb E_{\mathbf u} \|B(\mathbf u \mid P) - B(\mathbf u \mid P^{(k)})\|^2`,
i.e., it measures the move of the model rather than that of its control points.

The Gram matrix :math:`\Sigma` (see :func:`gram_matrix`) is computed exactly:

* for a Bézier simplex of degree :math:`D` on :math:`\Delta^a`, the moments of the flat Dirichlet
  distribution give
  :math:`\Sigma_{\boldsymbol\alpha\boldsymbol\beta} = \binom{D}{\boldsymbol\alpha} \binom{D}{\boldsymbol\beta}
  \frac{a! \prod_k (\alpha_k + \beta_k)!}{(2D + a)!}`;
* for a Bézier simploid, whose basis is a tensor product and whose uniform distribution is a product,
  :math:`\Sigma` is the Kronecker product of the matrices of its factors;
* for a simploidal Bézier space, :math:`\mathbf u` lies on the simploid :math:`c` with probability
  :math:`w_c` and is uniform on it, so :math:`\Sigma = \sum_c w_c S_c^\top \Sigma_c S_c`, where :math:`S_c`
  maps the proper control points to the control net of :math:`c`.
"""

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import lgamma
from typing import Literal, TypeAlias, TypeVar, cast

import torch

from torch_bsf._kernels.bezier import BezierSimploid
from torch_bsf._kernels.combinatorics import (
    SimploidIndex,
    SimploidType,
    _factor_tables,
    sample_simploid,
    simploid_n_params,
)
from torch_bsf._kernels.fitting import _incidence, _parameter, _smoothness_penalty
from torch_bsf._kernels.sets import Simploid
from torch_bsf._kernels.space import SimploidalBezierSpace

#: A loss ``loss(values, *inputs)`` returning the loss of every sample, of shape ``(batch,)``, where ``values``
#: is the output of the model and ``inputs`` are the arguments of its ``forward``: ``(t,)`` for a
#: :class:`BezierSimploid` and ``(t, simploid_ids)`` for a :class:`SimploidalBezierSpace`.
SGDLoss: TypeAlias = Callable[..., torch.Tensor]

#: Control points to freeze: indices of a :class:`BezierSimploid` (as for :func:`~torch_bsf.fit`), or pairs
#: ``(simploid, index)`` for a :class:`SimploidalBezierSpace` (as for :func:`fit_simploidal_set`).
FreezeSpec: TypeAlias = Iterable[SimploidIndex | Sequence[int] | tuple[Simploid, SimploidIndex]]

StepDecay = Literal["constant", "sqrt"]
Preconditioner = Literal["gram", "none"]

_Model = TypeVar("_Model", BezierSimploid, SimploidalBezierSpace)

#: A callback ``callback(k, model, batch_loss)`` of :func:`fit_sgd`, called after the update of every iteration
#: with the 0-based iteration number ``k``, the model being fitted and the mean loss of the minibatch at the
#: parameters before the update (without the smoothness penalty).  Returning ``True`` stops the fitting.
SGDCallback: TypeAlias = Callable[[int, _Model, float], bool | None]


def _factor_gram(a: int, d: int) -> torch.Tensor:
    """The Gram matrix of the Bernstein basis of degree ``d`` on the standard ``a``-simplex, in double precision."""
    idx, coeffs = _factor_tables(a, d)
    powers = torch.tensor(idx, dtype=torch.float64).reshape(len(idx), a + 1)
    c = torch.tensor(coeffs, dtype=torch.float64)
    # The moments E[t^gamma] = a! prod_k gamma_k! / (|gamma| + a)! of the flat Dirichlet distribution, at gamma = alpha + beta.
    log_factorials = torch.lgamma(powers.unsqueeze(1) + powers.unsqueeze(0) + 1).sum(dim=-1)
    log_moments = log_factorials + lgamma(a + 1) - lgamma(2 * d + a + 1)
    return c.unsqueeze(1) * c.unsqueeze(0) * torch.exp(log_moments)


def _simploid_gram(simploid_type: SimploidType, degree: tuple[int, ...]) -> torch.Tensor:
    """The Gram matrix of the basis of a Bézier simploid, in the order of :func:`simploid_indices`."""
    gram = torch.ones(1, 1, dtype=torch.float64)
    for a, d in zip(simploid_type, degree, strict=True):
        gram = torch.kron(gram, _factor_gram(a, d))
    return gram


def _cell_weights(space: SimploidalBezierSpace, weights: Mapping[Simploid, float] | None) -> dict[Simploid, float]:
    """The probability of every simploid of ``space``: ``weights`` normalized, or uniform over its main simploids."""
    if weights is None:
        # The main simploids of the structure captured by the space, which ignores later changes to its set.
        faces = {id(f.base) for s in space.simploids for _, f in s.faces()}
        weights = {s: 1.0 for s in space.simploids if id(s) not in faces}
    for s, w in weights.items():
        space.simploid_id(s)
        if not w >= 0:
            raise ValueError(f"weights must be non-negative, but {w} is given for {s!r}.")
    total = sum(weights.values())
    if not total > 0:
        raise ValueError("weights must have a positive sum.")
    return {s: w / total for s, w in weights.items() if w > 0}


def gram_matrix(
    model: BezierSimploid | SimploidalBezierSpace, weights: Mapping[Simploid, float] | None = None
) -> torch.Tensor:
    r"""The Gram matrix :math:`\Sigma = \mathbb E[\mathbf z(\mathbf u) \mathbf z(\mathbf u)^\top]` of a model.

    :math:`\mathbf z(\mathbf u)` is the row of ``model.design_matrix`` at :math:`\mathbf u`, which is uniform on
    the domain of the model (see :func:`sample_domain`).  The matrix is computed exactly (see
    :mod:`torch_bsf._kernels.sgd`).

    Parameters
    ----------
    model
        A Bézier simploid or a simploidal Bézier space.
    weights
        For a space, the probability of every non-degenerate simploid (normalized to sum to one).
        Uniform over the main simploids of the space (those that are not a face of another) if ``None``.
        Ignored for a Bézier simploid.

    Returns
    -------
    A positive definite matrix of shape ``(n, n)`` in double precision on the CPU, where ``n`` is the number
    of rows of the parameter of ``model``.

    Raises
    ------
    ValueError
        If ``weights`` has a negative entry, sums to zero, mentions a simploid that is not a non-degenerate
        simploid of the space, or leaves a proper control point outside every simploid it weights.

    Examples
    --------
    The Gram matrix of the linear Bézier curve (the paper's closed form for :math:`M = 2`, :math:`D = 1`):

    >>> from torch_bsf._kernels import BezierSimploid, gram_matrix
    >>> import torch
    >>> gram = gram_matrix(BezierSimploid.zeros((1,), 1, n_values=1))
    >>> torch.allclose(gram * 6, torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64))
    True
    """
    if isinstance(model, BezierSimploid):
        return _simploid_gram(model.simploid_type, model.degree)
    gram = torch.zeros(model.n_proper_control_points, model.n_proper_control_points, dtype=torch.float64)
    covered = torch.zeros(model.n_proper_control_points, dtype=torch.bool)
    for s, w in _cell_weights(model, weights).items():
        rows = torch.tensor(model.control_point_rows(s), dtype=torch.long)
        cell = _simploid_gram(s.simploid_type, model.simploid_degree(s))
        gram.index_put_((rows.unsqueeze(1), rows.unsqueeze(0)), w * cell, accumulate=True)
        covered[rows] = True
    if not covered.all():
        raise ValueError(
            f"weights must cover every proper control point, but rows {torch.nonzero(~covered).flatten().tolist()} "
            "lie on no weighted simploid."
        )
    return gram


def _generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _draw_seed(generator: torch.Generator) -> int:
    return int(torch.randint(2**62, (1,), generator=generator))


def sample_domain(
    model: BezierSimploid | SimploidalBezierSpace,
    n_samples: int,
    *,
    weights: Mapping[Simploid, float] | None = None,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Samples inputs uniformly on the domain of a model.

    For a Bézier simploid, the parameters are uniform on its simploid (see :func:`sample_simploid`).  For a
    simploidal Bézier space, each sample lies on a simploid drawn with probability ``weights`` and is uniform
    on it; its parameters are padded with zeros to the widest simploid, as :meth:`SimploidalBezierSpace.forward`
    expects.

    Parameters
    ----------
    model
        A Bézier simploid or a simploidal Bézier space.
    n_samples
        The number of samples.
    weights
        For a space, the probability of every non-degenerate simploid (normalized to sum to one).
        Uniform over the main simploids of the space (those that are not a face of another) if ``None``.
        Ignored for a Bézier simploid.
    seed
        Random seed for reproducibility.

    Returns
    -------
    The inputs of the model: ``[t]`` for a Bézier simploid and ``[t, simploid_ids]`` for a space,
    so that ``model(*inputs)`` evaluates it.

    Examples
    --------
    >>> from torch_bsf._kernels import SimploidalSet, SimploidalBezierSpace, sample_domain
    >>> S = SimploidalSet()
    >>> tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    >>> space = SimploidalBezierSpace(S, degree=2, n_values=1)
    >>> t, ids = sample_domain(space, 5, seed=0)
    >>> t.shape, ids.shape
    (torch.Size([5, 4]), torch.Size([5]))
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, but {n_samples} is given.")
    generator = _generator(seed)
    if isinstance(model, BezierSimploid):
        return [sample_simploid(model.simploid_type, n_samples, seed=_draw_seed(generator))]
    cells = _cell_weights(model, weights)
    simploids = list(cells)
    probabilities = torch.tensor([cells[s] for s in simploids], dtype=torch.float64)
    choice = torch.multinomial(probabilities, n_samples, replacement=True, generator=generator)
    width = max(simploid_n_params(s.simploid_type) for s in simploids)
    t = torch.zeros(n_samples, width)
    ids = torch.empty(n_samples, dtype=torch.long)
    for k, s in enumerate(simploids):
        (rows,) = torch.nonzero(choice == k, as_tuple=True)
        t[rows, : simploid_n_params(s.simploid_type)] = sample_simploid(s.simploid_type, len(rows), seed=_draw_seed(generator))
        ids[rows] = model.simploid_id(s)
    return [t, ids]


def linear_scalarization(objective: Callable[[torch.Tensor], torch.Tensor]) -> SGDLoss:
    r"""The loss :math:`\ell(\mathbf x, \mathbf t) = \mathbf t^\top f(\mathbf x)` of :cite:t:`hikima2025stochastic`.

    It weights the objectives :math:`f = (f_1, \ldots, f_M)` by the parameter :math:`\mathbf t` itself, so it
    is meant for a Bézier simplex on :math:`\Delta^{M-1}`, whose parameters have ``M`` columns.

    Parameters
    ----------
    objective
        The objectives: a differentiable map from points of shape ``(batch, L)`` to objective values of
        shape ``(batch, M)``.

    Returns
    -------
    A loss for :func:`fit_sgd` that raises :class:`ValueError` unless the parameters have ``M`` columns.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf._kernels import linear_scalarization
    >>> loss = linear_scalarization(lambda x: torch.stack([x[:, 0] ** 2, (x[:, 0] - 1) ** 2], dim=1))
    >>> loss(torch.tensor([[0.5]]), torch.tensor([[0.25, 0.75]]))
    tensor([0.2500])
    """

    def loss(values: torch.Tensor, t: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        fs = objective(values)
        if fs.shape != t.shape:
            raise ValueError(
                f"The objective values must have the shape {tuple(t.shape)} of the parameters, but {tuple(fs.shape)} "
                "is given; linear_scalarization needs a Bézier simplex with one vertex per objective."
            )
        return (t * fs).sum(dim=-1)

    return loss


#: The attributes of every :class:`torch.nn.Module` (parameters, buffers, submodules, hooks, ...).
_MODULE_STATE = frozenset(vars(torch.nn.Module()))


def _copy_model(model: _Model) -> _Model:
    """A copy of ``model`` with its own parameters, sharing its other attributes.

    The cells of a space are identified by identity, so the copy keeps them (and their simploidal set) as they are.
    """
    shared = {id(v): v for k, v in vars(model).items() if k not in _MODULE_STATE}
    return copy.deepcopy(model, shared)


def _check_sgd_options(
    max_iter: int, step_size: float, batch_size: int | None, step_decay: str, preconditioner: str, smoothness_weight: float
) -> None:
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, but {max_iter} is given.")
    if not step_size > 0:
        raise ValueError(f"step_size must be positive, but {step_size} is given.")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be positive, but {batch_size} is given.")
    if step_decay not in ("constant", "sqrt"):
        raise ValueError(f"Unknown step_decay {step_decay!r}.")
    if preconditioner not in ("gram", "none"):
        raise ValueError(f"Unknown preconditioner {preconditioner!r}.")
    if not smoothness_weight >= 0:
        raise ValueError(f"smoothness_weight must be non-negative, but {smoothness_weight} is given.")


def _frozen_rows(model: BezierSimploid | SimploidalBezierSpace, freeze: FreezeSpec | None) -> list[int]:
    """The sorted rows of the parameter of ``model`` holding the control points in ``freeze``.

    The indices are resolved by :meth:`BezierSimploid.row` or :meth:`SimploidalBezierSpace.row`, as in
    :func:`~torch_bsf.fit` and :func:`~torch_bsf._kernels.fit_simploidal_set`.
    """
    if freeze is None:
        return []
    if isinstance(model, BezierSimploid):
        return sorted({model.row(cast(SimploidIndex, index)) for index in freeze})
    rows = set()
    for item in freeze:
        try:
            simploid, index = cast(tuple[Simploid, SimploidIndex], item)
        except (TypeError, ValueError):
            raise ValueError(f"freeze of a space must give pairs (simploid, index), but {item!r} is given.") from None
        if not isinstance(simploid, Simploid):
            raise ValueError(f"freeze of a space must give pairs (simploid, index), but {item!r} is given.")
        rows.add(model.row(simploid, index))
    return sorted(rows)


def _prepare_inputs(inputs: Sequence[torch.Tensor], parameter: torch.Tensor) -> list[torch.Tensor]:
    """The inputs on the device of ``parameter``, the floating ones in its dtype, checking that they have one length."""
    if not inputs:
        raise ValueError("inputs must not be empty.")
    tensors = [torch.as_tensor(x).to(parameter.device) for x in inputs]
    tensors = [x.to(parameter.dtype) if x.is_floating_point() else x for x in tensors]
    n = len(tensors[0])
    if n == 0 or any(len(x) != n for x in tensors):
        raise ValueError(f"inputs must have one positive length, but {[tuple(x.shape) for x in tensors]} are given.")
    return tensors


class _Update:
    r"""The step :math:`P_F \leftarrow P_F - \alpha (M_{FF} + 2 \alpha w L_{FF})^{-1} \nabla_F` on the free rows ``F``.

    :math:`M` is the metric of the step (:math:`\Sigma`, or the identity for plain SGD), :math:`L` the Laplacian
    of the adjacency graph and :math:`w` the smoothness weight; see :func:`fit_sgd`.  The matrix is factorized
    once, or once per step size if :math:`w > 0`.  The frozen rows are never written, so they keep their values
    bit for bit.
    """

    def __init__(
        self,
        model: BezierSimploid | SimploidalBezierSpace,
        parameter: torch.Tensor,
        frozen: Sequence[int],
        preconditioner: str,
        weights: Mapping[Simploid, float] | None,
        smoothness_weight: float,
    ) -> None:
        n = len(parameter)
        self.free: torch.Tensor | None = None
        rows = torch.arange(n)
        if frozen:
            is_free = torch.ones(n, dtype=torch.bool)
            is_free[list(frozen)] = False
            rows = torch.nonzero(is_free).flatten()
            self.free = rows.to(parameter.device)
        self.device, self.dtype = parameter.device, parameter.dtype
        block = (rows.unsqueeze(1), rows.unsqueeze(0))
        self.metric: torch.Tensor | None = None
        if preconditioner == "gram":
            self.metric = gram_matrix(model, weights)[block]
        self.laplacian: torch.Tensor | None = None
        if smoothness_weight > 0:
            d = _incidence(model.adjacency(), n)
            self.laplacian = (2 * smoothness_weight * d.T @ d)[block]
            if self.metric is None:
                self.metric = torch.eye(len(rows), dtype=torch.float64)
        self.alpha: float | None = None
        self.inverse: torch.Tensor | None = None

    def _inverse(self, alpha: float) -> torch.Tensor | None:
        if self.metric is None:
            return None
        if self.inverse is None or (self.laplacian is not None and alpha != self.alpha):
            matrix = self.metric if self.laplacian is None else self.metric + alpha * self.laplacian
            self.inverse = torch.cholesky_inverse(torch.linalg.cholesky(matrix)).to(self.device, self.dtype)
            self.alpha = alpha
        return self.inverse

    def __call__(self, parameter: torch.Tensor, grad: torch.Tensor, alpha: float) -> None:
        if self.free is not None:
            grad = grad[self.free]
        inverse = self._inverse(alpha)
        direction = grad if inverse is None else inverse @ grad
        with torch.no_grad():
            if self.free is None:
                parameter.sub_(alpha * direction)
            else:
                parameter[self.free] -= alpha * direction


def fit_sgd(
    model: _Model,
    loss: SGDLoss,
    *,
    inputs: Sequence[torch.Tensor] | None = None,
    n_samples: int = 100,
    batch_size: int | None = None,
    max_iter: int = 100,
    step_size: float = 0.05,
    step_decay: StepDecay = "constant",
    preconditioner: Preconditioner = "gram",
    weights: Mapping[Simploid, float] | None = None,
    freeze: FreezeSpec | None = None,
    smoothness_weight: float = 0.0,
    seed: int | None = None,
    callback: SGDCallback[_Model] | None = None,
) -> _Model:
    r"""Fits a model to the minimizers of a loss by (preconditioned) stochastic gradient descent.

    This is Algorithm 1 of :cite:t:`hikima2025stochastic`, generalized to Bézier simploids and simploidal
    Bézier spaces (see :mod:`torch_bsf._kernels.sgd`).  It minimizes the mean of ``loss`` over fixed
    samples ``inputs`` of the domain: at every iteration ``k``, it draws a minibatch of ``batch_size``
    samples uniformly without replacement and updates the parameter :math:`P` of the model as
    :math:`P \leftarrow P - \alpha_k \Sigma^{-1} \nabla \mathcal L_B(P)`, where :math:`\Sigma` is
    :func:`gram_matrix` (or the identity for plain SGD).

    The control points in ``freeze`` keep their values in ``model``, and only the free rows :math:`F` of
    :math:`P` move.  With ``smoothness_weight`` :math:`w > 0`, the loss also has the smoothness penalty
    :math:`w \sum_{(p, q)} \|P_p - P_q\|^2 = w \operatorname{tr}(P^\top L P)` over the adjacent control points
    (see ``model.adjacency()``), as in :func:`~torch_bsf.fit`, where :math:`L` is the Laplacian of the
    adjacency graph.  The step then minimizes the linearized loss plus the exact penalty plus the move of the
    model over the free rows, which gives

    .. math::

       P_F \leftarrow P_F - \alpha_k (\Sigma_{FF} + 2 \alpha_k w L_{FF})^{-1} \nabla_F \bigl(\mathcal L_B(P)
       + w \operatorname{tr}(P^\top L P)\bigr).

    Treating the penalty exactly rather than by its gradient keeps the step stable for any :math:`w`, while
    :math:`\Sigma^{-1} L` has eigenvalues in the thousands already for a cubic Bézier triangle.  Without
    frozen rows and with :math:`w = 0`, this is the step above.

    Parameters
    ----------
    model
        The model to fit, which also gives the initial control points (zeros in the paper).
        It is not modified.
    loss
        The loss ``loss(values, *inputs)`` of every sample (see :data:`SGDLoss`), differentiable with
        respect to ``values``; e.g., :func:`linear_scalarization` to approximate a Pareto set.
    inputs
        The samples, as the arguments of ``model.forward``.  Drawn with :func:`sample_domain` if ``None``.
    n_samples
        The number of samples drawn when ``inputs`` is ``None``.
    batch_size
        The minibatch size (full batch if ``None`` or larger than the number of samples).
    max_iter
        The number of iterations.
    step_size
        The step size :math:`\alpha > 0`.
    step_decay
        ``"constant"`` uses :math:`\alpha_k = \alpha`; ``"sqrt"`` uses :math:`\alpha_k = \alpha / \sqrt{k + 1}`.
    preconditioner
        ``"gram"`` preconditions the gradient by :math:`\Sigma^{-1}` (the proposed method);
        ``"none"`` is plain stochastic gradient descent.
    weights
        For a space, the probability of every non-degenerate simploid, used both to sample ``inputs``
        and in :math:`\Sigma`.  Uniform over the main simploids of the space (those that are not a face of
        another) if ``None``.
    freeze
        Control points that keep their values in ``model``.  For a Bézier simploid, their indices as in
        :func:`~torch_bsf.fit` (a Bézier simplex also accepts its multi-indices alone, e.g. ``(2, 0, 0)``);
        for a space, pairs ``(simploid, index)`` as in :func:`~torch_bsf._kernels.fit_simploidal_set`.
    smoothness_weight
        The weight :math:`w \ge 0` of the smoothness penalty added to the loss.
    seed
        Random seed for the samples and the minibatches.
    callback
        Called as ``callback(k, model, batch_loss)`` after the update of every iteration ``k = 0, 1, ...``
        (see :data:`SGDCallback`), where ``model`` is the model being fitted and ``batch_loss`` is the mean
        loss of the minibatch at the parameters before the update, without the smoothness penalty of
        ``smoothness_weight``.  If it returns ``True``, the fitting
        stops after this iteration.  It may evaluate ``model``, preferably under :func:`torch.no_grad`,
        but must not modify it: copy the parameters to keep a snapshot.

    Returns
    -------
    A fitted copy of ``model``.

    Raises
    ------
    ValueError
        If an option is out of range, ``freeze`` has an invalid index, the inputs are empty or have
        different lengths, or ``loss`` does not return one value per sample.

    Examples
    --------
    The Pareto set of :math:`f_m(\mathbf x) = \|\mathbf x - \mathbf e_m\|^2` (:math:`m = 1, 2, 3`) is the
    triangle spanned by :math:`\mathbf e_1, \mathbf e_2, \mathbf e_3`, reached at :math:`\mathbf x = \mathbf t`:

    >>> import torch
    >>> import torch_bsf
    >>> from torch_bsf._kernels import BezierSimploid, linear_scalarization
    >>> def objective(x):
    ...     return ((x.unsqueeze(1) - torch.eye(3)) ** 2).sum(dim=-1)
    >>> init = BezierSimploid.zeros((2,), 1, n_values=3)
    >>> bs = torch_bsf.fit_sgd(init, linear_scalarization(objective), batch_size=20, step_size=0.2, seed=0)
    >>> torch.allclose(bs.control_points, torch.eye(3), atol=1e-3)
    True

    A callback records the loss of every iteration and stops the fitting once the loss stops decreasing:

    >>> history = []
    >>> def callback(k, model, batch_loss):
    ...     history.append(batch_loss)
    ...     return k > 0 and history[-2] - batch_loss < 1e-4
    >>> bs = torch_bsf.fit_sgd(init, linear_scalarization(objective), step_size=0.2, seed=0, callback=callback)
    >>> len(history) < 100  # stopped before max_iter
    True

    Keep the vertices of a quadratic Bézier triangle at the known minimizers of the single objectives,
    and smooth its control net:

    >>> init = BezierSimploid.zeros((2,), 2, n_values=3)
    >>> vertices = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    >>> with torch.no_grad():
    ...     for m, vertex in enumerate(vertices):
    ...         _ = init[vertex].copy_(torch.eye(3)[m])
    >>> bs = torch_bsf.fit_sgd(
    ...     init, linear_scalarization(objective), freeze=vertices, smoothness_weight=1e-3, step_size=0.2, seed=0
    ... )
    >>> bs[(2, 0, 0)].tolist()
    [1.0, 0.0, 0.0]
    """
    _check_sgd_options(max_iter, step_size, batch_size, step_decay, preconditioner, smoothness_weight)
    model = _copy_model(model)
    parameter = _parameter(model)
    if inputs is None:
        inputs = sample_domain(model, n_samples, weights=weights, seed=seed)
    samples = _prepare_inputs(inputs, parameter)
    n = len(samples[0])
    m = n if batch_size is None else min(batch_size, n)
    update = _Update(model, parameter, _frozen_rows(model, freeze), preconditioner, weights, smoothness_weight)
    adjacency = model.adjacency().to(parameter.device) if smoothness_weight > 0 else None
    generator = _generator(None if seed is None else seed + 1)
    for k in range(max_iter):
        rows = torch.randperm(n, generator=generator)[:m].to(parameter.device)
        batch = [x[rows] for x in samples]
        losses = loss(model(*batch), *batch)
        if losses.shape != (m,):
            raise ValueError(f"loss must return a tensor of shape ({m},), but {tuple(losses.shape)} is given.")
        objective = batch_loss = losses.mean()
        if adjacency is not None:
            objective = objective + smoothness_weight * _smoothness_penalty(parameter, adjacency)
        (grad,) = torch.autograd.grad(objective, parameter)
        alpha = step_size if step_decay == "constant" else step_size / (k + 1) ** 0.5
        update(parameter, grad, alpha)
        if callback is not None and callback(k, model, batch_loss.item()):
            break
    return model
