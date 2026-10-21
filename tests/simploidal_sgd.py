from math import factorial, prod
from typing import Any

import pytest
import torch

import torch_bsf
from torch_bsf.simploidal import (
    BezierSimploid,
    Simploid,
    SimploidalBezierSpace,
    SimploidalSet,
    fit_sgd,
    gram_matrix,
    linear_scalarization,
    sample_domain,
    sample_simploid,
    simploid_indices,
)


def _location_objective(x: torch.Tensor) -> torch.Tensor:
    """f_m(x) = ||x - e_m||^2 for m = 1, 2, 3, whose Pareto set is x = t on the triangle."""
    return ((x.unsqueeze(1) - torch.eye(3, dtype=x.dtype)) ** 2).sum(dim=-1)


def _squared_error(target: torch.nn.Module) -> Any:
    """A loss whose minimizer is ``target``."""

    def loss(values: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            expected: torch.Tensor = target(*inputs)
        return ((values - expected) ** 2).sum(dim=-1)

    return loss


def _mean_loss(model: torch.nn.Module, loss: Any, inputs: list[torch.Tensor]) -> float:
    with torch.no_grad():
        return float(loss(model(*inputs), *inputs).mean())


def _glued_space(n_values: int = 2) -> tuple[SimploidalSet, SimploidalBezierSpace, list[Simploid]]:
    """A triangle and a square glued along an edge, with an extra edge hanging from a vertex of the triangle."""
    S = SimploidalSet()
    tri, sq, edge = S.add_simplex(2), S.add_simploid((1, 1)), S.add_simplex(1)
    S.identify(tri.face(0, 2), sq.face(1, 1))
    S.identify(tri.vertices()[0], edge.vertices()[0])
    return S, SimploidalBezierSpace(S, 2, n_values).double(), [tri, sq, edge]


@pytest.mark.parametrize("a, d", [(a, d) for a in range(1, 4) for d in range(4)])
def test_gram_matrix_simplex_closed_form(a: int, d: int) -> None:
    """The Gram matrix of a Bézier simplex matches the closed form of Hikima et al. (2025)."""
    gram = gram_matrix(BezierSimploid.zeros((a,), d, n_values=1))
    indices = simploid_indices((a,), d)
    m = a + 1
    for i, (alpha,) in enumerate(indices):
        for j, (beta,) in enumerate(indices):
            binom = [factorial(d) / prod(factorial(c) for c in x) for x in (alpha, beta)]
            binom2 = factorial(2 * d) / prod(factorial(p + q) for p, q in zip(alpha, beta, strict=True))
            expected = factorial(2 * d) * factorial(m - 1) / factorial(2 * d + m - 1) * binom[0] * binom[1] / binom2
            assert gram[i, j].item() == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("simploid_type, degree", [((), ()), ((1,), 3), ((2,), 2), ((1, 1), (2, 1)), ((2, 1), 2)])
def test_gram_matrix_simploid_monte_carlo(simploid_type: tuple[int, ...], degree: Any) -> None:
    """The Gram matrix of a Bézier simploid is E[z z^T] under the uniform distribution."""
    bs = BezierSimploid.zeros(simploid_type, degree, n_values=1).double()
    t = sample_simploid(simploid_type, 200_000, seed=0).double()
    z = bs.design_matrix(t)
    gram = gram_matrix(bs)
    assert gram.dtype == torch.float64
    assert torch.allclose(gram, z.T @ z / len(t), atol=2e-3)
    assert torch.allclose(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() > 0


def test_gram_matrix_space_monte_carlo() -> None:
    """The Gram matrix of a space is E[z z^T], with the cells drawn by their weights."""
    _, space, (tri, sq, edge) = _glued_space()
    for weights in (None, {tri: 1.0, sq: 2.0, edge: 0.5}):
        t, ids = sample_domain(space, 200_000, weights=weights, seed=0)
        z = space.design_matrix(t.double(), ids)
        gram = gram_matrix(space, weights)
        assert torch.allclose(gram, z.T @ z / len(t), atol=2e-3)
        assert torch.linalg.eigvalsh(gram).min() > 0


def test_gram_matrix_space_of_one_simploid() -> None:
    """A space of a single simploid has the Gram matrix of its Bézier simploid, in the order of its rows."""
    S = SimploidalSet()
    sq = S.add_simploid((2, 1))
    space = SimploidalBezierSpace(S, 2, n_values=1)
    rows = torch.tensor(space.control_point_rows(sq))
    gram = gram_matrix(space)
    assert torch.allclose(gram[rows][:, rows], gram_matrix(space.bezier_simploid(sq)))


def test_gram_matrix_space_weights_errors() -> None:
    _, space, (tri, sq, edge) = _glued_space()
    with pytest.raises(ValueError, match="cover every proper control point"):
        gram_matrix(space, {tri: 1.0, sq: 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        gram_matrix(space, {tri: -1.0, sq: 1.0, edge: 1.0})
    with pytest.raises(ValueError, match="positive sum"):
        gram_matrix(space, {tri: 0.0})
    with pytest.raises(ValueError, match="not a non-degenerate simploid"):
        gram_matrix(space, {SimploidalSet().add_simplex(2): 1.0})


def test_sample_domain_simploid() -> None:
    bs = BezierSimploid.zeros((2, 1), 2, n_values=1)
    (t,) = sample_domain(bs, 50, seed=3)
    assert t.shape == (50, 5)
    assert torch.allclose(t[:, :3].sum(dim=1), torch.ones(50))
    assert torch.allclose(t[:, 3:].sum(dim=1), torch.ones(50))
    assert torch.equal(t, sample_domain(bs, 50, seed=3)[0])
    with pytest.raises(ValueError, match="n_samples"):
        sample_domain(bs, -1)


def test_sample_domain_space() -> None:
    _, space, (tri, sq, edge) = _glued_space()
    t, ids = sample_domain(space, 300, weights={tri: 1.0, sq: 1.0, edge: 0.0}, seed=0)
    assert t.shape == (300, 4) and ids.shape == (300,)
    assert set(ids.tolist()) == {space.simploid_id(tri), space.simploid_id(sq)}
    on_tri = ids == space.simploid_id(tri)
    assert torch.allclose(t[on_tri, :3].sum(dim=1), torch.ones(int(on_tri.sum())))
    assert (t[on_tri, 3] == 0).all()
    t2, ids2 = sample_domain(space, 300, weights={tri: 1.0, sq: 1.0, edge: 0.0}, seed=0)
    assert torch.equal(t, t2) and torch.equal(ids, ids2)
    assert set(sample_domain(space, 300, seed=0)[1].tolist()) == {space.simploid_id(s) for s in (tri, sq, edge)}


def test_linear_scalarization() -> None:
    loss = linear_scalarization(_location_objective)
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    t = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    assert torch.allclose(loss(x, t), torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="one vertex per objective"):
        loss(x, torch.tensor([[0.5, 0.5], [0.5, 0.5]]))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fit_sgd_recovers_pareto_set(degree: int) -> None:
    """The Pareto set of the location problem, x*(t) = t, is a Bézier triangle of any degree."""
    init = BezierSimploid.zeros((2,), degree, n_values=3)
    loss = linear_scalarization(_location_objective)
    bs = fit_sgd(init, loss, n_samples=100, batch_size=20, max_iter=200, step_size=0.2, seed=0)
    (t,) = sample_domain(bs, 1000, seed=1)
    with torch.no_grad():
        assert torch.allclose(bs(t), t, atol=1e-3)


def test_fit_sgd_preconditioner_beats_plain_sgd() -> None:
    """With the same budget, the preconditioned method reaches a lower loss than plain SGD (Hikima et al., 2025)."""
    init = BezierSimploid.zeros((2,), 3, n_values=3)
    loss = linear_scalarization(_location_objective)
    inputs = sample_domain(init, 100, seed=0)
    test_inputs = sample_domain(init, 1000, seed=1)
    optimum = _mean_loss(BezierSimploid((2,), 1, torch.eye(3)), loss, test_inputs)
    gaps = {
        pc: _mean_loss(
            fit_sgd(init, loss, inputs=inputs, batch_size=20, step_size=0.05, preconditioner=pc, seed=0), loss, test_inputs
        )
        - optimum
        for pc in ("gram", "none")
    }
    assert 0 <= gaps["gram"] < 0.1 * gaps["none"]


@pytest.mark.parametrize("simploid_type, degree", [((1, 1), (2, 1)), ((2, 1), 2), ((1, 1, 1), 1)])
def test_fit_sgd_simploid(simploid_type: tuple[int, ...], degree: Any) -> None:
    """fit_sgd converges to the minimizer on Bézier simploids of non-simplex types."""
    torch.manual_seed(0)
    target = BezierSimploid.randn(simploid_type, degree, n_values=2).double()
    init = BezierSimploid.zeros(simploid_type, degree, n_values=2).double()
    bs = fit_sgd(init, _squared_error(target), n_samples=200, batch_size=50, max_iter=100, step_size=0.25, seed=0)
    assert bs.control_points.dtype == torch.float64
    assert torch.allclose(bs.control_points, target.control_points, atol=1e-6)


def test_fit_sgd_space() -> None:
    """fit_sgd converges to the minimizer on a simploidal Bézier space with glued cells."""
    S, init, _ = _glued_space()
    target = SimploidalBezierSpace(S, 2, 2).double()
    torch.manual_seed(0)
    torch.nn.init.normal_(target.proper_control_points)
    space = fit_sgd(init, _squared_error(target), n_samples=300, batch_size=100, max_iter=100, step_size=0.25, seed=0)
    assert torch.allclose(space.proper_control_points, target.proper_control_points, atol=1e-6)


def test_fit_sgd_step_decay() -> None:
    init = BezierSimploid.zeros((2,), 2, n_values=3)
    loss = linear_scalarization(_location_objective)
    inputs = sample_domain(init, 100, seed=0)
    before = _mean_loss(init, loss, inputs)
    constant = fit_sgd(init, loss, inputs=inputs, max_iter=20, step_size=0.2, seed=0)
    decayed = fit_sgd(init, loss, inputs=inputs, max_iter=20, step_size=0.2, step_decay="sqrt", seed=0)
    assert _mean_loss(constant, loss, inputs) < _mean_loss(decayed, loss, inputs) < before


def test_fit_sgd_is_reproducible_and_copies() -> None:
    init = BezierSimploid.zeros((2,), 2, n_values=3)
    loss = linear_scalarization(_location_objective)
    a = fit_sgd(init, loss, batch_size=10, max_iter=10, seed=1)
    b = fit_sgd(init, loss, batch_size=10, max_iter=10, seed=1)
    assert torch.equal(a.control_points, b.control_points)
    assert (init.control_points == 0).all()
    assert torch_bsf.fit_sgd is fit_sgd


def test_fit_sgd_zero_iterations() -> None:
    init = BezierSimploid.zeros((1,), 2, n_values=1)
    bs = fit_sgd(init, linear_scalarization(lambda x: torch.cat([x, -x], dim=1)), max_iter=0)
    assert torch.equal(bs.control_points, init.control_points)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_iter=-1), "max_iter"),
        (dict(step_size=0.0), "step_size"),
        (dict(batch_size=0), "batch_size"),
        (dict(step_decay="linear"), "step_decay"),
        (dict(preconditioner="diag"), "preconditioner"),
        (dict(inputs=[]), "inputs must not be empty"),
        (dict(inputs=[torch.empty(0, 2)]), "one positive length"),
        (dict(n_samples=0), "one positive length"),
    ],
)
def test_fit_sgd_errors(kwargs: dict[str, Any], match: str) -> None:
    init = BezierSimploid.zeros((1,), 1, n_values=1)
    with pytest.raises(ValueError, match=match):
        fit_sgd(init, linear_scalarization(lambda x: torch.cat([x, -x], dim=1)), **kwargs)


def test_fit_sgd_loss_shape_error() -> None:
    init = BezierSimploid.zeros((1,), 1, n_values=1)
    with pytest.raises(ValueError, match="loss must return"):
        fit_sgd(init, lambda values, t: values, n_samples=5)


def _reference_sgd(
    init: BezierSimploid, loss: Any, t: torch.Tensor, batch_size: int, max_iter: int, step_size: float, seed: int
) -> tuple[list[tuple[torch.Tensor, float]], torch.Tensor]:
    """The control points and the minibatch loss before every iteration of fit_sgd, and the final control points."""
    precondition = torch.linalg.inv(gram_matrix(init)).to(init.control_points.dtype)
    generator = torch.Generator().manual_seed(seed + 1)
    p = init.control_points.detach().clone()
    iterates = []
    for _ in range(max_iter):
        rows = torch.randperm(len(t), generator=generator)[:batch_size]
        p_k = p.clone().requires_grad_(True)
        batch_loss = loss(init.design_matrix(t[rows]) @ p_k, t[rows]).mean()
        (grad,) = torch.autograd.grad(batch_loss, p_k)
        iterates.append((p, batch_loss.item()))
        p = p - step_size * (precondition @ grad)
    return iterates, p


def test_fit_sgd_callback_calls_and_batch_loss() -> None:
    """The callback is called once per iteration, in order, with the minibatch loss before the update."""
    init = BezierSimploid.zeros((2,), 2, n_values=3).double()
    loss = linear_scalarization(_location_objective)
    (t,) = sample_domain(init, 50, seed=0)
    t = t.double()
    calls: list[tuple[int, torch.nn.Module, float, torch.Tensor]] = []

    def callback(k: int, model: BezierSimploid, batch_loss: float) -> None:
        calls.append((k, model, batch_loss, model.control_points.detach().clone()))

    bs = fit_sgd(init, loss, inputs=[t], batch_size=10, max_iter=7, step_size=0.1, seed=3, callback=callback)
    iterates, final = _reference_sgd(init, loss, t, batch_size=10, max_iter=7, step_size=0.1, seed=3)
    assert [k for k, *_ in calls] == list(range(7))
    assert all(model is bs for _, model, _, _ in calls)
    for (_, _, batch_loss, _), (_, expected) in zip(calls, iterates, strict=True):
        assert isinstance(batch_loss, float)
        assert batch_loss == pytest.approx(expected, rel=1e-12)
    # The model passed at iteration k is already updated: it holds the control points before iteration k + 1.
    for (_, _, _, after), (before, _) in zip(calls[:-1], iterates[1:], strict=True):
        assert torch.allclose(after, before, rtol=0, atol=1e-12)
    assert torch.allclose(calls[-1][3], final, rtol=0, atol=1e-12)
    assert torch.equal(calls[-1][3], bs.control_points)


@pytest.mark.parametrize("stop_at", [0, 4])
def test_fit_sgd_callback_early_stopping(stop_at: int) -> None:
    """Returning True stops after that iteration, keeping its update."""
    init = BezierSimploid.zeros((2,), 2, n_values=3)
    loss = linear_scalarization(_location_objective)
    calls: list[int] = []

    def callback(k: int, model: BezierSimploid, batch_loss: float) -> bool:
        calls.append(k)
        return k == stop_at

    stopped = fit_sgd(init, loss, batch_size=10, max_iter=20, seed=0, callback=callback)
    expected = fit_sgd(init, loss, batch_size=10, max_iter=stop_at + 1, seed=0)
    assert calls == list(range(stop_at + 1))
    assert torch.equal(stopped.control_points, expected.control_points)
    assert not torch.equal(stopped.control_points, init.control_points)


@pytest.mark.parametrize("result", [None, False])
def test_fit_sgd_callback_does_not_change_the_result(result: bool | None) -> None:
    """A callback that does not stop gives bitwise the same model as no callback."""
    init = BezierSimploid.zeros((2,), 3, n_values=3)
    loss = linear_scalarization(_location_objective)
    calls: list[int] = []

    def callback(k: int, model: BezierSimploid, batch_loss: float) -> bool | None:
        calls.append(k)
        return result

    with_callback = fit_sgd(init, loss, batch_size=20, max_iter=30, step_decay="sqrt", seed=5, callback=callback)
    without = fit_sgd(init, loss, batch_size=20, max_iter=30, step_decay="sqrt", seed=5)
    assert calls == list(range(30))
    assert torch.equal(with_callback.control_points, without.control_points)


def test_fit_sgd_callback_records_generalization_loss() -> None:
    """A callback can evaluate the model being fitted, e.g. on held-out samples of a space."""
    S, init, _ = _glued_space()
    target = SimploidalBezierSpace(S, 2, 2).double()
    torch.manual_seed(0)
    torch.nn.init.normal_(target.proper_control_points)
    loss = _squared_error(target)
    test_inputs = sample_domain(init, 200, seed=1)
    history: list[float] = []

    def callback(k: int, model: SimploidalBezierSpace, batch_loss: float) -> None:
        assert isinstance(model, SimploidalBezierSpace)
        history.append(_mean_loss(model, loss, test_inputs))

    space = fit_sgd(init, loss, n_samples=300, batch_size=100, max_iter=20, step_size=0.25, seed=0, callback=callback)
    assert len(history) == 20
    assert history[-1] == pytest.approx(_mean_loss(space, loss, test_inputs))
    assert history[-1] < 0.01 * history[0]


def test_fit_sgd_space_keeps_cells() -> None:
    """The fitted space accepts the cells of the input space, also as keys of weights."""
    _, init, (tri, sq, edge) = _glued_space()
    loss = _squared_error(init)
    space = fit_sgd(init, loss, n_samples=20, max_iter=2, weights={tri: 1.0, sq: 1.0, edge: 1.0}, seed=0)
    assert space.simploids == init.simploids
    assert space.proper_control_points is not init.proper_control_points
    with torch.no_grad():
        space.proper_control_points.fill_(1.0)
    assert (init.proper_control_points == 0).all()
    assert torch.allclose(space.evaluate(tri, torch.tensor([[0.2, 0.3, 0.5]])), torch.ones(1, 2, dtype=torch.float64))


def test_space_default_weights_ignore_later_cells() -> None:
    """The default weights follow the structure captured by the space, not later changes to its set."""
    S, space, _ = _glued_space()
    gram = gram_matrix(space)
    S.add_simplex(2)
    assert torch.equal(gram_matrix(space), gram)
    _, ids = sample_domain(space, 50, seed=0)
    assert ids.max() < len(space.simploids)
