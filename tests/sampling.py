import math
import warnings
from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import seeds
from torch_bsf import BezierSpace
from torch_bsf._kernels._sampling import simplex_grid, simplex_random, simplex_sobol
from torch_bsf._kernels.combinatorics import _simplex_indices
from torch_bsf.sampling import _face_gram, optimal_skeleton_ratio, sample, sample_skeleton, skeleton_risk_coefficients

# ---------------------------------------------------------------------------
# simplex_grid
# ---------------------------------------------------------------------------


@example(n_params=0, degree=0)
@example(n_params=1, degree=5)
@example(n_params=3, degree=0)
@given(n_params=st.integers(0, 5), degree=st.integers(0, 6))
def test_simplex_grid_shape_and_coordinates(n_params, degree):
    result = simplex_grid(n_params, degree)
    count = 1 if n_params == 0 or degree == 0 else comb(degree + n_params - 1, n_params - 1)
    assert result.shape == (count, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    if n_params:
        assert len(torch.unique(result, dim=0)) == count
    if n_params:
        # Float32 division and n_params-term summation: at most (n_params + 1) unit roundoffs.
        bound = (n_params + 1) * torch.finfo(result.dtype).eps / 2
        assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    if degree > 0:
        # Each coordinate is an integer lattice coordinate divided by degree.
        integer_coordinates = result * degree
        assert ((integer_coordinates - integer_coordinates.round()).abs() <= degree * torch.finfo(result.dtype).eps).all()


def test_simplex_grid_invalid_n_params():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_grid(-1, 3)


# ---------------------------------------------------------------------------
# simplex_random
# ---------------------------------------------------------------------------


@example(n_params=1, n_samples=0, seed=0)
@example(n_params=1, n_samples=1, seed=0)
@given(n_params=st.integers(1, 8), n_samples=st.integers(0, 64), seed=seeds())
def test_simplex_random_shape_and_coordinates(n_params, n_samples, seed):
    result = simplex_random(n_params, n_samples, seed=seed)
    assert result.shape == (n_samples, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    # Dirichlet normalization is float64, then one float32 cast and a row reduction.
    bound = (n_params + 2) * torch.finfo(result.dtype).eps / 2
    assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    assert torch.equal(result, simplex_random(n_params, n_samples, seed=seed))


def test_simplex_random_invalid_n_params():
    with pytest.raises(ValueError, match="positive"):
        simplex_random(0, 5)


def test_simplex_random_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_random(3, -1)


def test_simplex_random_different_seeds_differ():
    """Different seeds produce different results (with very high probability)."""
    r1 = simplex_random(3, 50, seed=0)
    r2 = simplex_random(3, 50, seed=1)
    assert not torch.equal(r1, r2)


@given(seed=seeds())
def test_simplex_random_seed_does_not_mutate_global_state(seed):
    """Using seed= should not change the global numpy random state."""
    import numpy as np

    rng_state_before = np.random.get_state()
    simplex_random(3, 20, seed=seed)
    rng_state_after = np.random.get_state()
    assert isinstance(rng_state_before, tuple) and isinstance(rng_state_after, tuple)
    # Compare both the 'key' array and the position index in the MT state
    assert (rng_state_before[1] == rng_state_after[1]).all()
    assert rng_state_before[2:] == rng_state_after[2:]


# ---------------------------------------------------------------------------
# simplex_sobol
# ---------------------------------------------------------------------------

try:
    from scipy.stats import qmc  # noqa: F401
except ImportError:
    _has_scipy = False
else:
    _has_scipy = True

_scipy_skip = pytest.mark.skipif(
    not _has_scipy,
    reason="scipy.stats.qmc is required for simplex_sobol",
)


@_scipy_skip
@example(n_params=2, exponent=0, seed=0)
@given(n_params=st.integers(2, 8), exponent=st.integers(0, 6), seed=seeds())
def test_simplex_sobol_shape_and_coordinates(n_params, exponent, seed):
    n_samples = 2**exponent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = simplex_sobol(n_params, n_samples, seed=seed)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert result.shape == (n_samples, n_params)
    assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    # Sorted differences are computed in float64, then cast and summed in float32.
    bound = (n_params + 2) * torch.finfo(result.dtype).eps / 2
    assert ((result.sum(dim=1) - 1).abs() <= bound).all()
    assert torch.equal(result, simplex_sobol(n_params, n_samples, seed=seed))


@_scipy_skip
@given(n_params=st.integers(2, 8))
def test_simplex_sobol_zero_samples(n_params):
    assert simplex_sobol(n_params, 0).shape == (0, n_params)


@_scipy_skip
def test_simplex_sobol_invalid_n_params():
    with pytest.raises(ValueError, match="at least 2"):
        simplex_sobol(1, 5)


@_scipy_skip
def test_simplex_sobol_invalid_n_samples():
    with pytest.raises(ValueError, match="non-negative"):
        simplex_sobol(2, -1)


@_scipy_skip
@given(n_samples=st.integers(3, 256).filter(lambda n: n & (n - 1) != 0))
def test_simplex_sobol_non_power_of_two_warns(n_samples):
    """Non-power-of-2 sample sizes should emit exactly one UserWarning."""
    with pytest.warns(UserWarning, match="not a power of 2") as w:
        result = simplex_sobol(3, n_samples, seed=0)
    assert len(w) == 1, f"Expected exactly 1 UserWarning, got {len(w)}: {w}"
    # Result is still valid despite the warning.
    assert result.shape == (n_samples, 3)
    assert ((result.sum(dim=1) - 1).abs() <= 5 * torch.finfo(result.dtype).eps / 2).all()


@_scipy_skip
def test_simplex_sobol_different_seeds_differ():
    """Different seeds produce different samples (with very high probability)."""
    r1 = simplex_sobol(3, 64, seed=0)
    r2 = simplex_sobol(3, 64, seed=1)
    assert not torch.equal(r1, r2)


def test_simplex_sobol_scipy_import_error(monkeypatch):
    """simplex_sobol should raise ImportError when scipy is not available."""
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.stats", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.stats.qmc", raising=False)

    real_import = builtins.__import__

    def _import_without_scipy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("No module named 'scipy'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_scipy)

    # Re-import the function from a fresh copy of the module to bypass the cached import.
    import importlib

    import torch_bsf._kernels._sampling as sampling_mod

    importlib.reload(sampling_mod)

    with pytest.raises(ImportError, match="scipy"):
        sampling_mod.simplex_sobol(3, 4)


@pytest.mark.parametrize("wrap", [np.array, tuple, list, lambda x: np.array(x[0]), lambda x: np.array(x * 2)])
def test_method_rejects_non_string_choice(wrap):
    value = wrap(["sobol"])
    with pytest.raises(ValueError, match="method"):
        sample(BezierSpace.simplex(1, degree=1), 2, seed=0, method=value)


@pytest.mark.parametrize("option", ("random", "sobol"))
@pytest.mark.parametrize("string_type", [str, np.str_])
def test_method_accepts_string_choice(string_type, option):
    value = string_type(option)
    sample(BezierSpace.simplex(1, degree=1), 2, seed=0, method=value)


# ---------------------------------------------------------------------------
# skeleton_risk_coefficients and optimal_skeleton_ratio
# ---------------------------------------------------------------------------

U = 2**-53

# Table 1 of Tanaka et al. (AAAI 2020, arXiv:1906.06924): (n_vertices, degree) -> a_0, a_1, ... as printed.
_PAPER_COEFFICIENTS = {
    (2, 2): ("0.5", "1.0"),
    (3, 2): ("0.375", "3.0"),
    (4, 2): ("0.4571", "5.142"),
    (5, 2): ("0.625", "7.142"),
    (6, 2): ("0.8214", "8.928"),
    (7, 2): ("1.020", "10.5"),
    (8, 2): ("1.212", "11.87"),
    (2, 3): ("0.2666", "2.0"),
    (3, 3): ("0.1464", "3.535", "1.0"),
    (4, 3): ("0.1650", "4.714", "5.333"),
    (5, 3): ("0.2083", "6.666", "13.33"),
    (6, 3): ("0.2575", "9.740", "24.24"),
    (7, 3): ("0.3119", "13.84", "37.12"),
    (8, 3): ("0.3723", "18.73", "51.17"),
}

# Table 2 of the same paper: the minimum of N times the risk, (sum_m sqrt(a_m))^2.
_PAPER_OPTIMAL_RISKS = {
    (2, 2): "2.91421",
    (3, 2): "5.49632",
    (4, 2): "8.66660",
    (5, 2): "11.9936",
    (6, 2): "15.1663",
    (7, 2): "18.0687",
    (8, 2): "20.6799",
    (2, 3): "3.72726",
    (3, 3): "10.6472",
    (4, 3): "23.8821",
    (5, 3): "44.7548",
    (6, 3): "73.1387",
    (7, 3): "107.570",
    (8, 3): "146.206",
}


def _printed_unit(text: str) -> float:
    """One unit in the last printed digit; the paper truncates or rounds to it."""
    return 10.0 ** -len(text.partition(".")[2])


@pytest.mark.parametrize(("n_vertices", "degree"), sorted(_PAPER_COEFFICIENTS))
def test_skeleton_risk_coefficients_match_paper_tables(n_vertices, degree):
    space = BezierSpace.simplex(n_vertices - 1, degree=degree)
    coefficients = skeleton_risk_coefficients(space)
    printed = _PAPER_COEFFICIENTS[n_vertices, degree]
    assert len(coefficients) == n_vertices
    for value, text in zip(coefficients, printed, strict=False):
        assert abs(value - float(text)) < _printed_unit(text)
    assert all(value == 0 for value in coefficients[len(printed) :])
    optimum = sum(math.sqrt(a) for a in coefficients) ** 2
    text = _PAPER_OPTIMAL_RISKS[n_vertices, degree]
    assert abs(optimum - float(text)) < _printed_unit(text)


def _exact_solve(matrix: list[list[Fraction]], rhs: list[list[Fraction]]) -> list[list[Fraction]]:
    """Gauss-Jordan elimination over the rationals (the Gram blocks are positive definite)."""
    n = len(matrix)
    rows = [matrix[i][:] + rhs[i][:] for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if rows[r][col])
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = [x / rows[col][col] for x in rows[col]]
        for r in range(n):
            if r != col and rows[r][col]:
                rows[r] = [x - rows[r][col] * y for x, y in zip(rows[r], rows[col], strict=True)]
    return [row[n:] for row in rows]


def _exact_coefficients(n_vertices: int, degree: int) -> tuple[list[Fraction], list[list[list[Fraction]]]]:
    """An independent rational evaluation of the risk coefficients and the Gram matrices it solves with."""
    alpha = _simplex_indices(n_vertices, degree)
    support = [frozenset(i for i, a in enumerate(x) if a) for x in alpha]

    def gram(a: tuple[int, ...], b: tuple[int, ...], face: frozenset[int]) -> Fraction:
        k = len(face) - 1
        multinomials = Fraction(math.factorial(degree) ** 2, math.prod(math.factorial(x) for x in a + b))
        moment = Fraction(math.factorial(k) * math.prod(math.factorial(a[i] + b[i]) for i in face))
        return multinomials * moment / math.factorial(2 * degree + k)

    n = len(alpha)
    errors = [[Fraction(0)] * n for _ in range(n)]
    blocks = []
    for size in range(1, min(degree, n_vertices) + 1):
        for vertices in combinations(range(n_vertices), size):
            face = frozenset(vertices)
            interior = [i for i in range(n) if support[i] == face]
            boundary = [i for i in range(n) if support[i] < face]
            g_tt = [[gram(alpha[i], alpha[j], face) for j in interior] for i in interior]
            g_tb = [[gram(alpha[i], alpha[j], face) for j in boundary] for i in interior]
            rhs = [
                [sum((g_tb[r][c] * errors[boundary[c]][k] for c in range(len(boundary))), Fraction(0)) for k in range(n)]
                for r in range(len(interior))
            ]
            for r, row in enumerate(_exact_solve(g_tt, rhs)):
                errors[interior[r]] = [-x for x in row]
                errors[interior[r]][interior[r]] = Fraction(1)
            blocks.append((size, interior, g_tt))
    full = [[gram(a, b, frozenset(range(n_vertices))) for b in alpha] for a in alpha]
    coefficients = [Fraction(0)] * n_vertices
    for size, interior, g_tt in blocks:
        e = [[errors[i][j] for j in interior] for i in range(n)]
        ge = [[sum((full[i][k] * e[k][j] for k in range(n)), Fraction(0)) for j in range(len(interior))] for i in range(n)]
        ete = [
            [sum((e[k][i] * ge[k][j] for k in range(n)), Fraction(0)) for j in range(len(interior))]
            for i in range(len(interior))
        ]
        solved = _exact_solve(g_tt, ete)
        coefficients[size - 1] += comb(n_vertices, size) * sum(solved[i][i] for i in range(len(interior)))
    return coefficients, [full] + [g_tt for _, _, g_tt in blocks]


def _condition(matrix: list[list[Fraction]]) -> float:
    return float(torch.linalg.cond(torch.tensor([[float(x) for x in row] for row in matrix], dtype=torch.float64)).item())


@pytest.mark.parametrize(("n_vertices", "degree"), [(m, d) for m in range(2, 6) for d in range(1, 5)])
def test_skeleton_risk_coefficients_match_exact_rationals(n_vertices, degree):
    coefficients = skeleton_risk_coefficients(BezierSpace.simplex(n_vertices - 1, degree=degree))
    exact, grams = _exact_coefficients(n_vertices, degree)
    n = len(grams[0])
    kappa = max(_condition(g) for g in grams)
    # Gram entries have relative errors of at most (2M + 2) U (one rounding per binomial and the division).
    # To first order, each of the at most M levels of solves adds (2M + 2 + n) * kappa * U relative error
    # (entry perturbation plus backward-stable elimination), and the final traces add n U.
    bound = (n_vertices * (2 * n_vertices + 2 + n) * kappa + n) * U
    for value, reference in zip(coefficients, exact, strict=True):
        assert abs(value - float(reference)) <= bound * float(reference)


@given(n_vertices=st.integers(2, 12))
def test_linear_skeleton_risk_has_closed_form(n_vertices):
    # Degree one has only vertex controls, each a mean of N^(0) / M rows: a_0 = M * M * E[t_1^2] = 2M / (M + 1).
    coefficients = skeleton_risk_coefficients(BezierSpace.simplex(n_vertices - 1, degree=1))
    # The vertex Gram blocks are exactly one; the M diagonal Gram entries have relative errors of at most
    # (2M + 2) U and are summed with M more roundings.
    assert abs(coefficients[0] - 2 * n_vertices / (n_vertices + 1)) <= (3 * n_vertices + 3) * U * coefficients[0]
    assert coefficients[1:] == (0.0,) * (n_vertices - 1)


@given(n_vertices=st.integers(2, 5), degree=st.integers(1, 6))
def test_skeleton_risk_coefficients_structure(n_vertices, degree):
    space = BezierSpace.simplex(n_vertices - 1, degree=degree)
    coefficients = skeleton_risk_coefficients(space)
    assert all(a > 0 for a in coefficients[:degree]) and all(a == 0 for a in coefficients[degree:])
    if degree >= n_vertices:
        # The top face is the whole simplex and nothing depends on its controls, so a_{M-1} is
        # tr(G_TT^{-1} G_TT) = C(D - 1, M - 1), the number of interior controls, computed with the same G_TT.
        # A backward-stable solve gives each diagonal entry an error of at most about n * kappa * U.
        interior = comb(degree - 1, n_vertices - 1)
        alpha = torch.tensor(_simplex_indices(n_vertices, degree))
        g_tt = _face_gram(alpha[(alpha > 0).all(1)], alpha[(alpha > 0).all(1)], range(n_vertices), degree)
        kappa = torch.linalg.cond(g_tt).item()
        assert abs(coefficients[-1] - interior) <= 2 * interior * (interior * kappa + 1) * U * interior
    ratio = optimal_skeleton_ratio(space)
    # Square roots, a sum of M terms and divisions: a few M roundings in total.
    assert all(r >= 0 for r in ratio) and abs(sum(ratio) - 1) <= 4 * n_vertices * U
    optimum = sum(math.sqrt(a) for a in coefficients) ** 2
    # Cauchy-Schwarz: the optimal ratio beats any other, e.g. equal shares on the skeletons with controls.
    assert optimum <= sum(a * degree for a in coefficients if a) * (1 + 8 * n_vertices * U)
    achieved = sum(a / r for a, r in zip(coefficients, ratio, strict=True) if a)
    assert abs(achieved - optimum) <= 8 * n_vertices * U * optimum


def test_skeleton_risk_coefficients_reject_ill_conditioned_degrees():
    with pytest.raises(ValueError, match="degree is too large"):
        skeleton_risk_coefficients(BezierSpace.simplex(2, degree=40))


@pytest.mark.parametrize(
    "space",
    [BezierSpace.simplex(0, degree=2), BezierSpace.simplex(2, degree=0), BezierSpace.simploid((1, 1), degree=2)],
)
def test_skeleton_methods_reject_other_spaces(space):
    with pytest.raises(ValueError, match="single simplex"):
        skeleton_risk_coefficients(space)
    with pytest.raises(ValueError, match="single simplex"):
        sample_skeleton(space, 4)


def test_skeleton_methods_reject_non_spaces():
    with pytest.raises(TypeError, match="BezierSpace"):
        optimal_skeleton_ratio(2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sample_skeleton
# ---------------------------------------------------------------------------


def _face_counts(points, n_vertices):
    support = points.parameters > 0
    return {
        face: int((support == torch.tensor([i in face for i in range(n_vertices)])).all(1).sum())
        for size in range(1, n_vertices + 1)
        for face in combinations(range(n_vertices), size)
    }


@example(n_vertices=2, degree=2, n=0, seed=0, method="random")
@example(n_vertices=3, degree=3, n=1, seed=0, method="sobol")
@given(
    n_vertices=st.integers(2, 5),
    degree=st.integers(1, 4),
    n=st.integers(0, 300),
    seed=seeds(),
    method=st.sampled_from(["random", "sobol"]),
)
def test_sample_skeleton_allocates_optimal_counts(n_vertices, degree, n, seed, method):
    space = BezierSpace.simplex(n_vertices - 1, degree=degree)
    points = sample_skeleton(space, n, seed=seed, method=method)
    t = points.parameters
    assert t.shape == (n, n_vertices) and t.dtype == torch.float64 and (points.cell_ids == 0).all()
    assert (t >= 0).all()
    # Normalizing at most n_vertices exponentials: one division and a sum of n_vertices terms per row.
    assert ((t.sum(1) - 1).abs() <= (n_vertices + 1) * U * 2).all()
    counts = _face_counts(points, n_vertices)
    assert sum(counts.values()) == n
    ratio = optimal_skeleton_ratio(space)
    for size in range(1, n_vertices + 1):
        level = [counts[face] for face in combinations(range(n_vertices), size)]
        # Largest-remainder rounding: each skeleton gets the floor or ceiling of its quota, up to quota roundoff.
        assert math.floor(n * ratio[size - 1] - 1e-9) <= sum(level) <= math.ceil(n * ratio[size - 1] + 1e-9)
        assert max(level) - min(level) <= 1
    assert torch.equal(t, sample_skeleton(space, n, seed=seed, method=method).parameters)


def test_sample_skeleton_uses_explicit_ratio_and_local_generator():
    space = BezierSpace.simplex(2, degree=3)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    points = sample_skeleton(space, 10, ratio=[0, 1, 1], seed=4)
    assert torch.equal(state, torch.get_rng_state())
    counts = _face_counts(points, 3)
    assert [counts[f] for f in [(0, 1), (0, 2), (1, 2)]] in ([2, 2, 1], [2, 1, 2], [1, 2, 2])
    assert counts[(0, 1, 2)] == 5 and counts[(0,)] == counts[(1,)] == counts[(2,)] == 0
    assert not torch.equal(points.parameters, sample_skeleton(space, 10, ratio=[0, 1, 1], seed=5).parameters)


def test_sample_skeleton_faces_are_uniform():
    # On an edge, the first coordinate is uniform: its mean 1/2 and variance 1/12 have standard errors
    # sqrt(1/12 / n) and sqrt(1/180 / n); five standard errors make a seeded failure practically impossible.
    space = BezierSpace.simplex(1, degree=2)
    n = 20000
    t = sample_skeleton(space, n, ratio=[0, 1], seed=0).parameters[:, 0]
    assert abs(t.mean().item() - 0.5) <= 5 * math.sqrt(1 / 12 / n)
    assert abs(t.var().item() - 1 / 12) <= 5 * math.sqrt(1 / 180 / n)


@pytest.mark.parametrize("ratio", [[1.0], [1.0, -1.0], [0.0, 0.0], [float("nan"), 1.0], [[1.0, 1.0]]])
def test_sample_skeleton_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError, match="ratio"):
        sample_skeleton(BezierSpace.simplex(1, degree=2), 4, ratio=ratio)


def test_sample_skeleton_rejects_bad_arguments():
    space = BezierSpace.simplex(1, degree=2)
    with pytest.raises(ValueError, match="method"):
        sample_skeleton(space, 4, method="grid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        sample_skeleton(space, 4, method=np.array(["sobol"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must"):
        sample_skeleton(space, -1)
