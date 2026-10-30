import math
import warnings
from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations, count
from math import comb
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import degrees, seeds, simploid_types
from torch_bsf import BezierSpace, Samples, SimploidalSet, fit_skeleton_regression
from torch_bsf._kernels._sampling import simplex_grid, simplex_random, simplex_sobol
from torch_bsf._kernels.combinatorics import _indices, _simplex_indices
from torch_bsf._skeleton import _blocks, _faces_with_interior_controls
from torch_bsf.sampling import (
    _RISK_RELATIVE_ERROR,
    _exponentials,
    _face_gram,
    _n_skeleton_faces,
    _skeleton_counts,
    _skeleton_fractions,
    optimal_skeleton_ratio,
    sample,
    sample_skeleton,
    skeleton_risk_coefficients,
)

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


@pytest.mark.parametrize("sampler", [sample, sample_skeleton])
@pytest.mark.parametrize("n", [0, 1, 2, 50])
@pytest.mark.parametrize("seed", [0, 3])
def test_sobol_sampling_is_independent_of_default_dtype(sampler, n, seed):
    space = BezierSpace.simplex(2, degree=3)
    original_dtype = torch.get_default_dtype()
    random_state = torch.random.get_rng_state().clone()
    results = []
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            with patch("torch.set_default_dtype", side_effect=AssertionError("sampling must not change default dtype")):
                results.append(sampler(space, n, seed=seed, method="sobol"))
            assert torch.get_default_dtype() == dtype
    finally:
        torch.set_default_dtype(original_dtype)
    assert torch.equal(torch.random.get_rng_state(), random_state)
    assert all(result.parameters.dtype == torch.float64 for result in results)
    assert results[0].parameters.shape == (n, space.width)
    assert torch.equal(results[0].parameters, results[1].parameters)
    assert torch.equal(results[0].cell_ids, results[1].cell_ids)


@pytest.mark.parametrize("n", [0, 1, 2, 16])
@pytest.mark.parametrize("seed", [0, 3])
def test_sobol_exponentials_match_float64_reference(n, seed):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        generator = torch.Generator().manual_seed(seed)
        local_seed = int(torch.randint(2**31, (1,), generator=generator))
        engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=local_seed)
        uniforms = engine.draw(n, dtype=torch.float64) if n else torch.empty(0, 3, dtype=torch.float64)
        expected = -uniforms.clamp_min(torch.finfo(torch.float64).tiny).log()
        torch.set_default_dtype(torch.float32)
        actual = _exponentials(torch.Generator().manual_seed(seed), n, 3, "sobol")
    finally:
        torch.set_default_dtype(original_dtype)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, expected)


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


def _bernstein_moment(a: tuple[int, ...], b: tuple[int, ...], face: frozenset[int], degree: int) -> Fraction:
    """The Gram entry of the Bernstein polynomials ``a`` and ``b`` under the uniform distribution on ``face``."""
    k = len(face) - 1
    multinomials = Fraction(math.factorial(degree) ** 2, math.prod(math.factorial(x) for x in a + b))
    moment = Fraction(math.factorial(k) * math.prod(math.factorial(a[i] + b[i]) for i in face))
    return multinomials * moment / math.factorial(2 * degree + k)


@cache  # Pure; callers only read the returned lists.
def _exact_coefficients(
    n_vertices: int, degree: int
) -> tuple[list[Fraction], list[list[list[Fraction]]], list[list[Fraction]]]:
    """An independent rational evaluation of the risk coefficients, the Gram matrices it solves with, and the errors.

    ``errors[i][k]`` is the error of control ``i`` per unit error of the interior control ``k`` of its face, in the
    order of :func:`_simplex_indices`.
    """
    alpha = _simplex_indices(n_vertices, degree)
    support = [frozenset(i for i, a in enumerate(x) if a) for x in alpha]
    n = len(alpha)
    errors = [[Fraction(0)] * n for _ in range(n)]
    blocks = []
    for size in range(1, min(degree, n_vertices) + 1):
        for vertices in combinations(range(n_vertices), size):
            face = frozenset(vertices)
            interior = [i for i in range(n) if support[i] == face]
            boundary = [i for i in range(n) if support[i] < face]
            g_tt = [[_bernstein_moment(alpha[i], alpha[j], face, degree) for j in interior] for i in interior]
            g_tb = [[_bernstein_moment(alpha[i], alpha[j], face, degree) for j in boundary] for i in interior]
            rhs = [
                [sum((g_tb[r][c] * errors[boundary[c]][k] for c in range(len(boundary))), Fraction(0)) for k in range(n)]
                for r in range(len(interior))
            ]
            for r, row in enumerate(_exact_solve(g_tt, rhs)):
                errors[interior[r]] = [-x for x in row]
                errors[interior[r]][interior[r]] = Fraction(1)
            blocks.append((size, interior, g_tt))
    full = [[_bernstein_moment(a, b, frozenset(range(n_vertices)), degree) for b in alpha] for a in alpha]
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
    return coefficients, [full] + [g_tt for _, _, g_tt in blocks], errors


def _condition(matrix: list[list[Fraction]]) -> float:
    return float(torch.linalg.cond(torch.tensor([[float(x) for x in row] for row in matrix], dtype=torch.float64)).item())


@pytest.mark.parametrize(("n_vertices", "degree"), [(m, d) for m in range(2, 6) for d in range(1, 5)])
def test_skeleton_risk_coefficients_match_exact_rationals(n_vertices, degree):
    coefficients = skeleton_risk_coefficients(BezierSpace.simplex(n_vertices - 1, degree=degree))
    exact, grams, _ = _exact_coefficients(n_vertices, degree)
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
        inner = alpha[(alpha > 0).all(1)]
        g_tt = _face_gram(inner, inner, range(n_vertices), (n_vertices - 1,), (degree,))
        kappa = torch.linalg.cond(g_tt).item()
        assert abs(coefficients[-1] - interior) <= 2 * interior * (interior * kappa + 1) * U * interior
    ratio = optimal_skeleton_ratio(space)
    # Square roots, a sum of M terms and divisions: a few M roundings in total.
    assert all(r >= 0 for r in ratio) and abs(sum(ratio) - 1) <= 4 * n_vertices * U
    optimum = sum(math.sqrt(a) for a in coefficients) ** 2
    # Cauchy-Schwarz: the optimal ratio beats any other, e.g. equal shares on the k skeletons with controls.
    k = min(degree, n_vertices)
    assert optimum <= sum(a * k for a in coefficients if a) * (1 + 8 * n_vertices * U)
    achieved = sum(a / r for a, r in zip(coefficients, ratio, strict=True) if a)
    assert abs(achieved - optimum) <= 8 * n_vertices * U * optimum


# The table of docs/advanced/skeleton_sampling.rst: degree -> N R* / (sigma^2 L) for M = 2, ..., 6, as printed.
_DOCS_OPTIMAL_RISKS = {
    1: ("1.333", "1.5", "1.6", "1.667", "1.714"),
    2: ("2.914", "5.496", "8.667", "11.99", "15.17"),
    3: ("3.727", "10.65", "23.88", "44.75", "73.14"),
    4: ("4.581", "15.07", "43.04", "104.9", "220.2"),
    5: ("5.467", "20.20", "64.12", "184.3", "471.8"),
    6: ("6.374", "26.17", "90.36", "283.9", "824.0"),
}

# Entries of that table in which inductive skeleton fitting beats all-at-once least squares.
_DOCS_SKELETON_WINS = {(m, 1) for m in range(2, 7)} | {(m, 2) for m in range(2, 7)} | {(2, d) for d in range(1, 7)}
_DOCS_SKELETON_WINS |= {(3, 5), (3, 6)}


@pytest.mark.parametrize("degree", sorted(_DOCS_OPTIMAL_RISKS))
@pytest.mark.parametrize("n_vertices", range(2, 7))
def test_skeleton_risk_documented_comparison(n_vertices, degree):
    text = _DOCS_OPTIMAL_RISKS[degree][n_vertices - 2]
    optimum = sum(math.sqrt(a) for a in skeleton_risk_coefficients(BezierSpace.simplex(n_vertices - 1, degree=degree))) ** 2
    # Printed to four significant digits, rounded to nearest.
    assert abs(optimum - float(text)) <= _printed_unit(text) / 2
    # All-at-once least squares on uniform rows has risk (number of controls) / N.
    all_at_once = comb(degree + n_vertices - 1, degree)
    assert (optimum < all_at_once) == ((n_vertices, degree) in _DOCS_SKELETON_WINS)
    # The winner is decided beyond the error of the optimum (15.07 versus 15 is closest): relative errors of at
    # most _RISK_RELATIVE_ERROR in each a_m halve under the square roots and double under the square.
    assert abs(optimum - all_at_once) > 2 * _RISK_RELATIVE_ERROR * optimum


# P_{D,m} as printed in docs/advanced/skeleton_risk_verification.rst:
# (D, m) -> (factor, coefficients from the highest power of M), so that
# a_m = C(M, m + 1)^2 P_{D,m}(M) / (M)_{2D}. _counted_risk_polynomial regenerates it.
_RISK_POLYNOMIALS = {
    (1, 0): (Fraction(2), (1,)),
    (2, 0): (Fraction(3, 2), (3, -15, 28)),
    (2, 1): (Fraction(120), (1,)),
    (3, 0): (Fraction(1, 2), (9, -114, 835, -2610, 3320)),
    (3, 1): (Fraction(360), (3, -21, 58)),
    (3, 2): (Fraction(20160), (1,)),
    (4, 0): (Fraction(5, 8), (5, -115, 1825, -14765, 71250, -174840, 181152)),
    (4, 1): (Fraction(105, 2), (63, -1050, 9481, -38654, 67512)),
    (4, 2): (Fraction(136080), (3, -27, 94)),
    (4, 3): (Fraction(6652800), (1,)),
    (5, 0): (Fraction(1, 128), (225, -8100, 223650, -3352440, 33885265, -210684820, 810530924, -1692775952, 1526667648)),
    (5, 1): (Fraction(840), (7, -203, 3573, -34441, 201644, -629700, 867312)),
    (5, 2): (Fraction(56700), (45, -930, 9971, -49694, 106152)),
    (5, 3): (Fraction(79833600), (3, -33, 136)),
    (5, 4): (Fraction(3632428800), (1,)),
    (6, 0): (
        Fraction(7, 1536),
        (
            189,
            -9765,
            416010,
            -9917418,
            170241981,
            -1939736421,
            15319824604,
            -79461497164,
            261855225632,
            -485208011456,
            394380100608,
        ),
    ),
    (6, 1): (
        Fraction(63, 40),
        (4725, -207900, 5933250, -99912240, 1132957965, -8261943780, 38361752764, -100723167440, 118578731904),
    ),
    (6, 2): (Fraction(155925), (55, -1925, 38401, -431519, 2952160, -11026484, 18273744)),
    (6, 3): (Fraction(21621600), (117, -2886, 35695, -208734, 519368)),
    (6, 4): (Fraction(68108040000), (3, -39, 184)),
    (6, 5): (Fraction(2964061900800), (1,)),
    (7, 0): (
        Fraction(1, 10368),
        (
            3969,
            -277830,
            16911909,
            -585892314,
            15245701191,
            -274563067842,
            3631411308511,
            -34078439471286,
            227106974533684,
            -1028193132036840,
            3014866953859072,
            -5083595030097792,
            3804385671687168,
        ),
    ),
    (7, 1): (
        Fraction(315, 8),
        (
            189,
            -11655,
            488100,
            -12550518,
            229047537,
            -2894482743,
            25831659094,
            -156730040300,
            621915070424,
            -1437930406240,
            1507450665216,
        ),
    ),
    (7, 2): (
        Fraction(31185, 4),
        (2475, -128700, 4016430, -76100640, 969732275, -8107098420, 43614175604, -135327928048, 189756997248),
    ),
    (7, 3): (Fraction(151351200), (91, -3731, 83699, -1072645, 8358906, -36008544, 68795424)),
    (7, 4): (Fraction(363242880000), (9, -258, 3607, -24086, 67896)),
    (7, 5): (Fraction(80029671321600), (3, -45, 238)),
    (7, 6): (Fraction(3379030566912000), (1,)),
    (8, 0): (
        Fraction(1, 1536),
        (
            243,
            -22113,
            1822905,
            -86413257,
            3170012139,
            -82654759071,
            1640539616043,
            -24125599635171,
            266382561743494,
            -2157428183555988,
            12638903353801912,
            -51381774261439680,
            137262734136374272,
            -214224590422872576,
            149755742713294848,
        ),
    ),
    (8, 1): (
        Fraction(55, 384),
        (
            43659,
            -3580038,
            206363619,
            -7485895494,
            199202107797,
            -3823026542874,
            54535243641473,
            -568795558812642,
            4306623724542524,
            -22811488486847928,
            80426058871064384,
            -168310145729971584,
            161038581507542016,
        ),
    ),
    (8, 2): (
        Fraction(27027, 16),
        (
            19305,
            -1383525,
            61604400,
            -1734571410,
            34500231285,
            -483666978405,
            4831926907690,
            -33375332254980,
            152661774969416,
            -413966621739552,
            513284796940032,
        ),
    ),
    (8, 3): (
        Fraction(4729725, 2),
        (20475, -1228500, 42200550, -891274440, 12638555275, -118800088180, 722231576084, -2559256843376, 4107713331840),
    ),
    (8, 4): (Fraction(1543782240000), (17, -799, 19941, -286433, 2496802, -12115512, 26028000)),
    (8, 5): (Fraction(31678411564800), (171, -5586, 86965, -650590, 2040000)),
    (8, 6): (Fraction(124179373334016000), (3, -51, 298)),
    (8, 7): (Fraction(5129368400572416000), (1,)),
}


def _risk_polynomial_coefficients(degree: int, m: int) -> list[Fraction]:
    """The coefficients of P_{D,m}, from the highest power of M."""
    factor, coefficients = _RISK_POLYNOMIALS[degree, m]
    return [factor * c for c in coefficients]


def _risk_polynomial(degree: int, m: int, n_vertices: int) -> Fraction:
    value = Fraction(0)
    for c in _risk_polynomial_coefficients(degree, m):
        value = value * n_vertices + c
    return value


def _rising(n_vertices: int, k: int) -> int:
    """The rising factorial (M)_k."""
    return math.prod(range(n_vertices, n_vertices + k))


def _falling_polynomial(offset: int, s: int) -> list[Fraction]:
    """The coefficients of (M - offset)(M - offset - 1) ... (M - offset - s + 1), lowest power of M first."""
    coefficients = [Fraction(1)]
    for i in range(s):
        shifted = [Fraction(0)] + coefficients
        coefficients = [x - (offset + i) * y for x, y in zip(shifted, coefficients + [Fraction(0)], strict=True)]
    return coefficients


def _tabulated_coefficient(degree: int, m: int, n_vertices: int) -> Fraction:
    return comb(n_vertices, m + 1) ** 2 * _risk_polynomial(degree, m, n_vertices) / _rising(n_vertices, 2 * degree)


def _pair_representatives(
    weighted: dict[tuple[int, ...], list[Fraction]], face_size: int, group_sizes: tuple[int, ...]
) -> list[tuple[tuple[int, ...], list[Fraction], int]]:
    # The stabilizer of a permutes positions with equal outside exponents. Sum one b per such orbit.
    groups = []
    start = face_size
    for size in group_sizes:
        groups.append(tuple(range(start, start + size)))
        start += size
    result = []
    for b, w in weighted.items():
        if any(b[i] < b[j] for group in groups for i, j in zip(group, group[1:], strict=False)):
            continue
        multiplicity = math.prod(
            math.factorial(len(group)) // math.prod(math.factorial(c) for c in Counter(b[i] for i in group).values())
            for group in groups
        )
        result.append((b, w, multiplicity))
    return result


def _counted_risk_polynomial(degree: int, m: int) -> list[Fraction]:
    """P_{D,m} by the counting in the proof of the rational dependence on M, lowest power of M first.

    The errors from the face I = {0, ..., m} reach controls with at most D - m - 1 vertices outside I, so every class
    of pairs of such controls occurs on a simplex with V = 2D - m - 1 vertices. A class with s vertices outside I
    occurs (M - m - 1)(M - m - 2) ... (M - m - s) / ((V - m - 1) ... (V - m - s)) times as often on M vertices as
    on V. Propagation reduction constructs E_I from the solve on m + 2 vertices. The pair sum keeps one
    representative a per orbit of permutations outside I. Its multiplicity is q! / prod_k n_k!, where
    q = V - m - 1 and n_k counts outside exponents equal to k, including zeros. For fixed a, keep one b per
    orbit of its stabilizer, multiplying the same multinomial formula within each equal-exponent group.
    """
    n_vertices = 2 * degree - m - 1
    face = tuple(range(m + 1))
    # Propagation reduction: solve on just one extra vertex, then distribute its exponent.
    local = _simplex_indices(m + 2, degree)
    interior = [a for a in local if all(a[i] for i in face) and not a[-1]]
    top = [a for a in local if all(a)]
    n = len(interior)
    g_tt = [[_bernstein_moment(a, b, frozenset(range(m + 2)), degree) for b in top] for a in top]
    rhs = [[-_bernstein_moment(a, b, frozenset(range(m + 2)), degree) for b in interior] for a in top]
    local_errors = dict(zip(top, _exact_solve(g_tt, rhs), strict=True))
    local_errors.update({a: [Fraction(int(i == j)) for j in range(n)] for i, a in enumerate(interior)})
    errors = {}
    for a in _simplex_indices(n_vertices, degree):
        if all(a[i] for i in face):
            outside = a[m + 1 :]
            scale = Fraction(sum(outside) + 1, math.prod(x + 1 for x in outside))
            errors[a] = [scale * e for e in local_errors[a[: m + 1] + (sum(outside),)]]
    interior = [a[: m + 1] + (0,) * (n_vertices - m - 1) for a in interior]
    g_face = [[_bernstein_moment(a, b, frozenset(face), degree) for b in interior] for a in interior]
    covariance = _exact_solve(g_face, [[Fraction(int(i == j)) for j in range(n)] for i in range(n)])
    weighted = {
        b: [sum((covariance[i][j] * e[j] for j in range(n)), Fraction(0)) for i in range(n)] for b, e in errors.items()
    }

    representatives: dict[tuple[int, ...], list[tuple[tuple[int, ...], list[Fraction], int]]] = {}
    by_outside: dict[int, Fraction] = {}
    for a, e in errors.items():
        outside = a[m + 1 :]
        if outside != tuple(sorted(outside, reverse=True)):
            continue
        # Fix a representative under permutations outside I; summing all b makes the summand invariant.
        multiplicity = math.factorial(len(outside)) // math.prod(math.factorial(k) for k in Counter(outside).values())
        groups = tuple(Counter(outside).values())
        if groups not in representatives:
            representatives[groups] = _pair_representatives(weighted, m + 1, groups)
        for b, w, b_multiplicity in representatives[groups]:
            # Times 1 / (M)_{2D}, this is C_{ab} times the Gram entry of a and b on the whole simplex.
            value = sum((x * y for x, y in zip(e, w, strict=True)), Fraction(0)) * Fraction(
                math.factorial(degree) ** 2 * math.prod(math.factorial(x + y) for x, y in zip(a, b, strict=True)),
                math.prod(math.factorial(x) for x in a + b),
            )
            s = sum(1 for v in range(m + 1, n_vertices) if a[v] or b[v])
            by_outside[s] = by_outside.get(s, Fraction(0)) + multiplicity * b_multiplicity * value
    polynomial = [Fraction(0)] * (2 * (degree - m - 1) + 1)
    for s, value in by_outside.items():
        scale = value / math.prod(range(n_vertices - m - s, n_vertices - m))
        for power, c in enumerate(_falling_polynomial(m + 1, s)):
            polynomial[power] += scale * c
    return polynomial


@pytest.mark.parametrize(("degree", "m"), sorted(_RISK_POLYNOMIALS))
def test_risk_polynomial_table_is_the_counting(degree, m):
    assert _counted_risk_polynomial(degree, m) == _risk_polynomial_coefficients(degree, m)[::-1]


@pytest.mark.parametrize(
    ("n_vertices", "degree"),
    [(n, d) for n in range(2, 7) for d in range(1, 5)]
    + [(n, 5) for n in range(2, 6)]
    + [(n, d) for n in (2, 3) for d in (6, 7, 8)]
    + [(4, 6)],
)
def test_risk_polynomials_match_exact_coefficients(n_vertices, degree):
    # The table comes from counting classes of control pairs; the helper evaluates the definition directly.
    exact, _, _ = _exact_coefficients(n_vertices, degree)
    for m in range(degree):
        assert _tabulated_coefficient(degree, m, n_vertices) == (exact[m] if m < n_vertices else 0)


@pytest.mark.parametrize(("n_vertices", "degree"), [(n, d) for n in (7, 8) for d in range(1, 6)] + [(6, 5)])
def test_risk_polynomials_match_higher_dimensions(n_vertices, degree):
    coefficients = skeleton_risk_coefficients(BezierSpace.simplex(n_vertices - 1, degree=degree))
    for m in range(degree):
        reference = float(_tabulated_coefficient(degree, m, n_vertices))
        # The coefficients are guaranteed to _RISK_RELATIVE_ERROR; converting the reference adds one rounding.
        assert abs(coefficients[m] - reference) <= (_RISK_RELATIVE_ERROR + U) * reference


@pytest.mark.parametrize("degree", range(1, 9))
def test_risk_polynomials_match_the_closed_forms(degree):
    for n_vertices in range(degree, 30):
        # Faces with D vertices.
        rising = _rising(n_vertices, 2 * degree)
        top = Fraction(comb(n_vertices, degree) ** 2 * math.factorial(3 * degree - 1), math.factorial(degree - 1) * rising)
        assert _tabulated_coefficient(degree, degree - 1, n_vertices) == top
    for n_vertices in range(degree - 1, 30) if degree >= 2 else ():
        # Faces with D - 1 vertices, including M = D - 1, where the face is the whole simplex.
        rising = _rising(n_vertices, 2 * degree)
        quadratic = 3 * n_vertices**2 - (6 * degree + 3) * n_vertices + 3 * degree**2 + 15 * degree - 14
        second = Fraction(
            comb(n_vertices, degree - 1) ** 2 * (degree - 1) * math.factorial(3 * degree - 3) * quadratic,
            4 * math.factorial(degree - 2) * rising,
        )
        assert _tabulated_coefficient(degree, degree - 2, n_vertices) == second


def _large_dimension_limit(degree: int, m: int) -> Fraction:
    """The uniform limit in the theory, including vertices and the two highest active skeletons."""
    size = m + 1
    n = degree - size
    return Fraction(
        math.factorial(3 * size - 1) * comb(n + 3 * size - 1, n) ** 2,
        math.factorial(size - 1) * math.factorial(size) ** 2 * 4**n,
    )


@pytest.mark.parametrize("degree", range(1, 9))
def test_risk_polynomial_limits_in_high_dimensions(degree):
    # The degree bound 2(D - m - 1) is attained, so a_m tends to the leading coefficient over (m + 1)!^2.
    limits = []
    for m in range(degree):
        coefficients = _risk_polynomial_coefficients(degree, m)
        assert len(coefficients) == 2 * (degree - m - 1) + 1
        limits.append(coefficients[0] / math.factorial(m + 1) ** 2)
        assert limits[-1] == _large_dimension_limit(degree, m)
    assert limits[0] == Fraction(2 * degree**2 * (degree + 1) ** 2, 4**degree)
    top = Fraction(math.factorial(3 * degree - 1), math.factorial(degree - 1) * math.factorial(degree) ** 2)
    assert limits[degree - 1] == top
    if degree >= 2:
        second = Fraction(3 * math.factorial(3 * degree - 3), 4 * math.factorial(degree - 2) ** 2 * math.factorial(degree - 1))
        assert limits[degree - 2] == second
    # The exact coefficients converge too, with a residual c / M + O(1 / M^2): doubling M halves it.
    residual, half = (_tabulated_coefficient(degree, 0, n) - limits[0] for n in (10**6, 2 * 10**6))
    assert residual and abs(2 * half - residual) < abs(residual) / 1000


def _triangle_jacobi_modes(degree: int) -> list[dict[tuple[int, ...], Fraction]]:
    """Edge-error modes via the documented homogeneous Jacobi sums, converted to Bernstein controls."""
    n = degree - 2
    modes = []
    for k in range(n + 1):
        polynomial: dict[tuple[int, ...], int] = {}
        for h in range(k + 1):
            angular = (-1) ** (k - h) * comb(k + 2, h) * comb(k + 2, k - h)
            for j in range(n - k + 1):
                radial = (-1) ** (n - k - j) * comb(n - k, j) * comb(n + k + 5, n - k - j)
                for ell in range(j + 1):
                    alpha = (1 + h + ell, 1 + k - h + j - ell, n - k - j)
                    polynomial[alpha] = polynomial.get(alpha, 0) + angular * radial * comb(j, ell)
        modes.append(
            {
                alpha: Fraction(value * math.prod(math.factorial(a) for a in alpha), math.factorial(degree) * (alpha[2] + 1))
                for alpha, value in polynomial.items()
            }
        )
    return modes


# Every triangular edge mode and its reconstructed risk, not just a few fitted scalar values.
_TRIANGLE_EDGE_RISKS = {
    2: Fraction(3),
    3: Fraction(99, 28),
    4: Fraction(71, 20),
    5: Fraction(188, 55),
    6: Fraction(81259, 25025),
    7: Fraction(123021, 40040),
    8: Fraction(179861, 61880),
    9: Fraction(67983599, 24690120),
    10: Fraction(2600151, 994840),
}


def _harmonic(n: int) -> Fraction:
    return sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))


def _triangle_edge_risk(degree: int) -> Fraction:
    return 18 * (_harmonic(degree - 1) + _harmonic(2 * degree + 2) - _harmonic(degree + 3)) / (2 * degree + 3)


@pytest.mark.parametrize("degree", sorted(_TRIANGLE_EDGE_RISKS))
def test_triangle_jacobi_extension_matches_exact_propagation_and_risk(degree):
    alpha = _simplex_indices(3, degree)
    edge = [i for i, a in enumerate(alpha) if a[0] and a[1] and not a[2]]
    modes = _triangle_jacobi_modes(degree)
    exact, grams, errors = _exact_coefficients(3, degree)
    boundary = [[mode.get(alpha[i], Fraction(0)) for mode in modes] for i in edge]
    identity = [[Fraction(int(i == j)) for j in range(len(edge))] for i in range(len(edge))]
    inverse = _exact_solve(boundary, identity)
    # Recover all propagation columns from their independent Jacobi boundary modes.
    reconstructed = [
        [
            sum((mode.get(a, Fraction(0)) * inverse[k][j] for k, mode in enumerate(modes)), Fraction(0))
            for j in range(len(edge))
        ]
        for a in alpha
    ]
    assert reconstructed == [[row[j] for j in edge] for row in errors]
    gram_edge = [[_bernstein_moment(alpha[i], alpha[j], frozenset((0, 1)), degree) for j in edge] for i in edge]
    gram_full = grams[0]
    ge = [
        [sum((g * e[j] for g, e in zip(row, reconstructed, strict=True)), Fraction(0)) for j in range(len(edge))]
        for row in gram_full
    ]
    ete = [
        [sum((e[i] * g[j] for e, g in zip(reconstructed, ge, strict=True)), Fraction(0)) for j in range(len(edge))]
        for i in range(len(edge))
    ]
    trace = _exact_solve(gram_edge, ete)
    # Three equivalent edges, and C(3, 2) from the allocation convention.
    risk = 9 * sum(trace[i][i] for i in range(len(edge)))
    assert risk == exact[1] == _TRIANGLE_EDGE_RISKS[degree] == _triangle_edge_risk(degree)


def _codimension_one_risk(size: int, degree: int) -> Fraction:
    n = degree - size
    return (
        size
        * (size + 1) ** 2
        * sum(
            (
                Fraction(comb(k + size - 2, size - 2) if size >= 2 else int(k == 0), (n - k + 1) * (n + k + 3 * size))
                for k in range(n + 1)
            ),
            Fraction(0),
        )
    )


@pytest.mark.parametrize(("size", "degree"), [(s, d) for s in range(1, 5) for d in range(s, s + 4)])
def test_codimension_one_norm_sum_matches_general_scalar_sum(size, degree):
    assert _codimension_one_risk(size, degree) == _scalar_risk_sum(size + 1, degree, size)


@pytest.mark.parametrize("degree", [2, 3, 8, 9, 10, 20, 50, 100])
def test_triangle_edge_harmonics_match_the_single_rational_sum(degree):
    assert _triangle_edge_risk(degree) == _codimension_one_risk(2, degree)


def _curve_error(degree: int, k: int) -> Fraction:
    """The Bernstein coefficient b_k of the error along an edge from a unit error at a vertex, ``k`` = its exponent."""
    return Fraction((-1) ** (degree - k) * k * comb(degree + 2, k + 1), degree * (degree + 2))


@cache
def _vertex_kernel_power(s: int, p: int, q: int) -> Fraction:
    """[x^p y^q](F - 1)^s of thm-skeleton-vertex-sum, using only rational arithmetic."""
    if s == 0:
        return Fraction(int(p == q == 0))
    if s > p + q:
        return Fraction(0)
    return sum(
        (
            Fraction(comb(i + j, i), (i + 1) * (j + 1)) * _vertex_kernel_power(s - 1, p - i, q - j)
            for i in range(p + 1)
            for j in range(q + 1)
            if i + j
        ),
        Fraction(0),
    )


def _vertex_kernel(r: int, p: int, q: int) -> Fraction:
    return sum((comb(r, s) * _vertex_kernel_power(s, p, q) for s in range(min(r, p + q) + 1)), Fraction(0))


def _scalar_risk_sum(n_vertices: int, degree: int, size: int) -> Fraction:
    """The scalar angular-mode sum, with no propagation solve or Gram inverse."""
    n = degree - size
    total = Fraction(0)
    for k in range(n + 1):
        multiplicity = comb(k + size - 2, size - 2) if size >= 2 else int(k == 0)
        radial = [(-1) ** h * comb(n - k, h) * comb(n + k + 3 * size - 1, h) for h in range(n - k + 1)]
        total += multiplicity * sum(
            (
                radial[h]
                * radial[ell]
                * math.factorial(2 * degree + size - h - ell - 1)
                * math.factorial(h)
                * math.factorial(ell)
                * _vertex_kernel(n_vertices - size, h, ell)
                for h in range(n - k + 1)
                for ell in range(n - k + 1)
            ),
            Fraction(0),
        )
    return comb(n_vertices, size) ** 2 * total / (math.factorial(size - 1) * _rising(n_vertices, 2 * degree))


def _third_face_polynomial(degree: int, n_vertices: int) -> Fraction:
    """P_{D,D-3}(M) from the degree-independent quartic, not fitted table entries."""
    s = degree - 2
    q = n_vertices - s
    quartic = (
        9 * (3 * s + 1) * q**4
        - 78 * (3 * s + 1) * q**3
        + (432 * s**2 + 1209 * s + 547) * q**2
        - 2 * (648 * s**2 + 1221 * s + 623) * q
        + 288 * (s + 1) ** 2 * (3 * s + 2)
    )
    return Fraction(math.factorial(3 * s - 1) * s**2 * (3 * s + 1) * quartic, 64 * math.factorial(s - 1))


@pytest.mark.parametrize(("n_vertices", "degree"), [(m, d) for m in (2, 3, 4) for d in range(1, 6)])
def test_scalar_risk_sum_matches_independent_gram_recursion(n_vertices, degree):
    exact, _, _ = _exact_coefficients(n_vertices, degree)
    for size in range(1, min(degree, n_vertices) + 1):
        assert _scalar_risk_sum(n_vertices, degree, size) == exact[size - 1]


@pytest.mark.parametrize("degree", range(3, 9))
def test_third_face_quartic_matches_all_tabulated_polynomial_coefficients(degree):
    # Both sides have degree at most four: five exact evaluations certify polynomial equality.
    for n_vertices in range(degree - 2, degree + 3):
        assert _third_face_polynomial(degree, n_vertices) == _risk_polynomial(degree, degree - 3, n_vertices)


def test_third_face_quartic_outside_the_degree_table():
    # Independent local Gram solve and orbit counting at D=9; no quartic data is used by the reference.
    coefficients = _counted_risk_polynomial(9, 6)
    assert len(coefficients) == 5
    assert coefficients[-1] / math.factorial(7) ** 2 == _large_dimension_limit(9, 6)
    for n_vertices in range(7, 12):
        reference = sum((c * n_vertices**i for i, c in enumerate(coefficients)), Fraction(0))
        assert _third_face_polynomial(9, n_vertices) == reference


@pytest.mark.parametrize("degree", [3, 4, 5, 8, 9, 12, 30])
def test_third_face_quartic_at_the_top_face_boundary(degree):
    size = degree - 2
    risk = _third_face_polynomial(degree, size) / _rising(size, 2 * degree)
    assert risk == comb(degree - 1, size - 1)


def _vertex_risk_sum(n_vertices: int, degree: int) -> Fraction:
    numerator = sum(
        (
            (degree - k + 1)
            * _curve_error(degree, k)
            * (degree - ell + 1)
            * _curve_error(degree, ell)
            * comb(k + ell, k)
            * _vertex_kernel(n_vertices - 1, degree - k, degree - ell)
            for k in range(1, degree + 1)
            for ell in range(1, degree + 1)
        ),
        Fraction(0),
    )
    return n_vertices**2 * math.factorial(degree) ** 2 * numerator / _rising(n_vertices, 2 * degree)


def _vertex_risk_polynomial(degree: int) -> list[Fraction]:
    """Coefficients of the finite binomial expansion, lowest power of M first."""
    polynomial = [Fraction(0)] * (2 * degree - 1)
    for s in range(2 * degree - 1):
        coefficient = math.factorial(degree) ** 2 * sum(
            (
                (degree - k + 1)
                * _curve_error(degree, k)
                * (degree - ell + 1)
                * _curve_error(degree, ell)
                * comb(k + ell, k)
                * _vertex_kernel_power(s, degree - k, degree - ell)
                for k in range(1, degree + 1)
                for ell in range(1, degree + 1)
            ),
            Fraction(0),
        )
        for power, value in enumerate(_falling_polynomial(1, s)):
            polynomial[power] += coefficient * value / math.factorial(s)
    return polynomial


def _triangle_vertex_risk(degree: int) -> Fraction:
    d = degree
    bracket = (
        4 * _harmonic(d)
        - 2 * _harmonic(2 * d + 1)
        + Fraction(1, d)
        + Fraction(1, d + 1)
        + Fraction(1, d + 2)
        - Fraction(1, d * (2 * d + 1) * comb(2 * d, d))
    )
    return Fraction(18, d * (d + 1) * (d + 2) * (d + 3)) * bracket


def _triangle_vertex_single_sum(degree: int) -> Fraction:
    weighted = sum(
        (comb(degree + 1, h + 2) * comb(degree + 1, h) * _harmonic(h + 1) for h in range(degree)),
        Fraction(0),
    )
    return Fraction(36, degree * (degree + 1) * (degree + 2) * (degree + 3) * comb(2 * degree + 2, degree - 1)) * weighted


@pytest.mark.parametrize("degree", range(1, 11))
def test_triangle_vertex_closed_form_matches_independent_gram_recursion(degree):
    exact, _, _ = _exact_coefficients(3, degree)
    assert _triangle_vertex_risk(degree) == exact[0]


@pytest.mark.parametrize("degree", range(1, 13))
def test_triangle_vertex_closed_form_matches_the_original_kernel_sum(degree):
    assert _triangle_vertex_risk(degree) == _vertex_risk_sum(3, degree)


@pytest.mark.parametrize("degree", [1, 2, 3, 8, 10, 20, 81, 100])
def test_triangle_vertex_closed_form_matches_the_single_sum_beyond_exploration(degree):
    assert _triangle_vertex_risk(degree) == _triangle_vertex_single_sum(degree)


@pytest.mark.parametrize("degree", [1, 2, 3, 8, 12, 81, 100])
def test_triangle_vertex_telescoping_certificate_and_boundary_terms(degree):
    d = degree
    denominator = d * (d + 1) * (d + 2) * (d + 3)
    c = Fraction((2 * d + 3) * (d**2 + 2 * d - 1), denominator)
    weights = [comb(d + 1, h + 2) * comb(d + 1, h) for h in range(d + 1)]
    certificate = [weights[h] * Fraction((h + 2) * ((d**2 + 2 * d - 1) * h - 1), denominator * (h + 1)) for h in range(d + 1)]
    for h in range(d):
        assert weights[h] * (Fraction(1, h + 2) - c) == certificate[h + 1] - certificate[h]
    assert certificate[0] == Fraction(-1, (d + 2) * (d + 3))
    assert certificate[d] == 0
    weighted_reciprocals = sum((Fraction(weights[h], h + 2) for h in range(d)), Fraction(0))
    assert weighted_reciprocals == c * comb(2 * d + 2, d - 1) + Fraction(1, (d + 2) * (d + 3))


@pytest.mark.parametrize(("n_vertices", "degree"), [(m, d) for m in (3, 4) for d in range(1, 6)])
def test_vertex_double_sum_matches_exact_coefficients(n_vertices, degree):
    exact, _, _ = _exact_coefficients(n_vertices, degree)
    assert _vertex_risk_sum(n_vertices, degree) == exact[0]


@pytest.mark.parametrize("degree", range(1, 11))
def test_vertex_double_sum_reduces_to_curve_risk(degree):
    assert _vertex_risk_sum(2, degree) == Fraction(4, degree * (degree + 2))


@pytest.mark.parametrize("degree", sorted({d for d, m in _RISK_POLYNOMIALS if m == 0}))
def test_vertex_binomial_expansion_matches_polynomial_and_limit(degree):
    polynomial = _vertex_risk_polynomial(degree)
    assert polynomial == _risk_polynomial_coefficients(degree, 0)[::-1]
    # M^2 / (M)_{2D} leaves the leading coefficient as the large-M limit.
    assert polynomial[-1] == Fraction(2 * degree**2 * (degree + 1) ** 2, 4**degree)


@pytest.mark.parametrize("p", range(11))
def test_vertex_triangle_boundary_kernel_is_harmonic(p):
    harmonic = sum((Fraction(1, j) for j in range(1, p + 2)), Fraction(0))
    assert _vertex_kernel(2, p, 0) == _vertex_kernel(2, 0, p) == 2 * harmonic / (p + 2)


@pytest.mark.parametrize("degree", range(1, 11))
def test_curve_risk_closed_form(degree):
    exact, _, errors = _exact_coefficients(2, degree)
    assert exact == [Fraction(4, degree * (degree + 2)), Fraction(degree - 1)]
    # Bernstein coefficients of the error from a unit error at vertex 0, whose coordinate is x = t_0.
    alpha = _simplex_indices(2, degree)
    vertex = alpha.index((degree, 0))
    for i, (k, _) in enumerate(alpha):
        assert errors[i][vertex] == _curve_error(degree, k)
    # The corollary that inductive fitting wins on curves.
    assert (math.sqrt(degree - 1) + 2 / math.sqrt(degree * (degree + 2))) ** 2 < degree + 1


@pytest.mark.parametrize("degree", range(1, 11))
def test_curve_error_is_a_combination_of_legendre_derivatives(degree):
    def legendre_derivative(n: int) -> list[Fraction]:
        # P_n(2x - 1) has Bernstein coefficients (-1)^(n - k) C(n, k); differentiate, then raise the degree to D.
        c = [(-1) ** (n - k) * comb(n, k) for k in range(n + 1)]
        derivative = [Fraction(n * (c[k + 1] - c[k])) for k in range(n)]
        while len(derivative) < degree + 1:
            r = len(derivative)
            derivative = [
                (Fraction(k, r) * derivative[k - 1] if k else Fraction(0))
                + (Fraction(r - k, r) * derivative[k] if k < r else Fraction(0))
                for k in range(r + 1)
            ]
        return derivative

    low, high = legendre_derivative(degree), legendre_derivative(degree + 1)
    for k in range(degree + 1):
        combination = (low[k] / (degree * (degree + 1)) + high[k] / ((degree + 1) * (degree + 2))) / 2
        assert combination == _curve_error(degree, k)


@pytest.mark.parametrize("degree", range(11, 15))
def test_curve_risk_closed_form_at_high_degrees(degree):
    a = skeleton_risk_coefficients(BezierSpace.simplex(1, degree=degree))
    assert abs(a[0] - 4 / (degree * (degree + 2))) <= (_RISK_RELATIVE_ERROR + U) * a[0]
    assert abs(a[1] - (degree - 1)) <= _RISK_RELATIVE_ERROR * a[1]


@pytest.mark.parametrize(("n_vertices", "degree"), [(3, 3), (4, 3), (3, 4), (4, 4), (5, 4), (4, 5), (5, 5)])
def test_propagation_reduces_to_one_more_vertex(n_vertices, degree):
    _, _, errors = _exact_coefficients(n_vertices, degree)
    alpha = _simplex_indices(n_vertices, degree)
    index = {a: i for i, a in enumerate(alpha)}
    for size in range(1, min(degree, n_vertices)):
        face = tuple(range(size))
        interior = [i for i, a in enumerate(alpha) if all(a[v] for v in face) and not any(a[size:])]
        for i, a in enumerate(alpha):
            if not all(a[v] for v in face) or not any(a[size:]):
                continue
            # The control of the face with the one extra vertex `size` that carries all n outside units.
            n = sum(a[size:])
            one_more = index[a[:size] + (n,) + (0,) * (n_vertices - size - 1)]
            scale = Fraction(n + 1, math.prod(x + 1 for x in a[size:]))
            for k in interior:
                assert errors[i][k] == scale * errors[one_more][k]
        if size == 1:
            # Propagation from a vertex, with the curve coefficients b_k.
            for i, a in enumerate(alpha):
                if a[0]:
                    assert errors[i][interior[0]] == (degree - a[0] + 1) * _curve_error(degree, a[0]) / math.prod(
                        x + 1 for x in a[1:]
                    )


@pytest.mark.parametrize("degree", range(1, 6))
def test_vertex_error_at_the_center_of_a_degree_face(degree):
    _, _, errors = _exact_coefficients(degree, degree)
    alpha = _simplex_indices(degree, degree)
    vertex, center = alpha.index((degree,) + (0,) * (degree - 1)), alpha.index((1,) * degree)
    assert errors[center][vertex] == Fraction((-1) ** (degree - 1) * (degree + 1), 2**degree)


@pytest.mark.parametrize(
    ("degree", "a_0", "a_1"),
    [
        (3, Fraction(41, 280), Fraction(99, 28)),
        (4, Fraction(79, 1120), Fraction(71, 20)),
        (5, Fraction(1189, 30800), Fraction(188, 55)),
        (6, Fraction(2557, 110880), Fraction(81259, 25025)),
    ],
)
def test_triangle_closed_forms_match_documented_values(degree, a_0, a_1):
    exact, _, _ = _exact_coefficients(3, degree)
    assert exact[1] == a_1
    assert exact[0] == a_0
    assert _triangle_vertex_risk(degree) == a_0
    assert _triangle_edge_risk(degree) == a_1


def test_degree_two_wins_in_every_dimension():
    # a_0 = 3M (3M^2 - 15M + 28) / (2 (M+1)(M+2)(M+3)) and a_1 = 30 M (M - 1)^2 / ((M+1)(M+2)(M+3)), as cubics that
    # agree with the table at more than three points.
    for n_vertices in range(2, 12):
        cubic = (n_vertices + 1) * (n_vertices + 2) * (n_vertices + 3)
        a_0 = Fraction(3 * n_vertices * (3 * n_vertices**2 - 15 * n_vertices + 28), 2 * cubic)
        assert _tabulated_coefficient(2, 0, n_vertices) == a_0
        assert _tabulated_coefficient(2, 1, n_vertices) == Fraction(30 * n_vertices * (n_vertices - 1) ** 2, cubic)
    # Clearing denominators: 9 (M+1)(M+2)(M+3) - 3M (3M^2 - 15M + 28) = 99M^2 + 15M + 54 and
    # (M+1)(M+2)(M+3) - M (M - 1)^2 = 8M^2 + 10M + 6, both positive, give a_0 < 9/2 and a_1 < 30.
    # Identities of cubics, checked at more than four points (which need not be vertex counts).
    for n_vertices in range(-3, 4):
        cubic = (n_vertices + 1) * (n_vertices + 2) * (n_vertices + 3)
        assert (
            9 * cubic - 3 * n_vertices * (3 * n_vertices**2 - 15 * n_vertices + 28)
            == 99 * n_vertices**2 + 15 * n_vertices + 54
        )
        assert cubic - n_vertices * (n_vertices - 1) ** 2 == 8 * n_vertices**2 + 10 * n_vertices + 6
    # (sqrt(9/2) + sqrt(30))^2 = 69/2 + 2 sqrt(135) < 58, because 4 * 135 < (58 - 69/2)^2.
    assert (58 - Fraction(69, 2)) ** 2 > 4 * 135
    # M = 11 is the first dimension where the all-at-once risk C(M + 1, 2) exceeds 58; smaller M are checked in
    # test_skeleton_wins_exactly_where_documented.
    assert comb(10 + 1, 2) <= 58 < comb(11 + 1, 2)


# Dimensions (vertex counts) in which optimal inductive skeleton fitting does not beat all-at-once least squares, as
# in the proposition "Where inductive fitting wins".
_SKELETON_LOSSES = {
    1: set(),
    2: set(),
    3: set(range(3, 11)),
    4: set(range(3, 20)),
    5: set(range(4, 29)),
    6: set(range(4, 39)),
    7: set(range(4, 49)),
    8: set(range(5, 60)),
}


def test_skeleton_losses_agree_with_the_documented_table():
    table = {(n_vertices, degree) for degree in _DOCS_OPTIMAL_RISKS for n_vertices in range(2, 7)}
    wins = {(n_vertices, degree) for n_vertices, degree in table if n_vertices not in _SKELETON_LOSSES[degree]}
    assert wins == _DOCS_SKELETON_WINS & table


@pytest.mark.parametrize("degree", range(1, 9))
def test_skeleton_wins_exactly_where_documented(degree):
    # a_m <= sum |p_i| / (m + 1)!^2 for every M, because C(M, m+1)^2 <= M^(2m+2) / (m+1)!^2, (M)_{2D} >= M^(2D) and
    # P_{D,m} has degree at most 2(D - m - 1).
    bounds = [
        sum(abs(c) for c in _risk_polynomial_coefficients(degree, m)) / math.factorial(m + 1) ** 2 for m in range(degree)
    ]
    # The float sums below take D conversions, D square roots, D - 1 additions and a square: relative errors of at
    # most 4 (D + 1) U.
    rounding = 4 * (degree + 1) * U
    bound = sum(math.sqrt(b) for b in bounds) ** 2 * (1 + rounding)
    last = next(n for n in count(2) if comb(n + degree - 1, degree) > bound)
    if degree >= 3:
        # The thresholds quoted in the proof.
        assert last == {3: 55, 4: 111, 5: 190, 6: 295, 7: 429, 8: 593}[degree]
    losses = set()
    for n_vertices in range(2, last + 1):
        optimum = sum(math.sqrt(_tabulated_coefficient(degree, m, n_vertices)) for m in range(degree)) ** 2
        all_at_once = comb(n_vertices + degree - 1, degree)
        assert abs(optimum - all_at_once) > rounding * all_at_once
        if optimum >= all_at_once:
            losses.add(n_vertices)
    assert losses == _SKELETON_LOSSES[degree]


@pytest.mark.parametrize(
    ("dimension", "degree"),
    # Ill-conditioned; binomials beyond float64; and 46376 controls, which must be rejected before allocation.
    [(2, 40), (1, 600), (4, 30)],
)
def test_skeleton_risk_coefficients_reject_ill_conditioned_degrees(dimension, degree):
    space = BezierSpace.simplex(dimension, degree=degree)
    with pytest.raises(ValueError, match="degree is too large"):
        skeleton_risk_coefficients(space)
    with pytest.raises(ValueError, match="Pass ratio explicitly"):
        sample_skeleton(space, 4)


def _two_glued_segments() -> BezierSpace:
    domain = SimploidalSet()
    left, right = domain.add_simplex(1), domain.add_simplex(1)
    domain.identify(left.face(0, 0), right.face(0, 1))
    return BezierSpace.on_set(domain, degree=2)


@pytest.mark.parametrize(
    "space",
    [
        BezierSpace.simplex(0, degree=2),
        BezierSpace.simplex(2, degree=0),
        BezierSpace.simploid((1, 1), degree=(2, 0)),
        _two_glued_segments(),
    ],
)
def test_skeleton_methods_reject_other_spaces(space):
    with pytest.raises(ValueError, match="single simplex or simploid"):
        skeleton_risk_coefficients(space)
    with pytest.raises(ValueError, match="single simplex or simploid"):
        sample_skeleton(space, 4)


def test_sample_skeleton_default_ratio_needs_a_well_conditioned_degree():
    space = BezierSpace.simplex(2, degree=13)
    with pytest.raises(ValueError, match="Pass ratio explicitly"):
        sample_skeleton(space, 4)
    assert len(sample_skeleton(space, 4, ratio=[1, 1, 1], seed=0).parameters) == 4


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


def _assert_largest_remainder(points, fractions, n_vertices):
    """Rows per skeleton are the floor or ceiling of n times its fraction, split among faces within one row."""
    counts = _face_counts(points, n_vertices)
    n = len(points.parameters)
    assert sum(counts.values()) == n
    # The fractions carry a few M roundings of relative error (see test_skeleton_risk_coefficients_structure).
    slack = n * 4 * n_vertices * U
    for size in range(1, n_vertices + 1):
        level = [counts[face] for face in combinations(range(n_vertices), size)]
        quota = n * fractions[size - 1]
        assert math.floor(quota - slack) <= sum(level) <= math.ceil(quota + slack)
        assert max(level) - min(level) <= 1
        if fractions[size - 1] == 0:
            assert sum(level) == 0


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
    _assert_largest_remainder(points, optimal_skeleton_ratio(space), n_vertices)
    assert torch.equal(t, sample_skeleton(space, n, seed=seed, method=method).parameters)


@given(
    n_vertices=st.integers(2, 6),
    data=st.data(),
    n=st.integers(0, 500),
    seed=seeds(),
)
def test_sample_skeleton_allocates_explicit_ratios(n_vertices, data, n, seed):
    ratio = data.draw(
        st.lists(
            st.one_of(st.just(0.0), st.floats(0, 1e308), st.floats(0, 1)), min_size=n_vertices, max_size=n_vertices
        ).filter(lambda r: max(r) > 0)
    )
    points = sample_skeleton(BezierSpace.simplex(n_vertices - 1, degree=2), n, ratio=ratio, seed=seed)
    largest = max(ratio)
    fractions = [r / largest for r in ratio]
    _assert_largest_remainder(points, [f / sum(fractions) for f in fractions], n_vertices)


def test_sample_skeleton_chooses_extra_row_faces_at_random():
    # Four vertices and one or three rows on the 0-skeleton: one row per chosen vertex, picked by rejection
    # sampling (1 row) or by a random permutation (3 rows). Over 400 seeds each vertex is chosen with
    # probability rows / 4; five binomial standard errors bound the seeded frequencies.
    space = BezierSpace.simplex(3, degree=1)
    seeds_ = 400
    for rows in (1, 3):
        hits = torch.zeros(4)
        for seed in range(seeds_):
            hits += (sample_skeleton(space, rows, ratio=[1, 0, 0, 0], seed=seed).parameters > 0).sum(0)
        p = rows / 4
        assert ((hits / seeds_ - p).abs() <= 5 * math.sqrt(p * (1 - p) / seeds_)).all()


def test_sample_skeleton_spreads_few_rows_over_wide_skeletons():
    # 24 vertices have C(24, 12) = 2704156 faces of size 12; each of the few rows there lands on a distinct face.
    space = BezierSpace.simplex(23, degree=1)
    real_combinations = combinations

    def small_combinations(vertices, size):
        # Listing every face of a middle skeleton would take seconds; the few rows there must not need it.
        assert size not in range(4, 21)
        return real_combinations(vertices, size)

    with patch("torch_bsf.sampling.combinations", small_combinations):
        points = sample_skeleton(space, 48, ratio=[1.0] * 24, seed=0)
    sizes = (points.parameters > 0).sum(1)
    assert sizes.bincount(minlength=25)[1:].tolist() == [2] * 24
    middle = points.parameters[sizes == 12] > 0
    assert len({tuple(row.tolist()) for row in middle}) == 2
    again = sample_skeleton(space, 48, ratio=[1.0] * 24, seed=0)
    assert torch.equal(points.parameters, again.parameters)


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


def test_sample_skeleton_faces_of_larger_simplices_are_uniform():
    # Uniform on a triangle face of a tetrahedron: E t_i = 1/3 and E t_i^2 = 1/6 (variances 2/9 - 1/9 and
    # 2/15 - 1/36 per row), zero off the face.
    space = BezierSpace.simplex(3, degree=3)
    n = 20000
    t = sample_skeleton(space, n, ratio=[0, 0, 1, 0], seed=0).parameters
    face = t[(t[:, 3] == 0) & (t[:, :3] > 0).all(1)]
    assert len(face) * 4 >= n - 3 and ((t > 0).sum(1) == 3).all()
    assert (face[:, :3].mean(0) - 1 / 3).abs().max() <= 5 * math.sqrt(1 / 18 / len(face))
    assert ((face[:, :3] ** 2).mean(0) - 1 / 6).abs().max() <= 5 * math.sqrt((2 / 15 - 1 / 36) / len(face))


def test_sample_skeleton_accepts_ratio_entries_near_the_float64_limit():
    space = BezierSpace.simplex(2, degree=2)
    for ratio in ([1e308, 1e308, 1e308], torch.full((3,), 1e308, dtype=torch.float64), [2**1000] * 3):
        counts = _face_counts(sample_skeleton(space, 7, ratio=ratio, seed=0), 3)
        assert [sum(counts[f] for f in combinations(range(3), k)) for k in (1, 2, 3)] in ([3, 2, 2], [2, 3, 2], [2, 2, 3])


@pytest.mark.parametrize(
    "ratio",
    [
        [1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [[1.0, 1.0]],
        ["a", "b"],
        [True, False],
        [1j, 1.0],
        [True, 1.0],
        [torch.tensor(True), 1.0],
        [torch.tensor(1.0, device="meta"), 1.0],
        [np.array(1.0), 1.0],
        np.array([True, False]),
        np.array([1 + 1j, 1.0]),
        np.array([1.0, 1.0], dtype=np.complex64),
        torch.tensor([True, False]),
        torch.tensor([1j, 1.0]),
        torch.ones(2, device="meta"),
        object(),
    ],
)
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


# ---------------------------------------------------------------------------
# skeletons of simploids
# ---------------------------------------------------------------------------

_SIMPLOIDS = [((1, 1), (1, 1)), ((1, 1), (2, 2)), ((1, 1), (3, 1)), ((2, 1), (2, 2)), ((2, 1), (2, 3)), ((1, 1, 1), (2, 1, 2))]


def _skeleton_faces(dimensions, m):
    """Every face of dimension ``m``, listed by brute force over vertex subsets with one vertex in each factor."""
    blocks = _blocks(dimensions)
    width = sum(a + 1 for a in dimensions)
    return [
        face
        for size in range(len(dimensions), width + 1)
        for face in combinations(range(width), size)
        if all(any(i in block for i in face) for block in blocks) and size - len(dimensions) == m
    ]


def _sampled_faces(dimensions, degree, m):
    """The faces of dimension ``m`` with interior controls, or all of them when none has any."""
    faces = _skeleton_faces(dimensions, m)
    blocks = _blocks(dimensions)
    active = [f for f in faces if all(sum(i in b for i in f) <= d for b, d in zip(blocks, degree, strict=True))]
    return active or faces


@pytest.mark.parametrize(("dimensions", "degree"), _SIMPLOIDS)
def test_simploid_faces_are_products_of_factor_faces(dimensions, degree):
    for m in range(sum(dimensions) + 1):
        assert _n_skeleton_faces(dimensions, degree, m) == len(_sampled_faces(dimensions, degree, m))
    expected = [
        face
        for m in range(sum(dimensions) + 1)
        for face in _sampled_faces(dimensions, degree, m)
        if all(sum(i in block for i in face) <= d for block, d in zip(_blocks(dimensions), degree, strict=True))
    ]
    assert _faces_with_interior_controls(dimensions, degree) == expected


def test_square_risk_coefficients_have_closed_forms():
    # Degree 1: the four vertex controls are vertex means, each with N / 4 rows, and the squared bilinear basis
    # functions integrate to 1 / 9 each, so a_0 = 4 * 4 / 9. The Gram entry 1 / 9 is rounded once, and the trace,
    # the factor 4 and the sum over four vertices add six roundings: 8 U bounds the relative error.
    a = skeleton_risk_coefficients(BezierSpace.simploid((1, 1), degree=1))
    assert a == pytest.approx((16 / 9, 0, 0), rel=8 * U, abs=0)


@pytest.mark.parametrize(("dimensions", "degree"), _SIMPLOIDS)
def test_simploid_risk_coefficients_match_brute_force_propagation(dimensions, degree):
    # An independent evaluation of the same risk in exact rational arithmetic: Gram matrices from the moments of
    # the uniform distribution and the multinomial coefficients, Gauss-Jordan solves, and the error propagated
    # face by face. Float64 agrees to the 1e-6 relative error that skeleton_risk_coefficients guarantees.
    a = skeleton_risk_coefficients(BezierSpace.simploid(dimensions, degree=degree))
    assert len(a) == sum(dimensions) + 1
    assert a == pytest.approx(_rational_skeleton_risk(dimensions, degree), rel=1e-6, abs=0)


def _rational_solve(matrix, right):
    """Gauss-Jordan elimination of ``matrix x = right`` over the rationals."""
    n = len(matrix)
    augmented = [list(matrix[i]) + list(right[i]) for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if augmented[r][col] != 0)
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        for r in range(n):
            if r != col and augmented[r][col] != 0:
                factor = augmented[r][col] / augmented[col][col]
                augmented[r] = [x - factor * y for x, y in zip(augmented[r], augmented[col], strict=True)]
    return [[x / augmented[i][i] for x in augmented[i][n:]] for i in range(n)]


def _rational_gram(dimensions, degree, alphas, face, rows, columns):
    """Bernstein Gram entries under the uniform distribution on ``face``, the product of its factors' faces."""
    blocks = _blocks(dimensions)

    def moment(exponents):
        # E t^alpha on the uniform k-simplex: k! prod(alpha!) / (|alpha| + k)!.
        k = len(exponents) - 1
        return Fraction(
            math.factorial(k) * math.prod(math.factorial(e) for e in exponents), math.factorial(sum(exponents) + k)
        )

    def entry(r, c):
        value = Fraction(1)
        for block, d in zip(blocks, degree, strict=True):
            if any(alphas[r][i] + alphas[c][i] for i in block if i not in face):
                return Fraction(0)
            multinomials = [
                math.factorial(d) // math.prod(math.factorial(a[i]) for i in block) for a in (alphas[r], alphas[c])
            ]
            value *= multinomials[0] * multinomials[1] * moment([alphas[r][i] + alphas[c][i] for i in block if i in face])
        return value

    return [[entry(r, c) for c in columns] for r in rows]


def _rational_skeleton_risk(dimensions, degree):
    alphas = [sum(index, ()) for index in _indices(tuple(dimensions), tuple(degree))]
    n = len(alphas)
    supports = [tuple(i for i, v in enumerate(alpha) if v) for alpha in alphas]
    # errors[k][j]: the error of control k per unit noise in the estimate of control j on its own face.
    errors = [[Fraction(0)] * n for _ in range(n)]
    faces = []
    for face in _faces_with_interior_controls(dimensions, degree):
        interior = [k for k in range(n) if supports[k] == face]
        boundary = [k for k in range(n) if supports[k] != face and set(supports[k]) <= set(face)]
        g_tt = _rational_gram(dimensions, degree, alphas, face, interior, interior)
        g_tb = _rational_gram(dimensions, degree, alphas, face, interior, boundary)
        propagated = [[sum(row[j] * errors[b][c] for j, b in enumerate(boundary)) for c in range(n)] for row in g_tb]
        solved = _rational_solve(g_tt, propagated)
        for i, k in enumerate(interior):
            errors[k] = [-x for x in solved[i]]
            errors[k][k] = Fraction(1)
        faces.append((face, interior, g_tt))
    full = _rational_gram(dimensions, degree, alphas, tuple(range(len(alphas[0]))), range(n), range(n))
    coefficients = [Fraction(0)] * (sum(dimensions) + 1)
    for face, interior, g_tt in faces:
        m = len(face) - len(dimensions)
        columns = [[errors[r][k] for r in range(n)] for k in interior]
        inner = [[sum(x * full[r][c] * y[c] for r, x in enumerate(u) for c in range(n)) for y in columns] for u in columns]
        trace = sum(row[i] for i, row in enumerate(_rational_solve(g_tt, inner)))
        coefficients[m] += len(_sampled_faces(dimensions, degree, m)) * trace
    return [float(c) for c in coefficients]


def _assert_simploid_rows(space, degree, points, n, fractions):
    """Every row lies on one sampled face, and rows per skeleton follow ``fractions`` as in a simplex."""
    dimensions = space.cells[0].dimensions
    t = points.parameters
    assert t.shape == (n, space.width) and t.dtype == torch.float64 and (points.cell_ids == 0).all() and (t >= 0).all()
    for block in _blocks(dimensions):
        # Normalizing at most len(block) exponentials: one division and a sum per row.
        assert ((t[:, block.start : block.stop].sum(1) - 1).abs() <= (len(block) + 1) * U * 2).all()
    support = t > 0
    dimension_of_row = support.sum(1) - len(dimensions)
    assert (dimension_of_row[1:] >= dimension_of_row[:-1]).all()
    # The fractions carry a few roundings per level of relative error, as in _assert_largest_remainder.
    slack = n * 4 * space.width * U
    on_faces = 0
    for m in range(sum(dimensions) + 1):
        faces = _sampled_faces(dimensions, degree, m)
        level = [int((support == torch.tensor([i in face for i in range(space.width)])).all(1).sum()) for face in faces]
        on_faces += sum(level)
        assert math.floor(n * fractions[m] - slack) <= sum(level) <= math.ceil(n * fractions[m] + slack)
        assert max(level) - min(level) <= 1
        if fractions[m] == 0:
            assert sum(level) == 0
    assert on_faces == n


@pytest.mark.parametrize(("dimensions", "degree"), _SIMPLOIDS)
@pytest.mark.parametrize("method", ["random", "sobol"])
@pytest.mark.parametrize("n", [0, 1, 157])
def test_sample_skeleton_of_simploids_places_each_row_on_one_face(dimensions, degree, method, n):
    space = BezierSpace.simploid(dimensions, degree=degree)
    points = sample_skeleton(space, n, seed=3, method=method)
    _assert_simploid_rows(space, degree, points, n, optimal_skeleton_ratio(space))
    assert torch.equal(points.parameters, sample_skeleton(space, n, seed=3, method=method).parameters)


@given(
    dimensions=simploid_types(min_factors=2),
    data=st.data(),
    n=st.integers(0, 300),
    seed=seeds(),
    method=st.sampled_from(["random", "sobol"]),
)
def test_sample_skeleton_of_simploids_allocates_ratios(dimensions, data, n, seed, method):
    degree = data.draw(degrees(dimensions, min_degree=1))
    space = BezierSpace.simploid(dimensions, degree=degree)
    levels = sum(dimensions) + 1
    ratio = data.draw(
        st.none()
        | st.lists(st.one_of(st.just(0.0), st.floats(0, 1e308), st.floats(0, 1)), min_size=levels, max_size=levels).filter(
            lambda r: max(r) > 0
        )
    )
    points = sample_skeleton(space, n, ratio=ratio, seed=seed, method=method)
    if ratio is None:
        fractions = list(optimal_skeleton_ratio(space))
    else:
        largest = max(ratio)
        scaled = [r / largest for r in ratio]
        fractions = [f / sum(scaled) for f in scaled]
    _assert_simploid_rows(space, degree, points, n, fractions)
    assert torch.equal(points.parameters, sample_skeleton(space, n, ratio=ratio, seed=seed, method=method).parameters)


def _simplex_skeleton_reference(width: int, counts: list[int], seed: int, method: str) -> torch.Tensor:
    """The simplex-only ``sample_skeleton`` before simploids, with the same torch draws in the same order."""
    generator = torch.Generator().manual_seed(seed)

    def exponentials(n: int, size: int) -> torch.Tensor:
        if method == "random":
            uniforms = torch.rand(n, size, generator=generator, dtype=torch.float64)
        else:
            engine = torch.quasirandom.SobolEngine(
                size, scramble=True, seed=int(torch.randint(2**31, (1,), generator=generator))
            )
            uniforms = torch.empty(n, size, dtype=torch.float64)
            if n:
                uniforms[0] = engine.quasi.to(torch.float64) / 2**engine.MAXBIT
                if n > 1:
                    engine.fast_forward(1)
                    uniforms[1:] = engine.draw(n - 1, dtype=torch.float64)
        return -uniforms.clamp_min(torch.finfo(torch.float64).tiny).log()

    blocks = [torch.zeros(0, width, dtype=torch.float64)]
    for size, total in enumerate(counts, start=1):
        n_faces = comb(width, size)
        if 2 * total < n_faces:
            chosen: set[tuple[int, ...]] = set()
            while len(chosen) < total:
                chosen.add(tuple(sorted(torch.randperm(width, generator=generator)[:size].tolist())))
            faces = [(face, 1) for face in sorted(chosen)]
        else:
            per_face = torch.full((n_faces,), total // n_faces, dtype=torch.long)
            per_face[torch.randperm(n_faces, generator=generator)[: total % n_faces]] += 1
            faces = [(f, r) for f, r in zip(combinations(range(width), size), per_face.tolist(), strict=True) if r]
        for face, rows in faces:
            draws = exponentials(rows, size)
            block = torch.zeros(rows, width, dtype=torch.float64)
            block[:, face] = draws / draws.sum(1, keepdim=True)
            blocks.append(block)
    return torch.cat(blocks)


@pytest.mark.parametrize("dimension, degree", [(1, 3), (2, 2), (3, 2), (6, 2), (23, 1)])
@pytest.mark.parametrize("n", [0, 5, 50])
@pytest.mark.parametrize("method", ["random", "sobol"])
def test_sample_skeleton_of_a_simplex_keeps_its_draws(dimension, degree, n, method):
    # Generalizing to simploids must not change any simplex draw. Literals would pin one torch version's generator,
    # so the reference reruns the simplex-only algorithm with the installed torch. The wide simplex samples its
    # edges by rejection; the others mostly split equally.
    space = BezierSpace.simplex(dimension, degree=degree)
    ratio = [1.0] * (dimension + 1)
    counts = _skeleton_counts(n, _skeleton_fractions(ratio, space, dimension + 1))
    expected = _simplex_skeleton_reference(dimension + 1, counts, 7, method)
    assert torch.equal(sample_skeleton(space, n, ratio=ratio, seed=7, method=method).parameters, expected)


def test_skeleton_risk_coefficients_of_a_triangle_are_unchanged():
    space = BezierSpace.simplex(2, degree=2)
    assert skeleton_risk_coefficients(space) == (0.375, 3.0, 0.0)


def test_sample_skeleton_of_a_prism_chooses_edges_uniformly():
    # The nine edges of a prism: six triangle edges at an end of the interval and three intervals at a vertex.
    # With 2 rows, rejection sampling picks two distinct edges, each with probability 2 / 9; with 6 rows, a
    # random permutation does. Five binomial standard errors bound the seeded frequencies.
    space = BezierSpace.simploid((2, 1), degree=2)
    edges = _skeleton_faces((2, 1), 1)
    seeds_ = 600
    for rows in (2, 6):
        hits: Counter[tuple[int, ...]] = Counter()
        for seed in range(seeds_):
            support = sample_skeleton(space, rows, ratio=[0, 1, 0, 0], seed=seed).parameters > 0
            faces = {tuple(row.nonzero().flatten().tolist()) for row in support}
            assert len(faces) == rows and faces <= set(edges)
            hits.update(faces)
        p = rows / 9
        assert all(abs(hits[edge] / seeds_ - p) <= 5 * math.sqrt(p * (1 - p) / seeds_) for edge in edges)


def test_sample_skeleton_of_a_square_is_uniform_on_its_interior():
    # On the square, each first coordinate of a factor is uniform and the factors are independent.
    space = BezierSpace.simploid((1, 1), degree=2)
    n = 20000
    t = sample_skeleton(space, n, ratio=[0, 0, 1], seed=0).parameters
    u, v = t[:, 0], t[:, 2]
    assert abs(u.mean().item() - 0.5) <= 5 * math.sqrt(1 / 12 / n)
    assert abs(v.var().item() - 1 / 12) <= 5 * math.sqrt(1 / 180 / n)
    assert abs(((u - 0.5) * (v - 0.5)).mean().item()) <= 5 * math.sqrt(1 / 144 / n)


def test_sample_skeleton_of_a_simploid_checks_the_ratio_length():
    space = BezierSpace.simploid((2, 1), degree=2)
    with pytest.raises(ValueError, match="ratio must have 4 "):
        sample_skeleton(space, 4, ratio=[1, 1, 1])
    assert len(sample_skeleton(space, 4, ratio=[1, 1, 1, 1], seed=0).parameters) == 4


def test_sample_skeleton_skips_faces_without_interior_controls():
    # On a prism of degree (2, 1), the intervals at the triangle vertices have no interior controls, so every edge
    # row goes to the six triangle edges; with no square face having interior controls, a ratio on the
    # 2-skeleton spreads over all five two-dimensional faces.
    space = BezierSpace.simploid((2, 1), degree=(2, 1))
    support = sample_skeleton(space, 60, ratio=[0, 1, 0, 0], seed=0).parameters > 0
    assert set(support[:, :3].sum(1).tolist()) == {2} and set(support[:, 3:].sum(1).tolist()) == {1}
    support = sample_skeleton(space, 50, ratio=[0, 0, 1, 0], seed=0).parameters > 0
    assert len({tuple(row.tolist()) for row in support}) == 5
    points = sample_skeleton(space, 300, seed=0)
    assert "Ignored" not in fit_skeleton_regression(space, Samples(points, points.parameters[:, :1])).message
