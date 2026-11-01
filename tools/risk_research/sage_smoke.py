"""Exercise exact arithmetic, Ore recurrences, telescoping and certified balls."""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path


def run_checks() -> dict[str, object]:
    """Run mathematical checks inside the optional SageMath environment."""
    if not __debug__:
        raise RuntimeError("Run without -O or PYTHONOPTIMIZE; mathematical checks require assertions.")

    from sage.all import QQ, ZZ, PolynomialRing, RealBallField, binomial
    from sage.env import SAGE_VERSION

    # Sage's categories must initialize before Ore algebra imports Sage internals.
    ore_algebra = import_module("ore_algebra")
    assert QQ(1) / 3 + QQ(2) / 3 == 1
    polynomials = PolynomialRing(QQ, "n")
    n = polynomials.gen()
    algebra = ore_algebra.OreAlgebra(polynomials, "Sn")
    values = [binomial(2 * index, index) for index in range(40)]
    operator = ore_algebra.guess(values[:25], algebra, order=1, degree=1)
    assert operator.order() == 1
    coefficients = operator.coefficients(sparse=False)
    for index in range(39):
        residual = sum(coefficients[j](n=index) * values[index + j] for j in range(len(coefficients)))
        assert residual == 0

    # The binomial theorem example from ore_algebra/ideal.py. This checks
    # creative telescoping itself, separately from recurrence guessing.
    ring = PolynomialRing(ZZ, ("n", "k"))
    n, k = ring.gens()
    bivariate = ore_algebra.OreAlgebra(ring, "Sn", "Sk")
    sn, sk = bivariate.gens()
    ideal = bivariate.ideal([(k + 1) * sk + k - n, (1 - k + n) * sn - 1 - n])
    telescopers, certificates = ideal.ct(sk - 1)
    assert len(telescopers) == len(certificates) == 1
    telescoper = telescopers[0]
    assert telescoper.monic() == telescoper.parent().gen() - 2
    assert certificates[0] != 0
    for degree in range(20):
        assert sum(binomial(degree, index) for index in range(degree + 1)) == 2**degree

    balls = RealBallField(128)
    third = balls(1) / 3
    assert QQ(1) / 3 in third
    assert QQ(1) / 2 not in third
    native_extensions = ["binary_splitting_arb", "dac_sum_c", "eval_poly_at_int", "naive_sum_c"]
    for name in native_extensions:
        import_module("ore_algebra.analytic." + name)
    differential = ore_algebra.OreAlgebra(PolynomialRing(QQ, "x"), "Dx")
    exponential = (differential.gen() - 1).numerical_solution([1], [0, 1], QQ(1) / 10**30)
    reference = balls(1).exp()
    assert exponential.overlaps(exponential.parent()(reference))
    return {
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "sage": SAGE_VERSION,
        "ore_algebra": version("ore_algebra"),
        "python_flint": version("python-flint"),
        "cython": version("Cython"),
        "central_binomial_recurrence": str(operator),
        "recurrence_training_indices": [0, 24],
        "recurrence_heldout_indices": [25, 39],
        "creative_telescoper": str(telescoper),
        "creative_certificate": str(certificates[0]),
        "ball_precision_bits": 128,
        "native_extensions": native_extensions,
        "analytic_ode_value": str(exponential),
        "analytic_reference_overlap": True,
        "note": "Toolchain smoke checks; these do not prove a new skeleton-risk formula.",
    }


def main() -> None:
    """Run checks and optionally write a machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run_checks(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
