"""Check the algebraic lemmas in the trace-correct boundary-lift bound.

The accompanying proof establishes the universal mode traces, orthogonality
and Hilbert-space inequality.  This optional SymPy checker certifies their
q=3 boundary cancellation and the harmonic/rational inequalities used to
obtain the explicit five-vertex bound.  Finite coefficient checks are not
used as a proof of an all-degree statement.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import codimension_three as bounds
from .exact import atomic_json


def _trace_coefficients(outside: int, facet: int) -> dict[int, int]:
    """Restrict formal inclusion-exclusion faces to one outside facet."""
    result: dict[int, int] = {}
    for subset in range((1 << outside) - 1):
        restricted = subset & ~(1 << facet)
        result[restricted] = result.get(restricted, 0) + (-1) ** (outside - subset.bit_count() + 1)
    return {subset: coefficient for subset, coefficient in result.items() if coefficient}


def verify_lemmas() -> dict[str, Any]:
    """Prove the finite algebraic identities, domain signs and bound constant."""
    import sympy as sp

    d, z, j, h, u = sp.symbols("D z j H u")
    source = Path(__file__).read_bytes()
    bound_source = Path(bounds.__file__).read_bytes()
    expected = {((1 << 3) - 1) & ~(1 << facet): 1 for facet in range(3)}
    trace = [_trace_coefficients(3, facet) for facet in range(3)]
    if any(row != {subset: coefficient} for row, (subset, coefficient) in zip(trace, expected.items(), strict=True)):
        raise ArithmeticError("outside-facet inclusion-exclusion cancellation failed")
    # S_j=2-(H_(j+1)+1)/(j+1), with H_(j+1)=H_j+1/(j+1).
    current = 2 - (h + 1 / (j + 1) + 1) / (j + 1)
    previous = 2 - (h + 1) / j
    residual = sp.cancel(current - previous - (h + 1 / (j + 1)) / (j * (j + 1)))
    if residual != 0 or sp.cancel(current.subs({j: 1, h: 1}) - sp.Rational(3, 4)) != 0:
        raise ArithmeticError("finite harmonic telescoping identity failed")
    polynomials = {
        "source_radius": 3 * (2 * d + 2) * (2 * d + 3) * (2 * d + 4) - 24 * (d - 1) * d**2,
        "facet_radius": 27 * (2 * d + 3) * (2 * d + 4) - 108 * d**2,
        "edge_radius": 108 * (2 * d + 4) - 864,
        "two_coordinate_denominator": (d + 4 + u) * (d + 5 + u) - d**2,
        "positive_mode_correction": (j * (2 * d + 3 - j) - 1).subs(d, j + 1 + u).subs(j, z + 1),
    }
    coefficients = {}
    for name, polynomial in polynomials.items():
        shifted = sp.Poly(sp.expand(polynomial.subs(d, z + 2)), z, u)
        if any(value < 0 for value in shifted.coeffs()) or not shifted:
            raise ArithmeticError(f"domain sign certificate failed: {name}")
        coefficients[name] = [str(value) for value in shifted.coeffs()]
    # sqrt(r0*(D-1))<=sqrt(3)/D; sqrt(r1*a_triangle)<3sqrt(3)/D;
    # (1/2)sqrt(r2*a_tetrahedron)<6sqrt(3)/D, all for D>=2.
    derived_constant = 10**2 * 3 * (1 + 3 + 6) ** 2
    if derived_constant != bounds.EDGE_BOUND_NUMERATOR:
        raise ArithmeticError("explicit power-bound constant disagrees with the proof")
    if Path(__file__).read_bytes() != source or Path(bounds.__file__).read_bytes() != bound_source:
        raise RuntimeError("certificate source changed during verification")
    return {
        "schema": "skeleton-risk-boundary-lift-lemmas-v1",
        "status": "certified_algebraic_lemmas",
        "outside_coordinates": 3,
        "outside_facet_coefficients": [{str(k): value for k, value in row.items()} for row in trace],
        "harmonic_telescope_residual": "0",
        "harmonic_first_value": "3/4",
        "shifted_domain_polynomial_coefficients": coefficients,
        "domain": "integer D>=2; auxiliary z,u>=0",
        "edge_power_bound_numerator": derived_constant,
        "edge_power_bound_denominator": "D^2",
        "bound_source_sha256": hashlib.sha256(bound_source).hexdigest(),
        "checker_source_sha256": hashlib.sha256(source).hexdigest(),
        "proof": "docs/advanced/skeleton_risk_codimension_three.rst",
        "scope": "algebraic lemmas; universal trace/orthogonality/norm argument is in the proof chapter",
    }


def main(argv: list[str] | None = None) -> int:
    """Construct or recheck a source-bound algebraic-lemma certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = verify_lemmas()
        if arguments.verify is not None:
            supplied = json.loads(arguments.verify.read_bytes())
            if supplied != report:
                raise ValueError("supplied lemma certificate differs from verified identities or source")
        atomic_json(arguments.output, report)
    except (ValueError, ArithmeticError, RuntimeError, ModuleNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
