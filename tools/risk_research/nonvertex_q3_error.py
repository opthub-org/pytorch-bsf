"""Certify finite-degree errors for every nonvertex codimension-three face.

The all-tail error is O(log(D)/D), with explicit constants. Evaluation
uses exact rational arithmetic and does not generate a risk coefficient.
See ``docs/advanced/skeleton_risk_nonvertex_q3_error.rst`` for the proof.
"""

import argparse
import hashlib
import json
import os
import tempfile
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

SCHEMA = "skeleton-risk-nonvertex-q3-error-v1"
HASH_POLICY = "sha256-crlf-to-lf-v1"
PROOF = "docs/advanced/skeleton_risk_nonvertex_q3_error.rst"


def _decimal(value: int) -> str:
    """Encode an integer in bounded chunks without altering process limits."""
    if value == 0:
        return "0"
    sign, value = ("-", -value) if value < 0 else ("", value)
    parts = []
    while value:
        value, remainder = divmod(value, 10**500)
        parts.append(f"{remainder:0500d}" if value else str(remainder))
    return sign + "".join(reversed(parts))


def _decimal_integer(text: str) -> int:
    """Read decimal chunks, retaining ordinary signed CLI integer syntax."""
    text = text.strip()
    sign = -1 if text.startswith("-") else 1
    if text.startswith(("-", "+")):
        text = text[1:]
    if not all(part and all("0" <= digit <= "9" for digit in part) for part in text.split("_")):
        raise ValueError("invalid decimal integer")
    digits, result = text.replace("_", ""), 0
    for start in range(0, len(digits), 500):
        part = digits[start : start + 500]
        result = result * 10 ** len(part) + int(part)
    return sign * result


def fraction_record(value: Fraction) -> dict[str, str]:
    """Encode rational integers locally, including endpoints beyond 4300 digits."""
    return {"numerator": _decimal(value.numerator), "denominator": _decimal(value.denominator)}


def atomic_json(path: Path, value: object) -> None:
    """Atomically write deterministic JSON, keeping large parameters numeric."""
    text = _canonical(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _integer(value: object, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _size(face_size: object) -> int:
    return _integer(face_size, 2, "face_size")


def leading_constant_enclosure(face_size: int, terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose the known positive A_(s,3) by rational integral tails.

    The positive summand (n-1)/n**3 decreases for n>=2. Thus its
    tail after n=terms lies between its integrals from terms+1 and terms.
    """
    _size(face_size)
    _integer(terms, 2, "terms")
    partial = sum((Fraction(n - 1, n**3) for n in range(2, terms + 1)), Fraction(0))

    def integral(start: int) -> Fraction:
        return Fraction(1, start) - Fraction(1, 2 * start**2)

    scale = Fraction(
        3 * comb(face_size + 3, face_size) ** 2 * prod(range(face_size, face_size + 3)), 2 * factorial(face_size - 2)
    )
    return scale * (partial + integral(terms + 1)), scale * (partial + integral(terms))


def dyadic_tail_bound(face_size: int, exponent: int) -> Fraction:
    """Bound the relative error at EVERY D>=2**exponent, exponent>=2."""
    _size(face_size)
    _integer(exponent, 2, "exponent")
    denominator = 1 << exponent
    if denominator < 2 * (face_size - 1):
        raise ValueError("dyadic degree must be >= 2*(face_size-1)")
    logarithmic = Fraction(25 * face_size + 358, 4)
    constant = Fraction(25 * face_size**2 - 69 * face_size + 106, 4)
    return (logarithmic * (exponent + 1) + constant) / denominator


def relative_error_threshold(face_size: int, target: Fraction) -> int:
    """Return a sufficient D0 for relative error <=target at every D>=D0.

    target must be a strictly positive Fraction. This specializes an
    analytic all-tail bound; it does not scan individual risk values.
    """
    _size(face_size)
    if type(target) is not Fraction or target <= 0:
        raise ValueError("target must be a positive Fraction")
    exponent = max(2, (2 * face_size - 3).bit_length())
    while dyadic_tail_bound(face_size, exponent) > target:
        exponent += 1
    return 1 << exponent


def relative_error_bound(face_size: int, degree: int) -> Fraction:
    """Bound abs(a/(A*D**(s-5))-1) for every s>=2 and positive D.

    On the inactive prefix D<s the coefficient is zero and the relative
    error is exactly one. The active bound uses only O(log D)-bit index
    operations, not a harmonic table or any exact risk evaluation.
    """
    _size(face_size)
    _integer(degree, 1, "degree")
    if degree < face_size:
        return Fraction(1)
    count = degree - face_size + 1
    # H_count <= 1+log(count) <= 1+ceil(log2(count)); exact integer ceiling.
    harmonic_upper = 1 + (count - 1).bit_length()
    averaging = Fraction(408, degree + 2 * face_size + 2) * (harmonic_upper + Fraction(2, count + 1) - 2)
    tail = Fraction(25, count)
    angular = Fraction(25 * ((face_size - 2) * harmonic_upper + (face_size - 1) * (face_size - 2)), degree)
    prefactor = Fraction((2 * degree) ** 3, prod(range(2 * degree + face_size, 2 * degree + face_size + 3)))
    sharp = 1 - prefactor + (averaging + tail + angular) / 4
    exponent = degree.bit_length() - 1
    if exponent < 2 or 2**exponent < 2 * (face_size - 1):
        return sharp
    return min(sharp, dyadic_tail_bound(face_size, exponent))


def coefficient_enclosure(face_size: int, degree: int, terms: int = 64) -> tuple[Fraction, Fraction]:
    """Enclose the finite-D coefficient without evaluating the double sum."""
    lower, upper = leading_constant_enclosure(face_size, terms)
    error = relative_error_bound(face_size, degree)
    if degree < face_size:
        return Fraction(0), Fraction(0)
    scale = Fraction(degree) ** (face_size - 5)
    return lower * max(Fraction(0), 1 - error) * scale, upper * (1 + error) * scale


def _sources() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "tools/risk_research/nonvertex_q3_error.py",
        "tools/risk_research/exact.py",
        PROOF,
        "docs/advanced/skeleton_risk_codimension_three_exact.rst",
    ]
    return {name: (root / name).read_bytes() for name in names}


def certificate(face_size: int, degree: int, terms: int = 64, target: Fraction | None = None) -> dict[str, object]:
    """Specialize the proved finite-D error with source-bound exact intervals."""
    leading = leading_constant_enclosure(face_size, terms)
    interval = coefficient_enclosure(face_size, degree, terms)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "proof_status": "proved_finite_degree_error_and_all_tail_threshold",
        "proof": PROOF,
        "face_size": face_size,
        "vertices": face_size + 3,
        "outside": 3,
        "degree": degree,
        "active": degree >= face_size,
        "constant_terms": terms,
        "leading_constant": {"lower": fraction_record(leading[0]), "upper": fraction_record(leading[1])},
        "coefficient_enclosure": {"lower": fraction_record(interval[0]), "upper": fraction_record(interval[1])},
        "relative_error_bound": fraction_record(relative_error_bound(face_size, degree)),
        "source_hash_policy": HASH_POLICY,
        "source_sha256": {
            name: hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() for name, source in _sources().items()
        },
    }
    if target is not None:
        threshold = relative_error_threshold(face_size, target)
        report["relative_target"] = fraction_record(target)
        report["all_tail"] = {
            "from_degree": threshold,
            "relative_error_bound": fraction_record(dyadic_tail_bound(face_size, threshold.bit_length() - 1)),
            "requested_degree_in_tail": degree >= threshold,
        }
    return report


def _canonical(record: object) -> str:
    active: set[int] = set()

    def encode(value: object) -> str:
        if type(value) is int:
            return _decimal(value)
        if value is None or type(value) in (str, bool, float):
            return json.dumps(value, allow_nan=False)
        if not isinstance(value, (dict, list)):
            raise TypeError("unsupported JSON value")
        identity = id(value)
        if identity in active:
            raise ValueError("circular JSON value")
        active.add(identity)
        try:
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise ValueError("JSON object keys must be strings")
                return "{" + ",".join(json.dumps(key) + ":" + encode(value[key]) for key in sorted(value)) + "}"
            return "[" + ",".join(encode(item) for item in value) + "]"
        finally:
            active.remove(identity)

    return encode(record)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _record_parameters(record: object) -> tuple[int, int, int, Fraction | None]:
    if type(record) is not dict or record.get("schema") != SCHEMA:
        raise ValueError("unsupported nonvertex q3 error certificate")
    face_size = _size(record.get("face_size"))
    degree = _integer(record.get("degree"), 1, "degree")
    terms = _integer(record.get("constant_terms"), 2, "constant_terms")
    target = None
    if "relative_target" in record:
        value = record["relative_target"]
        if (
            type(value) is not dict
            or set(value) != {"numerator", "denominator"}
            or any(type(x) is not str for x in value.values())
        ):
            raise ValueError("invalid recorded relative target")
        try:
            target = Fraction(_decimal_integer(value["numerator"]), _decimal_integer(value["denominator"]))
        except (ValueError, ZeroDivisionError) as cause:
            raise ValueError("invalid recorded relative target") from cause
    return face_size, degree, terms, target


def verify_certificate(record: object) -> None:
    """Regenerate the complete record with its own saved nondefault options."""
    parameters = _record_parameters(record)
    if _canonical(record) != _canonical(certificate(*parameters)):
        raise ValueError("saved certificate differs from source or exact recomputation")


def _target_argument(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        try:
            numerator, denominator = text.split("/")
            return Fraction(_decimal_integer(numerator), _decimal_integer(denominator))
        except (ValueError, ZeroDivisionError) as error:
            raise argparse.ArgumentTypeError("relative target must be a rational number") from error


def _protect_output(path: Path | None, sources: dict[str, bytes]) -> None:
    if path is not None:
        root = Path(__file__).resolve().parents[2]
        if path.resolve() in {(root / name).resolve() for name in sources}:
            raise ValueError("output must not overwrite evaluator or proof sources")


def main(argv: list[str] | None = None) -> int:
    """Save a certificate, optionally verifying the complete previous record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-size", type=_decimal_integer)
    parser.add_argument("--degree", type=_decimal_integer)
    parser.add_argument("--constant-terms", type=_decimal_integer)
    parser.add_argument("--relative-target", type=_target_argument)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    sources = _sources()
    try:
        _protect_output(arguments.output, sources)
        if arguments.verify is not None:
            previous = json.loads(arguments.verify.read_bytes(), object_pairs_hook=_unique, parse_int=_decimal_integer)
            saved = _record_parameters(previous)
            supplied = (arguments.face_size, arguments.degree, arguments.constant_terms, arguments.relative_target)
            if any(value is not None and value != recorded for value, recorded in zip(supplied, saved, strict=True)):
                raise ValueError("generation override differs from the saved certificate")
            verify_certificate(previous)
            report = previous
        else:
            if arguments.face_size is None or arguments.degree is None or arguments.output is None:
                raise ValueError("generation requires --face-size, --degree and --output")
            terms = 64 if arguments.constant_terms is None else arguments.constant_terms
            report = certificate(arguments.face_size, arguments.degree, terms, arguments.relative_target)
        if _sources() != sources:
            raise RuntimeError("nonvertex q3 error sources changed during calculation")
        output = None if arguments.output is None else arguments.output.parent.resolve() / arguments.output.name
        _protect_output(output, sources)
        if output is not None:
            atomic_json(output, report)
        else:
            print("verified saved nonvertex q3 error certificate")
    except (ValueError, TypeError, ArithmeticError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
