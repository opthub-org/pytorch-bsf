"""Native rational certificates and rejection of false family reductions."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.risk_research import codimension_two_family_certificate as certificate

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def sage_python():
    executable = os.environ.get("BSF_SAGE_PYTHON")
    if executable is None:
        pytest.skip("set BSF_SAGE_PYTHON for native family construction certificates")
    return executable


def _native(executable, source, *arguments):
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).resolve().parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run(
        [executable, "-c", source, *arguments], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("face_size", [2, 3, 5, 6, 9, 10])
def test_native_symbolic_construction_and_domain_proofs(sage_python, face_size):
    result = _native(
        sage_python,
        """
import json,sys
from tools.risk_research.codimension_two_family_certificate import verify_construction
print(json.dumps(verify_construction(int(sys.argv[1]))))
""",
        str(face_size),
    )
    proof = json.loads(result.stdout)
    assert proof["status"] == "certified_algebraic_construction"
    assert proof["state_dimension"] == 15
    assert proof["active_domain_start"] == face_size
    assert proof["inactive_indices"] == list(range(1, face_size))
    assert proof["partial_fraction_residuals"] == ["0"] * 3
    assert proof["polynomial_antidifference_residuals"] == ["0"] * 3
    assert proof["universal_identity_residuals"] == ["0"] * 5
    assert proof["independent_polynomial_gosper_check"] is True
    assert proof["boundary_polynomial_identities"] is True
    assert proof["output_assembly_residuals"] == ["0"] * 15
    assert proof["output_denominators_positive_on_domain"] is True
    assert proof["both_transition_denominators_positive_from_h"] == 1
    assert len(proof["output_row"]) == 15


@pytest.mark.parametrize("mutation", ["partial", "output constant", "output pole", "transition pole"])
def test_native_verifier_rejects_false_algebra_and_invalid_domain(sage_python, mutation):
    _native(
        sage_python,
        """
import sys
from tools.risk_research import codimension_two_family as family
from tools.risk_research.codimension_two_family_certificate import verify_construction
mutation=sys.argv[1]
if mutation=='partial':
    original=family._partial
    def changed(p,poles):
        quotient,fractions=original(p,poles)
        i,k,c=fractions[0]
        fractions[0]=(i,k,c+1)
        return quotient,fractions
    family._partial=changed
elif mutation in ('output pole','output constant'):
    original=family._output_row
    def changed(s,h):
        row=original(s,h)
        row[0]+=1/(h-s-s//2) if mutation=='output pole' else 1
        return row
    family._output_row=changed
else:
    original=family._transition_rows
    def changed(h,parity):
        rows=original(h,parity)
        rows[0][0]+=1/(h-1)
        return rows
    family._transition_rows=changed
try:
    verify_construction(5)
except ArithmeticError:
    pass
else:
    raise AssertionError('false construction accepted')
""",
        mutation,
    )


def test_native_certificate_cli_roundtrips_report_and_both_source_hashes(sage_python, tmp_path):
    output = tmp_path / "proof.json"
    result = _native(
        sage_python,
        """
import hashlib,sys
from pathlib import Path
from tools.risk_research import codimension_two_family as family
from tools.risk_research import codimension_two_family_certificate as certificate
certificate.main(['--face-size','6','--output',sys.argv[1]])
""",
        str(output),
    )
    proof = json.loads(output.read_bytes())
    assert proof == json.loads(result.stdout)
    import hashlib

    from tools.risk_research import codimension_two_family as family

    assert proof["source_sha256"] == hashlib.sha256(Path(certificate.__file__).read_bytes()).hexdigest()
    assert proof["family_source_sha256"] == hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize("invalid", [True, 1, 0, -1, 2.0, "2"])
def test_invalid_face_is_rejected_before_optional_sage_import(monkeypatch, invalid):
    def forbidden(name):
        raise AssertionError("optional import happened before validation")

    monkeypatch.setattr(certificate, "import_module", forbidden)
    with pytest.raises(ValueError, match="face_size"):
        certificate.verify_construction(invalid)
