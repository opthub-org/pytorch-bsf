"""Geometry, dtype, topology and prediction survive persistence."""

import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import degrees, matrices, seeds, simploid_types
from torch_bsf import BezierModel, BezierSpace, SimploidalSet
from torch_bsf.io import load_model, save_model
from torch_bsf.sampling import sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["simplex", "product", "glued"])
@given(data=st.data(), seed=seeds())
def test_roundtrip(dtype, kind, data, seed):
    if kind == "glued":
        degree = data.draw(st.integers(1, 4))
        domain = SimploidalSet()
        a, b = domain.add_simplex(2), domain.add_simploid((1, 1))
        domain.identify(a.face(0, 0), b.face(0, 0))
        space = BezierSpace.on_set(domain, degree=degree)
    else:
        dimensions = (data.draw(st.integers(1, 3)),) if kind == "simplex" else data.draw(simploid_types(min_factors=2))
        factor_degrees = data.draw(degrees(dimensions, max_degree=4))
        space = BezierSpace.simploid(dimensions, degree=factor_degrees)
    n_values = data.draw(st.integers(1, 4))
    model = BezierModel(space, data.draw(matrices(space.n_controls, n_values)).to(dtype))
    points = sample(space, 12, seed=seed)
    # Every Hypothesis example owns its directory; a function-scoped fixture would
    # otherwise be reused across examples and could hide stale-file dependencies.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "arbitrary.extension"
        save_model(path, model)
        loaded = load_model(path)
    assert loaded.space.compatible(model.space)
    assert loaded.coefficients.dtype == dtype
    assert torch.equal(loaded.coefficients, model.coefficients)
    assert torch.equal(loaded(points), model(points))
    assert loaded.space.with_degree(3).n_cells == space.n_cells


def test_unknown_schema_rejected(tmp_path):
    path = tmp_path / "model"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.json", json.dumps({"schema_version": 999}))
        archive.writestr("coefficients.npy", b"")
    with pytest.raises(ValueError, match="schema version"):
        load_model(path)


def test_invalid_archive_and_dtype_metadata(tmp_path):
    import io

    import numpy as np

    model = BezierModel(BezierSpace.simplex(1, degree=1), [[1.0], [2.0]])
    saved = tmp_path / "original"
    save_model(saved, model)
    with zipfile.ZipFile(saved) as archive:
        metadata = json.loads(archive.read("model.json"))
        coefficients = archive.read("coefficients.npy")
    cases = [({"dtype": "unknown"}, coefficients), ({}, b"not-npy")]
    mismatched = io.BytesIO()
    np.save(mismatched, np.zeros((2, 1), dtype=np.float64), allow_pickle=False)
    cases.append(({}, mismatched.getvalue()))
    for i, (updates, array) in enumerate(cases):
        path = tmp_path / f"invalid-{i}"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("model.json", json.dumps({**metadata, **updates}))
            archive.writestr("coefficients.npy", array)
        with pytest.raises(ValueError):
            load_model(path)
    for text in ("{}", "not-json"):
        with zipfile.ZipFile(tmp_path / "invalid", "w") as archive:
            archive.writestr("model.json", text)
            archive.writestr("coefficients.npy", coefficients)
        with pytest.raises(ValueError):
            load_model(tmp_path / "invalid")
    with zipfile.ZipFile(tmp_path / "empty", "w"):
        pass
    with pytest.raises(ValueError):
        load_model(tmp_path / "empty")
    with pytest.raises(ValueError):
        save_model(tmp_path / "bf16", model.to(torch.bfloat16))


def test_atomic_failure_preserves_destination_and_csv_export(tmp_path):
    from unittest.mock import patch

    from torch_bsf.io import export_control_points

    model = BezierModel(BezierSpace.simplex(1, degree=1), [[1.0], [2.0]])
    saved = tmp_path / "model"
    save_model(saved, model)
    before = saved.read_bytes()
    with patch("torch_bsf.io.os.replace", side_effect=OSError("disk unavailable")), pytest.raises(OSError):
        save_model(saved, model)
    assert saved.read_bytes() == before
    assert list(tmp_path.iterdir()) == [saved]
    exported = tmp_path / "controls.csv"
    export_control_points(exported, model)
    assert exported.read_text().splitlines() == ["control_index,value_0", "0,1.0", "1,2.0"]
