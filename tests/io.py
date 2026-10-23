"""Geometry, dtype, topology and prediction survive persistence."""

import json
import zipfile

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, SimploidalSet
from torch_bsf.io import load_model, save_model
from torch_bsf.sampling import sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["simplex", "product", "glued"])
def test_roundtrip(tmp_path, dtype, kind):
    if kind == "glued":
        domain = SimploidalSet()
        a, b = domain.add_simplex(2), domain.add_simploid((1, 1))
        domain.identify(a.face(0, 0), b.face(0, 0))
        space = BezierSpace.on_set(domain, degree=2)
    else:
        space = BezierSpace.simploid((1, 1) if kind == "product" else (2,), degree=2)
    model = BezierModel(space, torch.randn(space.n_controls, 3, dtype=dtype))
    points = sample(space, 40, seed=42)
    path = tmp_path / "arbitrary.extension"
    save_model(path, model)
    loaded = load_model(path)
    assert loaded.space.compatible(model.space)
    assert loaded.coefficients.dtype == dtype
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
