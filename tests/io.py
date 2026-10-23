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
