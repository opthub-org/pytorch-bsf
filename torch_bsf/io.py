"""Versioned model persistence for both single-cell and glued-cell spaces."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_bsf.core import BezierModel, BezierSpace, Cell

_DTYPES = {"torch.float16": np.dtype("float16"), "torch.float32": np.dtype("float32"), "torch.float64": np.dtype("float64")}


def save_model(path: str | Path, model: BezierModel) -> None:
    """Save an explicit schema and non-pickled coefficient array atomically.

    The format is a ZIP containing ``model.json`` and ``coefficients.npy``,
    independent of the filename extension and Python class import paths.

    Parameters
    ----------
    path
        Destination file; its parent directory must exist.
    model
        Any unified Bézier model, including a glued multi-cell space.
    """
    path = Path(path)
    dtype = str(model.coefficients.dtype)
    if dtype not in _DTYPES:
        raise ValueError("model persistence supports float16, float32 and float64 coefficients.")
    space = model.space
    document = {
        "schema_version": 1,
        "dtype": dtype,
        "space": {
            "cells": [{"dimensions": c.dimensions, "degree": c.degree, "controls": c.controls} for c in space.cells],
            "n_controls": space.n_controls,
            "main_cells": space.main_cells,
            "edges": space._edges,
            "topology": space._topology,
        },
    }
    array = io.BytesIO()
    np.save(array, model.coefficients.detach().cpu().numpy(), allow_pickle=False)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = file.name
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("model.json", json.dumps(document, ensure_ascii=False))
                archive.writestr("coefficients.npy", array.getvalue())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _tuples(value: Any) -> Any:
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def load_model(path: str | Path) -> BezierModel:
    """Restore a validated model without importing a training framework.

    Original mutable builder handles are not persisted. Loaded cells are
    addressed by the same stable integer IDs recorded in the model schema.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["coefficients.npy", "model.json"]:
                raise ValueError("model archive must contain exactly its schema and coefficients.")
            metadata = json.loads(archive.read("model.json"))
            if metadata["schema_version"] != 1:
                raise ValueError("unsupported model schema version.")
            if metadata["dtype"] not in _DTYPES:
                raise ValueError("unsupported coefficient dtype.")
            data = metadata["space"]
            cells = tuple(Cell(tuple(c["dimensions"]), tuple(c["degree"]), tuple(c["controls"])) for c in data["cells"])
            space = BezierSpace(
                cells,
                data["n_controls"],
                tuple(data["main_cells"]),
                tuple(map(tuple, data["edges"])),
                _topology=_tuples(data["topology"]),
            )
            coefficients = np.load(io.BytesIO(archive.read("coefficients.npy")), allow_pickle=False)
            if coefficients.dtype != _DTYPES[metadata["dtype"]]:
                raise ValueError("coefficient dtype does not match the schema.")
        return BezierModel(space, torch.from_numpy(coefficients.copy()))
    except (KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise ValueError("invalid model archive or schema.") from error


def export_control_points(path: str | Path, model: BezierModel) -> None:
    """Export coefficient rows as CSV data, explicitly without pretending to save geometry."""
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["control_index", *(f"value_{j}" for j in range(model.n_outputs))])
        writer.writerows((i, *row) for i, row in enumerate(model.coefficients.detach().cpu().tolist()))
