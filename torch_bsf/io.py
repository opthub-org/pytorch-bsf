"""Versioned model persistence for both single-cell and glued-cell spaces."""

import csv
import io
import json
import os
import secrets
import stat
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from torch_bsf.core import BezierModel, BezierSpace, Cell

_DTYPES = {"torch.float16": np.dtype("float16"), "torch.float32": np.dtype("float32"), "torch.float64": np.dtype("float64")}


def save_model(path: str | Path, model: BezierModel) -> None:
    """Save an explicit schema and non-pickled coefficient array atomically.

    The format is a ZIP containing ``model.json`` and ``coefficients.npy``,
    independent of the filename extension and Python class import paths.
    The archive is written to a temporary sibling file, flushed to disk with
    :func:`os.fsync` and then renamed over ``path``. A new file gets the usual
    permissions ``0o666 & ~umask``; replacing an existing file keeps its mode.

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
    try:
        existing_mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None
    file, temporary = _create_temporary(path)
    try:
        with file:
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("model.json", json.dumps(document, ensure_ascii=False))
                archive.writestr("coefficients.npy", array.getvalue())
            if existing_mode is not None:
                os.chmod(temporary, existing_mode)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _create_temporary(path: Path) -> tuple[BinaryIO, Path]:
    """Exclusively create a sibling temporary file whose mode is ``0o666 & ~umask``.

    Unlike :func:`tempfile.mkstemp`, which always creates ``0o600`` files, the
    permissions match those of a file opened normally, so an atomic replacement
    does not silently make a new model private.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    for _ in range(100):
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
        return os.fdopen(descriptor, "wb"), temporary
    raise FileExistsError(f"could not create a temporary file next to '{path}'.")


def _fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX; directories cannot be opened for syncing elsewhere."""
    if os.name != "posix":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _tuples(value: Any) -> Any:
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def load_model(path: str | Path) -> BezierModel:
    """Restore a validated model without importing a training framework.

    Original mutable builder handles are not persisted. Loaded cells are
    addressed by the same stable integer IDs recorded in the model schema.

    Parameters
    ----------
    path : str or pathlib.Path
        Model archive written by :func:`save_model`.

    Returns
    -------
    BezierModel
        Validated model with restored coefficients, dtype and stable integer cell IDs.
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
    """Export coefficient rows as CSV data, explicitly without pretending to save geometry.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV file, replaced if it already exists.
    model : BezierModel
        Model whose global coefficient rows are written with output-column headers.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["control_index", *(f"value_{j}" for j in range(model.n_outputs))])
        writer.writerows((i, *row) for i, row in enumerate(model.coefficients.detach().cpu().tolist()))
