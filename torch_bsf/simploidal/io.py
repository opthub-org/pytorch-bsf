r"""Saving and loading Bézier simploids."""

import csv
import json
import pickle
from ast import literal_eval
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import torch
import yaml

from torch_bsf.simploidal.bezier import BezierSimploid
from torch_bsf.simploidal.combinatorics import SimploidIndex


def _index_key(index: SimploidIndex) -> str:
    """The string key of a control point index, e.g. ``"((2, 0, 0), (1, 0))"``."""
    return str(tuple(tuple(int(v) for v in alpha) for alpha in index))


def _parse_index_key(key: object) -> SimploidIndex:
    """Parses a key written by :func:`_index_key`."""
    try:
        index = literal_eval(key) if isinstance(key, str) else None
    except (ValueError, SyntaxError):
        index = None
    if not (
        isinstance(index, tuple) and all(isinstance(a, tuple) and all(type(v) is int and v >= 0 for v in a) for a in index)
    ):
        raise ValueError(
            f"Invalid control point index {key!r}: expected a tuple of multi-indices such as '((2, 0, 0), (1, 0))'."
        )
    return cast(SimploidIndex, index)


def _from_items(items: Iterable[tuple[object, Sequence[Any]]], path: Path) -> BezierSimploid:
    """Builds a Bézier simploid from ``(index key, values)`` pairs, inferring its type and degree from the keys."""
    points: dict[SimploidIndex, list[float]] = {}
    for key, value in items:
        index = _parse_index_key(key)
        if index in points:
            raise ValueError(f"Duplicate control point index {_index_key(index)} in '{path}'.")
        points[index] = [float(v) for v in value]
    if not points:
        raise ValueError(f"No control points found in '{path}'.")
    first = next(iter(points))
    simploid_type = tuple(len(alpha) - 1 for alpha in first)
    degree = tuple(sum(alpha) for alpha in first)
    n_values = len(points[first])
    for index, value in points.items():
        if tuple(len(alpha) - 1 for alpha in index) != simploid_type or tuple(sum(alpha) for alpha in index) != degree:
            raise ValueError(
                f"Control point index {_index_key(index)} in '{path}' does not match the simploid type "
                f"{simploid_type} and degree {degree} inferred from {_index_key(first)}."
            )
        if len(value) != n_values:
            raise ValueError(f"Control point {_index_key(index)} in '{path}' has {len(value)} values, expected {n_values}.")
    return BezierSimploid(simploid_type, degree, {index: torch.tensor(value) for index, value in points.items()})


def save(path: str | Path, data: BezierSimploid) -> None:
    r"""Saves a Bézier simploid to a file.

    The format is chosen by the file extension:

    * ``.csv`` / ``.tsv``: one control point per row, its index key followed by its values;
    * ``.json`` / ``.yml`` / ``.yaml``: a mapping from index keys to lists of values;
    * ``.pt``: a dictionary with the keys ``"simploid_type"``, ``"degree"`` and
      ``"control_points"`` saved with :func:`torch.save`.

    An index key is the string representation of the index
    :math:`(\boldsymbol\alpha^1, \ldots, \boldsymbol\alpha^n)`, e.g. ``"((2, 0, 0), (1, 0))"``.
    Keys are written in the order of :func:`simploid_indices`.  The simploid type and the
    degree are not stored separately: :func:`load` recovers them from the keys.

    Parameters
    ----------
    path
        The file path to save.
    data
        The Bézier simploid to save.

    Raises
    ------
    ValueError
        If the file type is unknown.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import torch
    >>> from torch_bsf.simploidal import BezierSimploid, save
    >>> bs = BezierSimploid((1, 1), 1, torch.arange(8.0).reshape(4, 2))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     save(Path(d) / "simploid.csv", bs)
    ...     print((Path(d) / "simploid.csv").read_text(), end="")
    "((1, 0), (1, 0))",0.0,1.0
    "((1, 0), (0, 1))",2.0,3.0
    "((0, 1), (1, 0))",4.0,5.0
    "((0, 1), (0, 1))",6.0,7.0
    """
    path = Path(path)
    suffix = path.suffix.lower()
    control_points = data.control_points.detach().cpu()
    rows = [(_index_key(index), value.tolist()) for index, value in zip(data.indices(), control_points)]
    if suffix == ".pt":
        torch.save(
            {"simploid_type": list(data.simploid_type), "degree": list(data.degree), "control_points": control_points.clone()},
            path,
        )
    elif suffix in (".csv", ".tsv"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="," if suffix == ".csv" else "\t", lineterminator="\n")
            writer.writerows([key] + value for key, value in rows)
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(rows), f)
    elif suffix in (".yml", ".yaml"):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(rows), f, sort_keys=False)
    else:
        raise ValueError(f"Unknown file type: {path}")


def _load_pt(path: Path) -> BezierSimploid:
    message = f"'{path}' must contain a dictionary with the keys 'simploid_type', 'degree' and 'control_points'."
    try:
        # Only plain containers and tensors are accepted: loading never executes arbitrary code.
        data = torch.load(path, weights_only=True)
    except pickle.UnpicklingError as e:
        raise ValueError(message) from e
    if not (isinstance(data, dict) and {"simploid_type", "degree", "control_points"} <= data.keys()):
        raise ValueError(message)
    return BezierSimploid(data["simploid_type"], data["degree"], data["control_points"])


def load(path: str | Path) -> BezierSimploid:
    r"""Loads a Bézier simploid saved by :func:`save`.

    The simploid type and the degree are inferred from the control point indices;
    every control point of that type and degree must be present exactly once.

    Parameters
    ----------
    path
        The path to a ``.csv``, ``.tsv``, ``.json``, ``.yml``, ``.yaml`` or ``.pt`` file.

    Returns
    -------
    The Bézier simploid.

    Raises
    ------
    ValueError
        If the file type is unknown or the control points are invalid or incomplete.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import torch
    >>> import torch_bsf
    >>> from torch_bsf.simploidal import load, sample_simploid, save
    >>> t = sample_simploid((2, 1), 50, seed=0)
    >>> bs = torch_bsf.fit(t, t[:, :3] * t[:, 3:4], degree=1, simploid_type=(2, 1))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     save(Path(d) / "simploid.json", bs)
    ...     loaded = load(Path(d) / "simploid.json")
    >>> loaded.simploid_type, loaded.degree
    ((2, 1), (1, 1))
    >>> torch.allclose(loaded(t), bs(t))
    True
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return _load_pt(path)
    if suffix in (".csv", ".tsv"):
        with open(path, encoding="utf-8", newline="") as f:
            rows = [row for row in csv.reader(f, delimiter="," if suffix == ".csv" else "\t") if row]
        return _from_items(((row[0], row[1:]) for row in rows), path)
    if suffix not in (".json", ".yml", ".yaml"):
        raise ValueError(f"Unknown file type: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f) if suffix == ".json" else yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' must contain a mapping of control point indices to value lists, got {type(raw).__name__}.")
    return _from_items(raw.items(), path)
