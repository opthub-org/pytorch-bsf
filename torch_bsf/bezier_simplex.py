import csv
import json
from ast import literal_eval
from functools import lru_cache
from math import factorial
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import lightning.pytorch as L
import numpy as np
import torch
import torch.optim
import yaml
from jsonschema import ValidationError, validate
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from torch_bsf.control_points import (
    ControlPoints,
    ControlPointsData,
    Index,
    simplex_indices,
    to_parameterdict_key,
)
from torch_bsf.preprocessing import MinMaxScaler, NoneScaler, QuantileScaler, StdScaler
from torch_bsf.validator import validate_simplex_indices

NormalizeType = Literal["max", "std", "quantile", "none"]


class BezierSimplexDataModule(L.LightningDataModule):
    r"""A data module for training a Bezier simplex.

    Parameters
    ----------
    params
        The path to a parameter file.
    values
        The path to a value file.
    header
        The number of header rows in the parameter file and the value file.
        The first ``header`` rows are skipped in reading the files.
    batch_size
        The size of each minibatch.
    split_ratio
        The ratio of train-val split.
        Must be greater than 0 and less than or equal to 1.
        If it is set to 1, then all the data are used for training and the validation step will be skipped.
    normalize
        The data normalization method.
        Either ``"max"``, ``"std"``, ``"quantile"``, or ``"none"``.
    """

    def __init__(
        self,
        params: Path,
        values: Path,
        header: int = 0,
        batch_size: int | None = None,
        split_ratio: float = 1.0,
        normalize: NormalizeType = "none",  # "max", "std", "quantile" or "none"
    ):
        # REQUIRED
        super().__init__()
        if header < 0:
            raise ValueError(f"{header=}. Must be non-negative.")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"{batch_size=}. Must be positive or None.")
        if split_ratio <= 0.0 or 1.0 < split_ratio:
            raise ValueError(f"{split_ratio=}. Must be 0.0 < sprit_ratio <= 1.0.")
        if normalize not in ("max", "std", "quantile", "none"):
            raise ValueError(f"{normalize=}. Must be one of ['max', 'std', 'quantile', 'none'].")

        self.params = params
        self.values = values
        self.header = header
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.scaler: MinMaxScaler | StdScaler | QuantileScaler | NoneScaler
        if normalize == "max":
            self.scaler = MinMaxScaler()
        elif normalize == "std":
            self.scaler = StdScaler()
        elif normalize == "quantile":
            self.scaler = QuantileScaler()
        else:
            self.scaler = NoneScaler()

        with open(self.params) as f:
            delimiter = "," if self.params.suffix == ".csv" else None
            self.n_params = len(f.readline().split(delimiter))
        with open(self.values) as f:
            delimiter = "," if self.values.suffix == ".csv" else None
            self.n_values = len(f.readline().split(delimiter))
        self.setup()

    def setup(self, stage: str | None = None):
        # OPTIONAL
        params = self.load_params()
        values = self.load_values()
        values = self.fit_transform(values)
        xy = TensorDataset(params, values)
        size = len(xy)
        if self.split_ratio == 1.0:
            self.trainset: TensorDataset | Subset[tuple[torch.Tensor, ...]] = xy
            self.valset: TensorDataset | Subset[tuple[torch.Tensor, ...]] = self.trainset
        else:
            n_train = int(size * self.split_ratio)
            trainset, valset = random_split(xy, [n_train, size - n_train])
            self.trainset = trainset
            self.valset = valset

    def load_data(self, path) -> torch.Tensor:
        delimiter = "," if path.suffix == ".csv" else None
        return torch.from_numpy(
            np.loadtxt(path, delimiter=delimiter, skiprows=self.header, ndmin=2)
        ).to(torch.get_default_dtype())

    def load_params(self) -> torch.Tensor:
        return self.load_data(self.params)

    def load_values(self) -> torch.Tensor:
        return self.load_data(self.values)

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        return self.scaler.fit_transform(values)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse_transform(values)

    def train_dataloader(self) -> DataLoader:
        # REQUIRED
        return DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=self.batch_size or len(self.trainset),
        )

    def val_dataloader(self) -> DataLoader:
        # OPTIONAL
        return DataLoader(
            self.valset,
            batch_size=self.batch_size or len(self.valset),
        )

    def test_dataloader(self) -> DataLoader:
        # OPTIONAL
        return self.val_dataloader()


@lru_cache(1024)
def polynom(degree: int, index: Iterable[int]) -> float:
    r"""Computes a polynomial coefficient :math:`\binom{D}{\mathbf d} = \frac{D!}{d_1!d_2!\cdots d_M!}`.

    Parameters
    ----------
    degree
        The degree :math:`D`.
    index
        The index :math:`\mathbf d`.

    Returns
    -------
    The polynomial coefficient :math:`\binom{D}{\mathbf d}`.
    """
    r: float = factorial(degree)
    for i in index:
        r /= factorial(i)
    return r


def monomial(variable: Iterable[float], degree: Iterable[int]) -> torch.Tensor:
    r"""Computes a monomial :math:`\mathbf t^{\mathbf d} = t_1^{d_1} t_2^{d_2}\cdots t_M^{d^M}`.

    Parameters
    ----------
    variable
        The bases :math:`\mathbf t`.
    degree
        The powers :math:`\mathbf d`.

    Returns
    -------
    The monomial :math:`\mathbf t^{\mathbf d}`.
    """
    v = torch.as_tensor(variable)
    d = torch.as_tensor(degree, device=v.device)
    ret: torch.Tensor = (v**d).prod(dim=-1)
    return ret


class BezierSimplex(L.LightningModule):
    r"""A Bezier simplex model.

    Parameters
    ----------
    control_points
        The control points of the Bezier simplex.  Pass ``None`` only when
        reconstructing a model from a Lightning checkpoint via
        :meth:`load_from_checkpoint` — in that case all three shape
        parameters (``_n_params``, ``_degree``, ``_n_values``) must be
        provided so that a correctly-shaped placeholder can be built before
        the saved state dict is loaded into it.
    smoothness_weight
        The weight of the smoothness penalty term added to the training loss.
        When greater than zero, adjacent control points are encouraged to have
        similar values.  Defaults to ``0.0`` (no penalty).
    _n_params
        *Checkpoint-reconstruction parameter — do not set manually.*
        The number of parameters (source dimension + 1) used to build the
        placeholder control points when ``control_points`` is ``None``.
        Automatically saved to, and restored from, Lightning checkpoints.
    _degree
        *Checkpoint-reconstruction parameter — do not set manually.*
        The degree of the Bezier simplex used to build the placeholder when
        ``control_points`` is ``None``.
        Automatically saved to, and restored from, Lightning checkpoints.
    _n_values
        *Checkpoint-reconstruction parameter — do not set manually.*
        The number of values (target dimension) used to build the placeholder
        when ``control_points`` is ``None``.
        Automatically saved to, and restored from, Lightning checkpoints.

    Examples
    --------
    >>> import lightning.pytorch as L
    >>> from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> ts = torch.tensor(  # parameters on a simplex
    ...     [
    ...         [3/3, 0/3, 0/3],
    ...         [2/3, 1/3, 0/3],
    ...         [2/3, 0/3, 1/3],
    ...         [1/3, 2/3, 0/3],
    ...         [1/3, 1/3, 1/3],
    ...         [1/3, 0/3, 2/3],
    ...         [0/3, 3/3, 0/3],
    ...         [0/3, 2/3, 1/3],
    ...         [0/3, 1/3, 2/3],
    ...         [0/3, 0/3, 3/3],
    ...     ]
    ... )
    >>> xs = 1 - ts * ts  # values corresponding to the parameters
    >>> dl = DataLoader(TensorDataset(ts, xs))
    >>> bs = torch_bsf.bezier_simplex.randn(
    ...     n_params=int(ts.shape[1]),
    ...     n_values=int(xs.shape[1]),
    ...     degree=3,
    ... )
    >>> trainer = L.Trainer(
    ...     callbacks=[EarlyStopping(monitor="train_mse")],
    ...     enable_progress_bar=False,
    ... )
    >>> trainer.fit(bs, dl)
    >>> ts, xs = bs.meshgrid()

    """

    def __init__(
        self,
        control_points: ControlPoints | ControlPointsData | None = None,
        smoothness_weight: float = 0.0,
        _n_params: int | None = None,
        _degree: int | None = None,
        _n_values: int | None = None,
    ):
        # REQUIRED
        super().__init__()
        if control_points is None:
            # Called by load_from_checkpoint: reconstruct a placeholder from
            # the saved dimensions so that the state dict can be loaded into it.
            if _n_params is None or _degree is None or _n_values is None:
                raise TypeError(
                    "BezierSimplex.__init__() requires either 'control_points' "
                    "or all of '_n_params', '_degree', and '_n_values' "
                    "(e.g. when loading from a checkpoint)."
                )
            control_points = {
                idx: [0.0 for _ in range(_n_values)]
                for idx in simplex_indices(_n_params, _degree)
            }
        self.control_points = (
            control_points
            if isinstance(control_points, ControlPoints)
            else ControlPoints(control_points)
        )
        self.smoothness_weight = smoothness_weight
        # save_hyperparameters() captures local variable values at the point of
        # the call.  Reassigning _n_params/_degree/_n_values from the actual
        # model dimensions here ensures they are stored correctly in every
        # checkpoint — both when called normally (where the caller passes None
        # for these) and when called by load_from_checkpoint (where the caller
        # passes the saved values).  Without this reassignment, a normal save
        # would persist _n_params=None and checkpoint loading would fail.
        _n_params = self.n_params
        _degree = self.degree
        _n_values = self.n_values
        # Exclude the submodule from hyperparameters; it is already saved as
        # part of the module state dict.
        self.save_hyperparameters(ignore=["control_points"])

        # Track row indices whose gradients should be zeroed (frozen control points).
        self._fixed_rows: set[int] = set()

        # Cache indices and coefficients for vectorized forward
        if self.n_params > 0:
            indices = list(self.control_points.indices())
            self.register_buffer(
                "indices_",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "coeffs_",
                torch.tensor(
                    [polynom(self.degree, i) for i in indices],
                    dtype=torch.get_default_dtype(),
                ),
                persistent=False,
            )

            # Adjacency for smoothness penalty.
            # Built analytically in O(N·M²) instead of the naive O(N²) scan:
            # two control points are adjacent iff their indices differ by
            # exactly +1 in one component and −1 in another (L₁ distance = 2).
            if smoothness_weight > 0:
                index_to_row = {idx: row for row, idx in enumerate(indices)}
                adj: list[tuple[int, int]] = []
                m = self.n_params
                for a_row, a in enumerate(indices):
                    for i in range(m):
                        if a[i] == 0:
                            continue
                        for j in range(m):
                            if i == j:
                                continue
                            b = list(a)
                            b[i] -= 1
                            b[j] += 1
                            b_row = index_to_row.get(tuple(b))
                            if b_row is not None and b_row > a_row:
                                adj.append((a_row, b_row))
                if adj:
                    self.register_buffer(
                        "adjacency_indices_",
                        torch.tensor(adj, dtype=torch.long),
                        persistent=False,
                    )

    @property
    def n_params(self) -> int:
        r"""The number of parameters, i.e., the source dimension + 1."""
        return self.control_points.n_params

    @property
    def n_values(self) -> int:
        r"""The number of values, i.e., the target dimension."""
        return self.control_points.n_values

    @property
    def degree(self) -> int:
        r"""The degree of the Bezier simplex."""
        return self.control_points.degree

    def fix_row(self, index: "Index") -> None:
        """Freeze a control point so its gradient is zeroed after every backward.

        Parameters
        ----------
        index
            The index of the control point to freeze.
        """
        from ast import literal_eval as _literal_eval

        k = to_parameterdict_key(index)
        idx = tuple(_literal_eval(k))
        row = self.control_points._index_to_row[idx]
        self._fixed_rows.add(row)

    def on_after_backward(self) -> None:
        """Zero gradients for frozen control-point rows after each backward pass."""
        if not self._fixed_rows:
            return
        grad = self.control_points.matrix.grad
        if grad is not None:
            grad[list(self._fixed_rows)] = 0.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        r"""Process a forwarding step of training.

        Parameters
        ----------
        t
            A minibatch of parameter vectors :math:`\mathbf t`.

        Returns
        -------
        A minibatch of value vectors.
        """
        # REQUIRED
        if self.n_params == 0:
            # Constant simplex: single control-point row, broadcast over batch.
            return self.control_points.matrix[0].unsqueeze(0).expand(t.shape[0], -1)

        t = torch.as_tensor(t, device=self.device, dtype=cast(torch.Tensor, self.coeffs_).dtype)

        # Vectorized monomial calculation: (batch, n_indices)
        monomials = torch.pow(t.unsqueeze(1), cast(torch.Tensor, self.indices_).unsqueeze(0)).prod(dim=-1)

        # self.control_points.matrix is a direct nn.Parameter reference — no
        # Python loop or torch.stack overhead here.
        # Weighted control points (n_indices, n_values)
        wcp = cast(torch.Tensor, self.coeffs_).unsqueeze(1) * self.control_points.matrix

        # Matrix multiplication: (batch, n_indices) @ (n_indices, n_values) -> (batch, n_values)
        return torch.matmul(monomials, wcp)

    def smoothness_penalty(self) -> torch.Tensor:
        """Computes the smoothness penalty of the Bezier simplex.

        Returns
        -------
            The smoothness penalty.
        """
        X = self.control_points.matrix
        if not hasattr(self, "adjacency_indices_"):
            # Return a scalar tensor on the same device/dtype so this composes
            # safely with the training loss (e.g., on GPU/AMP).
            return torch.zeros((), device=X.device, dtype=X.dtype)
        adj_indices = cast(torch.Tensor, self.adjacency_indices_)
        i = adj_indices[:, 0]
        j = adj_indices[:, 1]
        return torch.sum((X[i] - X[j]).pow(2))

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        loss = mse
        if self.smoothness_weight > 0:
            loss += self.smoothness_weight * self.smoothness_penalty()
        tensorboard_logs = {"train_loss": loss, "train_mse": mse}
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_mse", mse, sync_dist=True)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx) -> None:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("val_mse", mse, sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        self.log("val_avg_mse", mse, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", mae, sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # REQUIRED
        optimizer = torch.optim.LBFGS(self.parameters())
        return optimizer

    def meshgrid(self, num: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a meshgrid of the Bezier simplex.

        Parameters
        ----------
        num
            The number of grid points on each edge.

        Returns
        -------
        ts
            A parameter matrix of the mesh grid.
        xs
            A value matrix of the mesh grid.
        """
        from torch_bsf.sampling import simplex_grid

        # Determine an appropriate dtype for the grid:
        # - Prefer self.coeffs_.dtype when coefficients are registered.
        # - For constant simplices (n_params == 0), fall back to the dtype of
        #   the control point at the empty index, if available.
        # - As a last resort, use torch.get_default_dtype().
        if hasattr(self, "coeffs_"):
            dtype = cast(torch.Tensor, self.coeffs_).dtype
        else:
            try:
                cp = self.control_points[()]
            except (KeyError, IndexError):
                cp = None
            if isinstance(cp, torch.Tensor):
                dtype = cp.dtype
            else:
                dtype = torch.get_default_dtype()

        ts = simplex_grid(n_params=self.n_params, degree=num).to(
            device=self.device, dtype=dtype
        )
        xs = self.forward(ts)
        return ts, xs


def zeros(
    n_params: int, n_values: int, degree: int, smoothness_weight: float = 0.0
) -> BezierSimplex:
    r"""Generates a Bezier simplex with control points at origin.

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.
    smoothness_weight
        The weight of smoothness penalty.

    Returns
    -------
        A Bezier simplex filled with zeros.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.zeros(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(n_params=2, degree=2, n_values=3)
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))
    tensor([[..., ..., ...]], grad_fn=<...>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.zeros(n_values) for i in simplex_indices(n_params, degree)},
        smoothness_weight=smoothness_weight,
    )


def rand(
    n_params: int, n_values: int, degree: int, smoothness_weight: float = 0.0
) -> BezierSimplex:
    r"""Generates a random Bezier simplex.

    The control points are initialized by random values.
    The values are uniformly distributed in [0, 1).

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.
    smoothness_weight
        The weight of smoothness penalty.

    Returns
    -------
        A random Bezier simplex.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.rand(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(n_params=2, degree=2, n_values=3)
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))  # doctest: +ELLIPSIS
    tensor([[..., ..., ...]], grad_fn=<...>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.rand(n_values) for i in simplex_indices(n_params, degree)},
        smoothness_weight=smoothness_weight,
    )


def randn(
    n_params: int, n_values: int, degree: int, smoothness_weight: float = 0.0
) -> BezierSimplex:
    r"""Generates a random Bezier simplex.

    The control points are initialized by random values.
    The values are normally distributed with mean 0 and standard deviation 1.

    Parameters
    ----------
    n_params
        The number of parameters, i.e., the source dimension + 1.
    n_values
        The number of values, i.e., the target dimension.
    degree
        The degree of the Bezier simplex.
    smoothness_weight
        The weight of smoothness penalty.

    Returns
    -------
        A random Bezier simplex.

    Raises
    ------
    ValueError
        If ``n_params`` or ``n_values`` or ``degree`` is negative.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.randn(n_params=2, n_values=3, degree=2)
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(n_params=2, degree=2, n_values=3)
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))  # doctest: +ELLIPSIS
    tensor([[..., ..., ...]], grad_fn=<...>)
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative: {n_params}")
    if n_values < 0:
        raise ValueError(f"n_values must be non-negative: {n_values}")
    if degree < 0:
        raise ValueError(f"degree must be non-negative: {degree}")

    return BezierSimplex(
        {i: torch.randn(n_values) for i in simplex_indices(n_params, degree)},
        smoothness_weight=smoothness_weight,
    )


def save(path: str | Path, data: BezierSimplex) -> None:
    r"""Saves a Bezier simplex to a file.

    Parameters
    ----------
    path
        The file path to save.
    data
        The Bezier simplex to save.

    Raises
    ------
    ValueError
        If the file type is unknown.

    Examples
    --------
    >>> import torch_bsf
    >>> bs = torch_bsf.bezier_simplex.randn(n_params=2, n_values=3, degree=2)
    >>> torch_bsf.bezier_simplex.save("tests/data/bezier_simplex.pt", bs)
    >>> torch_bsf.bezier_simplex.save("tests/data/bezier_simplex.csv", bs)
    >>> torch_bsf.bezier_simplex.save("tests/data/bezier_simplex.tsv", bs)
    >>> torch_bsf.bezier_simplex.save("tests/data/bezier_simplex.json", bs)
    >>> torch_bsf.bezier_simplex.save("tests/data/bezier_simplex.yml", bs)

    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        torch.save(data, path)

    elif suffix == ".csv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for index, value in data.control_points.items():
                writer.writerow([index] + value.tolist())

    elif suffix == ".tsv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for index, value in data.control_points.items():
                writer.writerow([index] + value.tolist())

    elif suffix == ".json":
        dic = {index: value.tolist() for index, value in data.control_points.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dic, f)

    elif suffix in (".yml", ".yaml"):
        dic = {to_parameterdict_key(index): value.tolist() for index, value in data.control_points.items()}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(dic, f)

    else:
        raise ValueError(f"Unknown file type: {path}")


CONTROLPOINTS_JSONSCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "patternProperties": {
        r"^\((\d+, *)*\d*\)$": {  # e.g., "(0, 0, 0)": [0.0, 0.0, 0.0]
            "type": "array",
            "items": {
                "type": "number",
            },
        }
    },
    "additionalProperties": False,
}


def validate_control_points(data: dict[str, list[float]]):
    r"""Validates control points.

    Parameters
    ----------
    data
        The control points.

    Raises
    ------
    ValidationError
        If the control points are invalid.

    Examples
    --------
    >>> from torch_bsf.bezier_simplex import validate_control_points
    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0],
    ... })

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "0, 0, 1": [0.0, 0.0, 1.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: '0, 0, 1' is not valid under any of the given schemas

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0],
    ...     "(0, 0)": [0.0, 0.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: (0, 0)

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1, 0)": [0.0, 0.0, 1.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: (0, 0, 1, 0)

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0, 1.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: [0.0, 0.0, 1.0, 0.0]

    >>> validate_control_points({
    ...     "(1, 0, 0)": [1.0, 0.0, 0.0],
    ...     "(0, 1, 0)": [0.0, 1.0, 0.0],
    ...     "(0, 0, 1)": [0.0, 0.0],
    ... })
    Traceback (most recent call last):
        ...
    jsonschema.exceptions.ValidationError: Dimension mismatch: [0.0, 0.0]
    """
    validate(instance=data, schema=CONTROLPOINTS_JSONSCHEMA)
    index, value = next(iter(data.items()))
    n_params = len(literal_eval(index))
    n_values = len(value)
    for index, value in data.items():
        if len(literal_eval(index)) != n_params:
            raise ValidationError(f"Dimension mismatch: {index}")
        if len(value) != n_values:
            raise ValidationError(f"Dimension mismatch: {value}")


def load(
    path: str | Path,
    *,
    pt_weights_only: bool | None = None,
) -> BezierSimplex:
    r"""Loads a Bezier simplex from a file.

    Parameters
    ----------
    path
        The path to a file.
    pt_weights_only
        Whether to load weights only. This parameter is only effective when loading PyTorch (``.pt``) files.
        For other formats (e.g., ``.json``, ``.yml``), data loading is inherently safe and this parameter is ignored.
        If ``None``, it defaults to ``False``.

    Returns
    -------
    A Bezier simplex.

    Raises
    ------
    ValueError
        If the file type is unknown.
    ValidationError
        If the control points are invalid.

    Notes
    -----
    Setting ``pt_weights_only=True`` will fail if the model contains
    classes not allowed by PyTorch's ``WeightsUnpickler`` (like lightning's
    ``AttributeDict``), even if they are in the safe globals list.

    Examples
    --------
    >>> from torch_bsf import bezier_simplex
    >>> bs = bezier_simplex.load("tests/data/bezier_simplex.csv")
    >>> print(bs)
    BezierSimplex(
      (control_points): ControlPoints(n_params=2, degree=2, n_values=3)
    )
    >>> print(bs(torch.tensor([[0.2, 0.8]])))
    tensor([[..., ..., ...]], grad_fn=<...>)
    """
    cpdata: dict[str, list[float]]
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        has_safe_globals = hasattr(torch.serialization, "safe_globals")

        kwargs: dict[str, Any] = {}
        import inspect

        has_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        assert not (has_safe_globals and not has_weights_only)

        # PyTorch 2.6 defaults to True, but our models contain Lightning's AttributeDict
        # which fails under weights_only=True due to PyTorch's SETITEM restrictions.
        # Therefore, we currently default to False to maintain usability.
        # The safe_globals implementation below is retained as a forward-compatible
        # foundation for when upstream support improves.
        if pt_weights_only is None:
            pt_weights_only = False

        if has_weights_only:
            kwargs["weights_only"] = pt_weights_only

        if has_safe_globals and pt_weights_only:
            safe_classes: list[Any] = [
                BezierSimplex,
                ControlPoints,
                MinMaxScaler,
                StdScaler,
                QuantileScaler,
                NoneScaler,
            ]
            try:
                from lightning.fabric.utilities.data import AttributeDict

                safe_classes.append(AttributeDict)
            except (ImportError, AttributeError):
                pass

            with torch.serialization.safe_globals(safe_classes):
                data = torch.load(path, **kwargs)
        else:
            data = torch.load(path, **kwargs)

        if isinstance(data, BezierSimplex):
            return data
        raise ValueError(f"Unknown data type: {type(data)}")

    elif suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            cpdata = {
                to_parameterdict_key(row[0]): [float(v) for v in row[1:]]
                for row in csv.reader(f)
                if row
            }
        if not cpdata:
            raise ValueError(f"No control points found in '{path}'")
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif suffix == ".tsv":
        with open(path, encoding="utf-8", newline="") as f:
            cpdata = {
                to_parameterdict_key(row[0]): [float(v) for v in row[1:]]
                for row in csv.reader(f, delimiter="\t")
                if row
            }
        if not cpdata:
            raise ValueError(f"No control points found in '{path}'")
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"JSON file '{path}' must contain a mapping of control point keys to value lists, got {type(raw).__name__}"
            )
        if not raw:
            raise ValueError(f"No control points found in '{path}'")
        cpdata = {
            to_parameterdict_key(index): [float(v) for v in value]
            for index, value in raw.items()
        }
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    elif suffix in (".yml", ".yaml"):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML file '{path}' must contain a mapping of control point keys to value lists, got {type(raw).__name__}"
            )
        if not raw:
            raise ValueError(f"No control points found in '{path}'")
        cpdata = {
            to_parameterdict_key(index): [float(v) for v in value]
            for index, value in raw.items()
        }
        validate_control_points(cpdata)
        return BezierSimplex(cpdata)

    else:
        raise ValueError(f"Unknown file type: {path}")


def fit(
    params: torch.Tensor,
    values: torch.Tensor,
    degree: int | None = None,
    init: BezierSimplex | ControlPoints | ControlPointsData | None = None,
    smoothness_weight: float = 0.0,
    fix: Iterable[Index] | None = None,
    batch_size: int | None = None,
    **kwargs,
) -> BezierSimplex:
    r"""Fits a Bezier simplex.

    Parameters
    ----------
    params
        The data.
    values
        The label data.
    degree
        The degree of the Bezier simplex.
    init
        The initial values of a bezier simplex or control points.
    smoothness_weight
        The weight of smoothness penalty.
    fix
        The indices of control points to exclude from training.
    batch_size
        The size of minibatch.
    kwargs
        All arguments for lightning.pytorch.Trainer

    Returns
    -------
    A trained Bezier simplex.

    Raises
    ------
    TypeError
        From Trainer or DataLoader.
    MisconfigurationException
        From Trainer.

    Examples
    --------
    >>> import torch
    >>> import torch_bsf

    Prepare training data

    >>> ts = torch.tensor(  # parameters on a simplex
    ...     [
    ...         [3/3, 0/3, 0/3],
    ...         [2/3, 1/3, 0/3],
    ...         [2/3, 0/3, 1/3],
    ...         [1/3, 2/3, 0/3],
    ...         [1/3, 1/3, 1/3],
    ...         [1/3, 0/3, 2/3],
    ...         [0/3, 3/3, 0/3],
    ...         [0/3, 2/3, 1/3],
    ...         [0/3, 1/3, 2/3],
    ...         [0/3, 0/3, 3/3],
    ...     ]
    ... )
    >>> xs = 1 - ts * ts  # values corresponding to the parameters

    Train a model

    >>> bs = torch_bsf.fit(params=ts, values=xs, degree=3)

    Predict by the trained model

    >>> t = [[0.2, 0.3, 0.5]]
    >>> x = bs(t)
    >>> print(f"{t} -> {x}")
    [[0.2, 0.3, 0.5]] -> tensor([[..., ..., ...]], grad_fn=<...>)

    See Also
    --------
    lightning.pytorch.Trainer : Argument descriptions.
    torch.DataLoader : Argument descriptions.
    """
    data = TensorDataset(params, values)
    dl = DataLoader(data, batch_size=batch_size or len(data))

    if degree is None and init is None:
        raise ValueError("Either degree or init must be specified")
    if degree is not None and init is not None:
        raise ValueError("Either degree or init must be specified, not both")

    if isinstance(init, BezierSimplex):
        # If the existing BezierSimplex already has the desired smoothness_weight
        # and an adjacency buffer, we can safely reuse it. Otherwise, recreate a
        # new instance from its control points so that __init__ can rebuild any
        # internal state (e.g., adjacency indices) that depends on smoothness_weight.
        same_weight = getattr(init, "smoothness_weight", None) == smoothness_weight
        has_adjacency = hasattr(init, "adjacency_indices_")
        if same_weight and has_adjacency:
            bs = init
        else:
            bs = BezierSimplex(init.control_points, smoothness_weight=smoothness_weight)
    elif init is not None:
        bs = BezierSimplex(init, smoothness_weight=smoothness_weight)
    else:
        bs = randn(
            n_params=int(params.shape[1]),
            n_values=int(values.shape[1]),
            degree=cast(int, degree),
            smoothness_weight=smoothness_weight,
        )

    fix = fix or []
    validate_simplex_indices(fix, bs.n_params, bs.degree)

    for index in fix:
        bs.fix_row(index)

    trainer = L.Trainer(**kwargs)
    trainer.fit(bs, dl)
    return bs
