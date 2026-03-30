import shutil
from pathlib import Path

import pytest
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

import torch_bsf as tb
import torch_bsf.bezier_simplex as tbbs
from torch_bsf.bezier_simplex import (
    BezierSimplexDataModule,
    monomial,
    polynom,
)

_DATA_DIR = Path(__file__).parent / "data"
_REPO_ROOT = Path(__file__).parent.parent
_PARAMS_CSV = _REPO_ROOT / "params.csv"
_VALUES_CSV = _REPO_ROOT / "values.csv"

@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_zeros(n_params: int, n_values: int, degree: int):
    bs = tbbs.zeros(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_rand(n_params: int, n_values: int, degree: int):
    bs = tbbs.rand(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, n_values, degree",
    (
        (n_params, n_values, degree)
        for n_params in range(3)
        for n_values in range(3)
        for degree in range(3)
    ),
)
def test_randn(n_params: int, n_values: int, degree: int):
    bs = tbbs.randn(n_params, n_values, degree)
    assert bs.n_params == n_params
    assert bs.n_values == n_values
    if n_params == 0:
        assert bs.degree == 0
    else:
        assert bs.degree == degree


@pytest.mark.parametrize(
    "n_params, degree",
    ((n_params, degree) for n_params in range(3) for degree in range(3)),
)
def test_simplex_indices(n_params: int, degree: int):
    indices = list(tbbs.simplex_indices(n_params, degree))
    if n_params <= 1 or degree == 0:
        assert len(indices) == 1
    else:
        assert len(indices) > 1

    if n_params == 0:
        assert indices[0] == ()
        assert indices[-1] == ()
    else:
        assert indices[0] == (degree,) + (0,) * (n_params - 1)
        assert indices[-1] == (0,) * (n_params - 1) + (degree,)


@pytest.mark.parametrize(
    "data",
    (
        {str(index): [0] for index in tbbs.simplex_indices(0, 1)},
        {str(index): [0] for index in tbbs.simplex_indices(1, 2)},
        {str(index): [0] for index in tbbs.simplex_indices(2, 3)},
    ),
)
def test_validate_control_points(data):
    tbbs.validate_control_points(data)


def test_fit():
    ts = torch.tensor(  # parameters on a simplex
        [
            [3 / 3, 0 / 3, 0 / 3],
            [2 / 3, 1 / 3, 0 / 3],
            [2 / 3, 0 / 3, 1 / 3],
            [1 / 3, 2 / 3, 0 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 0 / 3, 2 / 3],
            [0 / 3, 3 / 3, 0 / 3],
            [0 / 3, 2 / 3, 1 / 3],
            [0 / 3, 1 / 3, 2 / 3],
            [0 / 3, 0 / 3, 3 / 3],
        ]
    )
    xs = 1 - ts * ts  # values corresponding to the parameters

    # Train a model
    bs = tb.fit(params=ts, values=xs, degree=3)

    # Predict by the trained model
    t = [[0.2, 0.3, 0.5]]
    x = bs(t)
    print(f"{t} -> {x}")


@pytest.mark.parametrize(
    "init_type",
    ("instance", "rand", "file"),
)
def test_partial_fit(init_type):
    ts = torch.tensor(  # parameters on a simplex
        [
            [8 / 8, 0 / 8],
            [7 / 8, 1 / 8],
            [6 / 8, 2 / 8],
            [5 / 8, 3 / 8],
            [4 / 8, 4 / 8],
            [3 / 8, 5 / 8],
            [2 / 8, 6 / 8],
            [1 / 8, 7 / 8],
            [0 / 8, 8 / 8],
        ]
    )
    xs = 1 - ts * ts  # values corresponding to the parameters

    if init_type == "instance":
        # Initialize 2D control points of a Bezier triangle of degree 3
        init = {
            # index: value
            (3, 0): [0.0, 0.1],
            (2, 1): [1.0, 1.1],
            (1, 2): [2.0, 2.1],
            (0, 3): [3.0, 3.1],
        }
    elif init_type == "rand":
        # Or, generate random control points in [0, 1)
        init = tbbs.rand(n_params=2, n_values=2, degree=3)
    elif init_type == "file":
        # Or, load control points from a file
        init = tbbs.load("control_points.yml")
    else:
        raise ValueError()
    # Train the edge of a Bezier curve while its vertices are fixed
    bs = tbbs.fit(
        params=ts,  # input observations (training data)
        values=xs,  # output observations (training data)
        init=init,  # initial values of control points
        fix=[[3, 0], [0, 3]],  # fix vertices of the Bezier curve
    )

    # Predict by the trained model
    t = [
        [0.2, 0.8],
        [0.7, 0.3],
    ]
    x = bs(t)
    print(f"{t} -> {x}")


def test_smoothness_penalty_returns_zero_tensor_without_adjacency():
    """smoothness_penalty() should return a scalar tensor even when adjacency is absent."""
    bs = tbbs.randn(n_params=2, n_values=2, degree=2, smoothness_weight=0.0)
    penalty = bs.smoothness_penalty()
    assert isinstance(penalty, torch.Tensor)
    assert penalty.shape == ()
    assert float(penalty) == 0.0


def test_smoothness_penalty_nonzero_with_weight():
    """smoothness_penalty() should return a non-negative tensor when smoothness_weight > 0."""
    # Use random control points – penalty is the sum of squared differences between adjacent ones,
    # which is non-negative and typically non-zero for random initialisation.
    bs = tbbs.randn(n_params=3, n_values=2, degree=2, smoothness_weight=0.1)
    assert hasattr(bs, "adjacency_indices_"), "adjacency_indices_ must be built when smoothness_weight > 0"
    penalty = bs.smoothness_penalty()
    assert isinstance(penalty, torch.Tensor)
    assert float(penalty) >= 0.0


def test_smoothness_affects_training_loss():
    """smoothness_penalty() contributes a positive value when smoothness_weight > 0."""
    # Use a random model to ensure control points differ, making penalty non-zero
    bs_with_smooth = tbbs.randn(n_params=3, n_values=2, degree=2, smoothness_weight=1.0)
    assert hasattr(bs_with_smooth, "adjacency_indices_")
    penalty = bs_with_smooth.smoothness_penalty()
    assert isinstance(penalty, torch.Tensor)
    # Penalty must be non-negative (sum of squared differences)
    assert float(penalty) >= 0.0

    # The penalty should be zero if all adjacent control points are identical
    bs_zeros = tbbs.zeros(n_params=3, n_values=2, degree=2, smoothness_weight=1.0)
    assert float(bs_zeros.smoothness_penalty()) == 0.0


def test_fit_with_bezier_simplex_init_rebuilds_adjacency():
    """fit() with init=BezierSimplex should correctly rebuild adjacency for smoothness."""
    ts = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    xs = ts * ts

    # First create a model WITHOUT smoothness
    bs_no_smooth = tb.fit(
        params=ts, values=xs, degree=2,
        smoothness_weight=0.0,
        max_epochs=1, enable_progress_bar=False,
    )
    assert not hasattr(bs_no_smooth, "adjacency_indices_")

    # Now use it as init WITH smoothness – adjacency must be rebuilt
    bs_with_smooth = tb.fit(
        params=ts, values=xs,
        init=bs_no_smooth,
        smoothness_weight=0.5,
        max_epochs=1, enable_progress_bar=False,
    )
    assert hasattr(bs_with_smooth, "adjacency_indices_"), (
        "adjacency_indices_ should be built when smoothness_weight > 0"
    )


def test_meshgrid_n_params_zero():
    """meshgrid() should work for n_params == 0 (constant simplex)."""
    bs = tbbs.zeros(n_params=0, n_values=2, degree=0)
    ts, xs = bs.meshgrid(num=5)
    # For n_params == 0, the simplex has a single trivial point
    assert xs.shape[1] == 2


def test_forward_vectorized():
    """Vectorized forward should produce the same result as the scalar reference."""
    ts = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    bs = tbbs.randn(n_params=2, n_values=2, degree=3)
    # Forward should accept a list-of-lists as well as a Tensor
    xs_tensor = bs(ts)
    xs_list = bs([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    assert xs_tensor.shape == (3, 2)
    assert torch.allclose(xs_tensor, xs_list, atol=1e-6)


@pytest.mark.parametrize("ext", [".csv", ".CSV", ".Csv"])
def test_load_case_insensitive_csv(tmp_path, ext):
    dest = tmp_path / f"model{ext}"
    shutil.copy(_DATA_DIR / "bezier_simplex.csv", dest)
    bs = tbbs.load(str(dest))
    assert isinstance(bs, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".json", ".JSON", ".Json"])
def test_load_case_insensitive_json(tmp_path, ext):
    dest = tmp_path / f"model{ext}"
    shutil.copy(_DATA_DIR / "bezier_simplex.json", dest)
    bs = tbbs.load(str(dest))
    assert isinstance(bs, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".yml", ".YML", ".yaml", ".YAML"])
def test_load_case_insensitive_yaml(tmp_path, ext):
    dest = tmp_path / f"model{ext}"
    shutil.copy(_DATA_DIR / "bezier_simplex.yml", dest)
    bs = tbbs.load(str(dest))
    assert isinstance(bs, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".csv", ".CSV", ".Csv"])
def test_save_case_insensitive_csv(tmp_path, ext):
    bs = tbbs.load(str(_DATA_DIR / "bezier_simplex.csv"))
    dest = tmp_path / f"model{ext}"
    tbbs.save(str(dest), bs)
    assert dest.exists()
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".json", ".JSON", ".Json"])
def test_save_case_insensitive_json(tmp_path, ext):
    bs = tbbs.load(str(_DATA_DIR / "bezier_simplex.json"))
    dest = tmp_path / f"model{ext}"
    tbbs.save(str(dest), bs)
    assert dest.exists()
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".yml", ".YML", ".Yml", ".yaml", ".YAML", ".Yaml"])
def test_save_case_insensitive_yaml(tmp_path, ext):
    bs = tbbs.load(str(_DATA_DIR / "bezier_simplex.yml"))
    dest = tmp_path / f"model{ext}"
    tbbs.save(str(dest), bs)
    assert dest.exists()
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".tsv", ".TSV", ".Tsv"])
def test_load_case_insensitive_tsv(tmp_path, ext):
    dest = tmp_path / f"model{ext}"
    shutil.copy(_DATA_DIR / "bezier_simplex.tsv", dest)
    bs = tbbs.load(str(dest))
    assert isinstance(bs, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".tsv", ".TSV", ".Tsv"])
def test_save_case_insensitive_tsv(tmp_path, ext):
    bs = tbbs.load(str(_DATA_DIR / "bezier_simplex.tsv"))
    dest = tmp_path / f"model{ext}"
    tbbs.save(str(dest), bs)
    assert dest.exists()
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".pt", ".PT", ".Pt"])
def test_load_case_insensitive_pt(tmp_path, ext):
    # Create a real .pt file via save() to avoid relying on a committed fixture
    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    src = tmp_path / "source_model.pt"
    tbbs.save(str(src), bs)
    dest = tmp_path / f"model{ext}"
    shutil.copy(src, dest)
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("ext", [".pt", ".PT", ".Pt"])
def test_save_case_insensitive_pt(tmp_path, ext):
    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    dest = tmp_path / f"model{ext}"
    tbbs.save(str(dest), bs)
    assert dest.exists()
    bs2 = tbbs.load(str(dest))
    assert isinstance(bs2, tbbs.BezierSimplex)


@pytest.mark.parametrize("content,ext", [
    ("", ".csv"),
    ("\n\n", ".csv"),
    ("", ".tsv"),
    ("\n\n", ".tsv"),
])
def test_load_empty_csv_tsv_raises(tmp_path, content, ext):
    f = tmp_path / f"empty{ext}"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="No control points found"):
        tbbs.load(str(f))


def test_load_empty_json_raises(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No control points found"):
        tbbs.load(str(f))


def test_load_non_dict_json_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        tbbs.load(str(f))


def test_load_empty_yaml_raises(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No control points found"):
        tbbs.load(str(f))


def test_load_null_yaml_raises(tmp_path):
    f = tmp_path / "null.yaml"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        tbbs.load(str(f))


def test_load_non_dict_yaml_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        tbbs.load(str(f))


def test_val_avg_mse_logged_at_epoch_end():
    """val_avg_mse must appear in callback_metrics after validation under Lightning v2."""
    ts = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    xs = 1 - ts * ts

    train_dl = DataLoader(TensorDataset(ts, xs), batch_size=len(ts))
    val_dl = DataLoader(TensorDataset(ts, xs), batch_size=len(ts))

    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(bs, train_dl, val_dl)

    assert "val_avg_mse" in trainer.callback_metrics, (
        "val_avg_mse should be logged at epoch end and available in callback_metrics"
    )


def test_early_stopping_monitors_val_avg_mse():
    """EarlyStopping(monitor='val_avg_mse') must not raise under Lightning v2."""
    ts = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    xs = 1 - ts * ts

    train_dl = DataLoader(TensorDataset(ts, xs), batch_size=len(ts))
    val_dl = DataLoader(TensorDataset(ts, xs), batch_size=len(ts))

    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val_avg_mse", patience=2)],
    )
    # Should complete without MisconfigurationException about missing monitor key
    trainer.fit(bs, train_dl, val_dl)
    assert "val_avg_mse" in trainer.callback_metrics


# ---------------------------------------------------------------------------
# BezierSimplex.__init__ — optional control_points / checkpoint-reconstruction
# ---------------------------------------------------------------------------


def test_init_no_args_raises():
    """BezierSimplex() with no arguments must raise TypeError."""
    with pytest.raises(TypeError):
        tbbs.BezierSimplex()


def test_init_partial_shape_params_raises():
    """Passing control_points=None with only some shape params must raise TypeError."""
    with pytest.raises(TypeError):
        tbbs.BezierSimplex(control_points=None, _n_params=2)


def test_init_with_all_shape_params_creates_placeholder():
    """Passing control_points=None with all shape params creates a valid model."""
    bs = tbbs.BezierSimplex(control_points=None, _n_params=2, _degree=2, _n_values=3)
    assert bs.n_params == 2
    assert bs.degree == 2
    assert bs.n_values == 3


# ---------------------------------------------------------------------------
# Lightning checkpoint round-trip
# ---------------------------------------------------------------------------


def test_load_from_checkpoint_round_trip(tmp_path):
    """BezierSimplex.load_from_checkpoint must restore a usable model."""
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = 1.0 - ts * ts
    dl = DataLoader(TensorDataset(ts, xs), batch_size=3)

    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    ckpt_path = tmp_path / "model.ckpt"

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(bs, dl)
    trainer.save_checkpoint(ckpt_path)

    bs2 = tbbs.BezierSimplex.load_from_checkpoint(str(ckpt_path))
    assert bs2.n_params == bs.n_params
    assert bs2.degree == bs.degree
    assert bs2.n_values == bs.n_values
    # Predictions should match
    with torch.no_grad():
        out1 = bs(ts)
        out2 = bs2(ts)
    assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# polynom / monomial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "degree, index, expected",
    [
        (3, (3,), 1.0),           # 3! / 3! = 1
        (3, (2, 1), 3.0),         # 3! / (2! 1!) = 3
        (3, (1, 1, 1), 6.0),      # 3! / (1! 1! 1!) = 6
        (0, (0,), 1.0),           # 0! / 0! = 1
    ],
)
def test_polynom(degree, index, expected):
    assert polynom(degree, index) == pytest.approx(expected)


@pytest.mark.parametrize(
    "variable, degree, expected",
    [
        ([2.0, 3.0], [1, 1], 6.0),   # 2^1 * 3^1 = 6
        ([2.0, 3.0], [2, 0], 4.0),   # 2^2 * 3^0 = 4
        ([0.5, 0.5], [1, 1], 0.25),  # 0.5 * 0.5 = 0.25
        ([1.0], [0], 1.0),           # scalar identity
    ],
)
def test_monomial(variable, degree, expected):
    result = monomial(variable, degree)
    # monomial() returns a per-sample tensor; squeeze for scalar comparison
    assert float(result.squeeze()) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# zeros / rand / randn negative-argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [tbbs.zeros, tbbs.rand, tbbs.randn])
def test_factory_negative_n_params_raises(fn):
    with pytest.raises(ValueError, match="non-negative"):
        fn(n_params=-1, n_values=2, degree=2)


@pytest.mark.parametrize("fn", [tbbs.zeros, tbbs.rand, tbbs.randn])
def test_factory_negative_n_values_raises(fn):
    with pytest.raises(ValueError, match="non-negative"):
        fn(n_params=2, n_values=-1, degree=2)


@pytest.mark.parametrize("fn", [tbbs.zeros, tbbs.rand, tbbs.randn])
def test_factory_negative_degree_raises(fn):
    with pytest.raises(ValueError, match="non-negative"):
        fn(n_params=2, n_values=2, degree=-1)


# ---------------------------------------------------------------------------
# save / load unknown extension
# ---------------------------------------------------------------------------


def test_save_unknown_extension_raises(tmp_path):
    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    with pytest.raises(ValueError, match="Unknown file type"):
        tbbs.save(str(tmp_path / "model.xyz"), bs)


def test_load_unknown_extension_raises(tmp_path):
    f = tmp_path / "model.xyz"
    f.write_text("dummy")
    with pytest.raises(ValueError, match="Unknown file type"):
        tbbs.load(str(f))


# ---------------------------------------------------------------------------
# fit — error paths and alternative init types
# ---------------------------------------------------------------------------


def test_fit_no_degree_no_init_raises():
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = ts * ts
    with pytest.raises(ValueError, match="Either degree or init must be specified"):
        tbbs.fit(params=ts, values=xs, max_epochs=1, enable_progress_bar=False, logger=False)


def test_fit_both_degree_and_init_raises():
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = ts * ts
    init = tbbs.rand(n_params=2, n_values=2, degree=1)
    with pytest.raises(ValueError, match="Either degree or init must be specified"):
        tbbs.fit(params=ts, values=xs, degree=1, init=init,
                 max_epochs=1, enable_progress_bar=False, logger=False)


def test_fit_with_dict_init():
    """fit() accepts a plain dict of control points as init."""
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = ts * ts
    from torch_bsf.control_points import simplex_indices

    init = {idx: [0.5, 0.5] for idx in simplex_indices(2, 1)}
    bs = tbbs.fit(params=ts, values=xs, init=init,
                  max_epochs=1, enable_progress_bar=False, logger=False)
    assert bs.n_params == 2
    assert bs.n_values == 2


# ---------------------------------------------------------------------------
# fix_row / on_after_backward
# ---------------------------------------------------------------------------


def test_fix_row_zeros_gradient():
    """fix_row() must zero out the gradient for the fixed control point after backward."""
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = ts * ts
    bs = tbbs.randn(n_params=2, n_values=2, degree=1)

    # Fix the first vertex
    first_index = (1, 0)
    bs.fix_row(first_index)

    # Manual forward + backward
    pred = bs(ts)
    loss = torch.nn.functional.mse_loss(pred, xs)
    loss.backward()
    bs.on_after_backward()

    # The gradient of the fixed row must be zero
    grad = bs.control_points.matrix.grad
    assert grad is not None
    row = bs.control_points._index_to_row[first_index]
    assert torch.all(grad[row] == 0.0)


# ---------------------------------------------------------------------------
# forward — constant simplex (n_params == 0)
# ---------------------------------------------------------------------------


def test_forward_constant_simplex_broadcasts():
    """forward() for n_params == 0 must broadcast the single control point over batch."""
    bs = tbbs.zeros(n_params=0, n_values=3, degree=0)
    # Constant simplex: the single control-point value is [0, 0, 0]
    # We feed an arbitrary batch tensor of shape (5, 0)
    t = torch.zeros(5, 0)
    out = bs(t)
    assert out.shape == (5, 3)
    assert torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# meshgrid with n_params > 0
# ---------------------------------------------------------------------------


def test_meshgrid_returns_correct_shapes():
    """meshgrid() must return paired (ts, xs) tensors with matching batch dimensions."""
    bs = tbbs.randn(n_params=2, n_values=3, degree=2)
    ts, xs = bs.meshgrid(num=5)
    assert ts.ndim == 2
    assert ts.shape[1] == 2
    assert xs.ndim == 2
    assert xs.shape[1] == 3
    assert ts.shape[0] == xs.shape[0]


# ---------------------------------------------------------------------------
# test_step logs metrics
# ---------------------------------------------------------------------------


def test_test_step_logs_metrics():
    """test_step() must log test_mse and test_mae."""
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = 1.0 - ts * ts
    dl = DataLoader(TensorDataset(ts, xs), batch_size=3)

    bs = tbbs.randn(n_params=2, n_values=2, degree=1)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(bs, dl)
    trainer.test(bs, dl, verbose=False)

    assert "test_mse" in trainer.callback_metrics
    assert "test_mae" in trainer.callback_metrics


# ---------------------------------------------------------------------------
# BezierSimplexDataModule
# ---------------------------------------------------------------------------


def test_data_module_basic():
    """BezierSimplexDataModule must infer n_params and n_values from CSV headers."""
    dm = BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV)
    assert dm.n_params == 2
    assert dm.n_values == 2


def test_data_module_train_dataloader():
    """train_dataloader() must return batches of (params, values) tensors."""
    dm = BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV)
    dl = dm.train_dataloader()
    params_batch, values_batch = next(iter(dl))
    assert params_batch.ndim == 2
    assert params_batch.shape[1] == 2
    assert values_batch.ndim == 2
    assert values_batch.shape[1] == 2


def test_data_module_val_and_test_dataloader():
    """val_dataloader() and test_dataloader() must return valid loaders."""
    dm = BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, split_ratio=0.8)
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))
    assert len(val_batch) == 2
    assert len(test_batch) == 2


def test_data_module_batch_size():
    """Setting batch_size must limit the minibatch size."""
    dm = BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, batch_size=2)
    params_batch, _ = next(iter(dm.train_dataloader()))
    assert params_batch.shape[0] <= 2


@pytest.mark.parametrize("normalize", ["max", "std", "quantile", "none"])
def test_data_module_normalize_modes(normalize):
    """All normalize modes must produce valid float tensors."""
    dm = BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, normalize=normalize)
    _, values_batch = next(iter(dm.train_dataloader()))
    assert values_batch.dtype in (torch.float32, torch.float64)


def test_data_module_invalid_header_raises():
    with pytest.raises(ValueError, match="non-negative"):
        BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, header=-1)


def test_data_module_invalid_batch_size_raises():
    with pytest.raises(ValueError, match="positive"):
        BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, batch_size=0)


def test_data_module_invalid_split_ratio_raises():
    with pytest.raises(ValueError, match="split_ratio"):
        BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, split_ratio=0.0)


def test_data_module_invalid_normalize_raises():
    with pytest.raises(ValueError, match="normalize"):
        BezierSimplexDataModule(params=_PARAMS_CSV, values=_VALUES_CSV, normalize="unknown")

