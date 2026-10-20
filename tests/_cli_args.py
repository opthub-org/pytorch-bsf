"""Tests for torch_bsf._cli_args."""

from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import pytest

from torch_bsf._cli_args import add_common_training_args, add_model_args, fit_options, load_model_data, simploid_type


def test_add_common_training_args_registers_all():
    """add_common_training_args should register all expected arguments."""
    parser = ArgumentParser()
    add_common_training_args(parser)

    args = parser.parse_args([])
    assert args.split_ratio == 1.0
    assert args.batch_size is None
    assert args.max_epochs == 2
    assert args.smoothness_weight == 0.0
    assert args.accelerator == "auto"
    assert args.strategy == "auto"
    assert args.devices == "auto"
    assert args.num_nodes == 1
    assert args.precision == "32-true"
    assert args.enable_checkpointing is False
    assert args.log_every_n_steps == 1


def test_add_common_training_args_overrides():
    """Passing values overrides defaults."""
    parser = ArgumentParser()
    add_common_training_args(parser)

    args = parser.parse_args(
        [
            "--split_ratio=0.8",
            "--batch_size=32",
            "--max_epochs=100",
            "--smoothness_weight=0.5",
            "--accelerator=cpu",
            "--strategy=ddp",
            "--devices=2",
            "--num_nodes=2",
            "--precision=16-mixed",
            "--enable_checkpointing",
            "--log_every_n_steps=10",
        ]
    )
    assert args.split_ratio == 0.8
    assert args.batch_size == 32
    assert args.max_epochs == 100
    assert args.smoothness_weight == 0.5
    assert args.accelerator == "cpu"
    assert args.strategy == "ddp"
    assert args.devices == 2
    assert args.num_nodes == 2
    assert args.precision == "16-mixed"
    assert args.enable_checkpointing is True
    assert args.log_every_n_steps == 10


@pytest.mark.parametrize(
    "text, expected",
    [("2,1", (2, 1)), ("1", (1,)), (" 1, 1, 1 ", (1, 1, 1)), ("(2, 1)", (2, 1)), ("[3,2]", (3, 2))],
)
def test_simploid_type_parses(text, expected):
    """simploid_type should parse comma-separated positive integers."""
    assert simploid_type(text) == expected


@pytest.mark.parametrize("text", ["", "a", "2,", "2,0", "-1", "1.5", "()"])
def test_simploid_type_rejects_invalid(text):
    """simploid_type should reject anything but a non-empty list of positive integers."""
    with pytest.raises(ArgumentTypeError, match="invalid simploid type"):
        simploid_type(text)


def _parse(*argv: str):
    parser = ArgumentParser()
    add_model_args(parser)
    add_common_training_args(parser)
    return parser, parser.parse_args(["--params=p.csv", "--values=v.csv", *argv])


def test_add_model_args():
    """add_model_args should register the data, model and solver arguments."""
    _, args = _parse("--degree=3")
    assert args.degree == 3
    assert args.init is None
    assert args.simploid_type is None
    assert args.freeze is None
    assert args.normalize == "none"
    assert args.solver == "lstsq"
    assert args.ridge == 0.0
    assert args.seed is None

    _, args = _parse("--degree=2", "--simploid-type=2,1", "--solver=trainer", "--normalize=max", "--seed=1")
    assert args.simploid_type == (2, 1)
    assert args.solver == "trainer"
    assert args.normalize == "max"
    assert args.seed == 1
    assert _parse("--degree=1", "--simploid_type=1,1")[1].simploid_type == (1, 1)
    assert _parse("--degree=1", "--simploid_type=")[1].simploid_type is None


@pytest.mark.parametrize(
    "argv",
    [
        [],  # neither --degree nor --init
        ["--degree=1", "--init=m.json"],
        ["--degree=1", "--solver=adam"],
    ],
)
def test_add_model_args_rejects(argv):
    with pytest.raises(SystemExit):
        _parse(*argv)


def _write(tmp_path, params, values):
    np.savetxt(tmp_path / "p.csv", params, delimiter=",")
    np.savetxt(tmp_path / "v.csv", values, delimiter=",")


def _load(tmp_path, *argv: str):
    parser = ArgumentParser()
    add_model_args(parser)
    add_common_training_args(parser)
    args = parser.parse_args([f"--params={tmp_path / 'p.csv'}", f"--values={tmp_path / 'v.csv'}", *argv])
    return args, load_model_data(parser, args)


def test_load_model_data_infers_a_simplex(tmp_path):
    _write(tmp_path, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0], [2.0]])
    _, (params, values, st, init) = _load(tmp_path, "--degree=1")
    assert params.shape == (2, 3)
    assert values.shape == (2, 1)
    assert st == (2,)
    assert init is None


def test_load_model_data_takes_the_type_from_init(tmp_path):
    from torch_bsf.simploidal import BezierSimploid, save

    _write(tmp_path, [[1.0, 0.0, 1.0, 0.0]], [[1.0]])
    save(tmp_path / "m.pt", BezierSimploid.zeros((1, 1), 1, n_values=1))
    args, (_, _, st, init) = _load(tmp_path, f"--init={tmp_path / 'm.pt'}", "--freeze=[[[1,0],[1,0]]]")
    assert st == (1, 1)
    assert init is not None and init.simploid_type == (1, 1)
    options = fit_options(args, st, init)
    assert options["init"] is init
    assert options["freeze"] == [[[1, 0], [1, 0]]]
    assert "max_epochs" not in options  # trainer options apply to the trainer solver only


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--degree=-1"], "--degree must be non-negative"),
        (["--degree=1", "--ridge=-1"], "--ridge must be non-negative"),
        (["--degree=1", "--ridge=1", "--solver=trainer"], "--ridge is supported only"),
        (["--degree=1", "--freeze=[[1,0,0]]"], "--freeze requires --init"),
        (["--degree=1", "--simploid-type=1,1"], "requires sum"),
        (["--init=missing.pt"], "cannot load --init"),
    ],
)
def test_load_model_data_rejects(tmp_path, capsys, argv, message):
    _write(tmp_path, [[1.0, 0.0, 0.0]], [[1.0]])
    with pytest.raises(SystemExit):
        _load(tmp_path, *argv)
    assert message in capsys.readouterr().err


def test_load_model_data_rejects_row_mismatch_and_one_column(tmp_path, capsys):
    _write(tmp_path, [[1.0, 0.0], [0.0, 1.0]], [[1.0]])
    with pytest.raises(SystemExit):
        _load(tmp_path, "--degree=1")
    assert "rows" in capsys.readouterr().err
    _write(tmp_path, [[1.0], [1.0]], [[1.0], [2.0]])
    with pytest.raises(SystemExit):
        _load(tmp_path, "--degree=1")
    assert "needs at least 2" in capsys.readouterr().err


def test_fit_options_passes_trainer_options_to_the_trainer_solver(tmp_path):
    _write(tmp_path, [[1.0, 0.0]], [[1.0]])
    args, (_, _, st, init) = _load(tmp_path, "--degree=1", "--solver=trainer", "--max_epochs=7", "--batch_size=4")
    options = fit_options(args, st, init)
    assert options["solver"] == "trainer"
    assert options["max_epochs"] == 7
    assert options["batch_size"] == 4
    assert options["simploid_type"] == (1,)
