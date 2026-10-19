"""Tests for torch_bsf._cli_args."""

from argparse import ArgumentParser, ArgumentTypeError

import pytest

from torch_bsf._cli_args import add_common_training_args, add_simploid_args, simploid_type


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


def test_add_simploid_args():
    """add_simploid_args should register --simploid-type (with an underscore alias), --solver and --ridge."""
    parser = ArgumentParser()
    add_simploid_args(parser)

    args = parser.parse_args([])
    assert args.simploid_type is None
    assert args.solver is None
    assert args.ridge is None

    args = parser.parse_args(["--simploid-type=2,1", "--solver=trainer", "--ridge=0.5"])
    assert args.simploid_type == (2, 1)
    assert args.solver == "trainer"
    assert args.ridge == 0.5
    assert parser.parse_args(["--simploid_type=1,1"]).simploid_type == (1, 1)
    with pytest.raises(SystemExit):
        parser.parse_args(["--solver=adam"])
