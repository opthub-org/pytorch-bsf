"""Tests for torch_bsf._cli_args."""

from argparse import ArgumentParser

from torch_bsf._cli_args import add_common_training_args


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
