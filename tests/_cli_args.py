"""Shared CLI parsing contracts."""

from argparse import ArgumentTypeError

import pytest

from torch_bsf._cli_args import import_callable, integers, load_table


def test_table_single_column_and_row(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("1\n2\n")
    assert load_table(path).shape == (2, 1)
    path.write_text("1,2\n")
    assert load_table(path).shape == (1, 2)


def test_integers():
    assert integers("2,1") == (2, 1)
    with pytest.raises(ArgumentTypeError):
        integers("-1")


def test_callable():
    assert callable(import_callable("torch:sin"))
    with pytest.raises(ValueError):
        import_callable("torch:__version__")


def test_callable_import_error_is_explanatory():
    with pytest.raises(ValueError, match="cannot import"):
        import_callable("missing_package:not_available")
