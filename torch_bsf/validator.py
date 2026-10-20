"""Parsers of command-line arguments."""

import json
from typing import cast


def index_list(val: str) -> list[list[int]]:
    r"""Parse ``val`` into a list of indices.

    Parameters
    ----------
    val
        A string expression of a list of indices.

    Returns
    -------
        The persed indices.
    """
    val = val.replace("(", "[").replace(")", "]").replace("{", "[").replace("}", "]")
    indices = cast(list[list[int]], json.loads(val))
    return indices


def int_or_str(val: str) -> int | str:
    r"""Try to parse int.
    Return the int value if the parse is succeeded; the original string otherwise.

    Parameters
    ----------
    val
        The value to try to convert into int.

    Returns
    -------
    typing.Union[int, str]
        The converted integer or the original value.
    """
    try:
        return int(val)
    except ValueError:
        return val
