import pytest

from torch_bsf.validator import index_list, int_or_str


@pytest.mark.parametrize(
    "val, expected",
    (
        ("-1", -1),
        ("1", 1),
        ("+1", 1),
        ("0", 0),
        ("-0", 0),
        ("+0", 0),
        ("1_0", 10),
        ("0_0", 0),
        ("auto", "auto"),
        ("None", "None"),
        ("null", "null"),
        ("-inf", "-inf"),
        ("inf", "inf"),
        ("+inf", "+inf"),
        ("0.0", "0.0"),
        ("1E2", "1E2"),
        ("1e2", "1e2"),
        ("0b1", "0b1"),
        ("0o1", "0o1"),
        ("0x1", "0x1"),
        ("[]", "[]"),
        ("()", "()"),
        ("{}", "{}"),
        ("", ""),
    ),
)
def test_int_or_str(val, expected):
    assert int_or_str(val) == expected


@pytest.mark.parametrize(
    "val, expected",
    (
        ("[1, 0]", [1, 0]),
        ("(1, 0)", [1, 0]),
        ("{1, 0}", [1, 0]),
        ("()", []),
        ("(())", [[]]),
        ("[()]", [[]]),
    ),
)
def test_index_list(val, expected):
    assert index_list(val) == expected
