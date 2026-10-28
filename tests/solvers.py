import numpy as np
import pytest

from torch_bsf.solvers import DenseLeastSquares


@pytest.mark.parametrize("wrap", [np.array, tuple, list, lambda x: np.array(x[0]), lambda x: np.array(x * 2)])
def test_diagnostics_rejects_non_string_choice(wrap):
    value = wrap(["full"])
    with pytest.raises(ValueError, match="diagnostics"):
        DenseLeastSquares(diagnostics=value)


@pytest.mark.parametrize("option", ("basic", "full"))
@pytest.mark.parametrize("string_type", [str, np.str_])
def test_diagnostics_accepts_string_choice(string_type, option):
    value = string_type(option)
    DenseLeastSquares(diagnostics=value)
