"""Whole-pipeline invariants and sharp continuous counterchecks."""

import random

import numpy as np
import pytest
import torch

from torch_bsf.adaptive_curve import fit_adaptive_curve

pytest.importorskip("scipy")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _signal(x):
    jump = (x > 0.4137).double()
    return torch.cat((x + jump, x.square() - 0.3 * jump, 0.2 * x + 0.7 * jump), 1)


def test_vector_pipeline_preserves_output_rotations_and_global_random_states():
    rotation = torch.tensor([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=torch.float64)
    torch_state = torch.get_rng_state().clone()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    assert isinstance(numpy_state, tuple)
    original = fit_adaptive_curve(_signal, (0.0, 1.0), budget=64, location_tolerance=2e-6)
    changed = fit_adaptive_curve(lambda x: _signal(x) @ rotation, (0.0, 1.0), budget=64, location_tolerance=2e-6)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert isinstance(actual_numpy, tuple)
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    torch.testing.assert_close(original.inputs, changed.inputs, rtol=0, atol=1e-12)
    assert len(original.gaps) == len(changed.gaps) == 1
    assert original.gaps[0].left <= 0.4137 <= original.gaps[0].right
    assert original.gaps[0].right - original.gaps[0].left <= 2e-6
    at = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
    torch.testing.assert_close(changed.predict(at), original.predict(at) @ rotation, rtol=2e-9, atol=2e-10, equal_nan=True)


@pytest.mark.parametrize("width", [1e-2, 1e-3])
@pytest.mark.parametrize("outputs", [1, 3])
def test_resolved_sharp_continuous_curves_have_no_final_gap(width, outputs):
    def oracle(x):
        value = torch.tanh((x - 0.4137) / width)
        return value if outputs == 1 else torch.cat((value, -0.3 * value, value + 0.01 * x), 1)

    result = fit_adaptive_curve(oracle, (0.0, 1.0), budget=128)
    assert result.cost == 128
    assert result.gaps == ()
    assert result.domain.betti_numbers()[0] == 1
    assert torch.isfinite(result.predict(torch.linspace(0, 1, 1001)[:, None])).all()
