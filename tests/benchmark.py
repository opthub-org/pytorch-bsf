"""Basic benchmarks for torch_bsf.

Run with::

    pytest tests/benchmark.py --benchmark-only

These benchmarks are intentionally skipped in the normal ``pytest`` run so
they do not slow down the unit-test suite.  They are useful for detecting
performance regressions in the forward pass and in :func:`torch_bsf.fit`.
"""

import sys
from typing import Any

import pytest
import torch

pytest.importorskip("pytest_benchmark")

if "--benchmark-only" not in sys.argv:
    pytest.skip(
        "Benchmarks are only run when --benchmark-only is specified.",
        allow_module_level=True,
    )
import torch_bsf  # noqa: E402
from tests import requires_lightning
from torch_bsf import BezierSimploid  # noqa: E402
from torch_bsf.sampling import simplex_grid  # noqa: E402

# ---------------------------------------------------------------------------
# Forward-pass benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="forward")
@pytest.mark.parametrize(
    "n_params, n_values, degree, batch_size",
    [
        (3, 3, 3, 100),
        (3, 3, 5, 100),
        (4, 4, 3, 100),
        (3, 3, 3, 1000),
    ],
)
def test_benchmark_forward(n_params, n_values, degree, batch_size, benchmark):
    """Benchmark the forward pass of a Bézier simplex."""
    bs = BezierSimploid.zeros((n_params - 1,), degree, n_values)
    pts = simplex_grid(n_params=n_params, degree=degree)[:batch_size]
    if len(pts) < batch_size:
        # Repeat to reach batch_size
        repeats = (batch_size + len(pts) - 1) // len(pts)
        pts = pts.repeat(repeats, 1)[:batch_size]

    def _forward():
        with torch.no_grad():
            return bs(pts)

    benchmark(_forward)


# ---------------------------------------------------------------------------
# fit() benchmark
# ---------------------------------------------------------------------------


@requires_lightning
@pytest.mark.benchmark(group="fit")
def test_benchmark_fit(benchmark):
    """Benchmark a short training run of torch_bsf.fit(solver="trainer")."""
    params = simplex_grid(n_params=3, degree=3)
    values = params.pow(2).sum(dim=1, keepdim=True)

    fast: dict[str, Any] = dict(
        solver="trainer",
        max_epochs=5,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )

    def _fit():
        return torch_bsf.fit(params=params, values=values, degree=2, **fast)

    benchmark(_fit)


@pytest.mark.benchmark(group="fit")
def test_benchmark_fit_lstsq(benchmark):
    """Benchmark the exact least-squares solver of torch_bsf.fit()."""
    params = simplex_grid(n_params=3, degree=10)
    values = params.pow(2).sum(dim=1, keepdim=True)
    benchmark(lambda: torch_bsf.fit(params=params, values=values, degree=4))
