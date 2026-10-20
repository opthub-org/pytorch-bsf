"""Comparison of Parameter Distribution and Value Scaling for Bézier Simplex Fitting.

This script demonstrates two key concepts in Bézier Simplex Fitting (BSF):
1. How parameter distribution (uniform vs skewed) affects fitting accuracy.
2. How target value scaling affects the balance between multiple objectives.
"""

import logging
import warnings

import torch

import torch_bsf
from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid
from torch_bsf.sampling import simplex_grid

# Suppress progress bars and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def f_parameter_experiment(t: torch.Tensor) -> torch.Tensor:
    """A smooth target function for parameter distribution experiment."""
    # A simple smooth function that a degree 3 Bézier simplex can fit well
    return torch.sin(3 * t[:, 0])


def f_value_experiment(t: torch.Tensor) -> torch.Tensor:
    """A multi-objective target function with different scales."""
    # y1: small scale (approx [0, 1]), high difficulty
    y1 = torch.sin(10 * t[:, 0])
    # y2: large scale (approx [0, 1000]), low difficulty (linear-like)
    y2 = 1000 * (t[:, 1] + t[:, 2])
    return torch.stack([y1, y2], dim=1)


def run_parameter_experiment():
    print("--- 1. Parameter Distribution Experiment ---")
    torch.manual_seed(42)

    # A: Uniformly distributed parameters (Triangular grid, degree 9 -> 55 samples)
    ts_uni = simplex_grid(n_params=3, degree=9)
    xs_uni = f_parameter_experiment(ts_uni).unsqueeze(1)

    # B: Skewed parameters (Log-grid style, 55 samples, base 1000)
    ts_skew = torch.tensor(elastic_net_grid(n_lambdas=10, n_alphas=6, base=1000), dtype=torch.float32)
    xs_skew = f_parameter_experiment(ts_skew).unsqueeze(1)

    # Fit using high epochs to ensure convergence
    bs_uni = torch_bsf.fit(
        params=ts_uni,
        values=xs_uni,
        degree=3,
        solver="trainer",
        max_epochs=200,
        enable_progress_bar=False,
        accelerator="cpu",
        enable_checkpointing=False,
    )
    bs_skew = torch_bsf.fit(
        params=ts_skew,
        values=xs_skew,
        degree=3,
        solver="trainer",
        max_epochs=200,
        enable_progress_bar=False,
        accelerator="cpu",
        enable_checkpointing=False,
    )

    # Evaluate on a fine uniform test set
    ts_test = torch.distributions.Dirichlet(torch.ones(3)).sample((2000,))
    xs_test = f_parameter_experiment(ts_test).unsqueeze(1)

    mse_uni = torch.mean((bs_uni(ts_test) - xs_test) ** 2).item()
    mse_skew = torch.mean((bs_skew(ts_test) - xs_test) ** 2).item()

    print(f"MSE (from Uniform samples): {mse_uni:2.2e}")
    print(f"MSE (from Skewed samples) : {mse_skew:2.2e}")
    print("Bezier simplices perform best with uniform parameter coverage.")


def run_value_experiment():
    print("\n--- 2. Value Scaling Experiment ---")
    torch.manual_seed(42)

    ts = torch.distributions.Dirichlet(torch.ones(3)).sample((400,))
    xs = f_value_experiment(ts)

    # The gradient-based trainer is sensitive to the scale of values; the exact least-squares
    # solver (the default of torch_bsf.fit) fits each value independently and is not.
    # Case A: Fit on raw values (optimizer will naturally prioritize the larger-scale y2)
    bs_raw = torch_bsf.fit(
        params=ts,
        values=xs,
        degree=2,
        solver="trainer",
        max_epochs=200,
        enable_progress_bar=False,
        accelerator="cpu",
        enable_checkpointing=False,
    )

    # Case B: Fit on normalized (standardized) values; the fitted model predicts in the original scale
    bs_norm = torch_bsf.fit(
        params=ts,
        values=xs,
        degree=2,
        normalize="std",
        solver="trainer",
        max_epochs=200,
        enable_progress_bar=False,
        accelerator="cpu",
        enable_checkpointing=False,
    )

    # Evaluate on original scale
    ts_test = torch.distributions.Dirichlet(torch.ones(3)).sample((1000,))
    xs_test = f_value_experiment(ts_test)

    # Raw fit result
    pred_raw = bs_raw(ts_test)
    mse_raw = torch.mean((pred_raw - xs_test) ** 2, dim=0)

    # Scaled fit result
    pred_norm = bs_norm(ts_test)
    mse_norm = torch.mean((pred_norm - xs_test) ** 2, dim=0)

    print(f"Raw Fit   MSE (y1, y2): {mse_raw[0].item():2.2e}, {mse_raw[1].item():2.2e}")
    print(f"Scale Fit MSE (y1, y2): {mse_norm[0].item():2.2e}, {mse_norm[1].item():2.2e}")
    print("Scaling balances the contribution of axes with different magnitudes.")


if __name__ == "__main__":
    run_parameter_experiment()
    run_value_experiment()
