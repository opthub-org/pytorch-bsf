"""Same empirical operator conversion, initially partitioned at known supports.

Only the initial partition changes. Degree 12, tolerance, required points,
endpoint pinning, physical checks, maximum 1024 pieces and 16 further subdivision
levels per seed interval are unchanged. For a Matern posterior these supports
are the known absolute-radius kernel breakpoints, not inferred true gaps.
Scalar conversion retains the original converter.
"""

import math
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from .component_kernel import ComponentKernelModel
from .polynomial import _Spline
from .shared_spline import restrict_spline
from .spectral import _bernstein_controls


class SmoothOperator(Protocol):
    """A fitted scalar-coordinate vector operator in normalized [-1,1] units."""

    support: NDArray[np.float64]

    def predict(self, at: NDArray[np.float64]) -> NDArray[np.float64]: ...


def component_splines(model: ComponentKernelModel) -> list[_Spline]:
    """Use the separate support partition only for vector Matérn posteriors.

    Other fitted operators retain the frozen conversion path. This conversion
    profile changes the reachable subdivision geometry: depth16 is counted per
    support interval rather than from the whole domain. A prior rejection can
    therefore become a success; numerical/statistical equivalence is not assumed.
    """
    if model.values.shape[1] == 1 or model._smooth.kind not in ("matern32", "matern52"):
        return model.as_splines()
    smooth = approximate_operator(model.inputs, model.values, model._smooth, model._smooth_predict, model.diagnostics)
    models = []
    maximum = 0.0
    for i, group in enumerate(model.groups):
        left, right = float(model.inputs[group[0], 0]), float(model.inputs[group[-1], 0])
        chart = restrict_spline(smooth, left, right)
        chart.center += torch.from_numpy(model.offsets[i])
        at = torch.unique(torch.cat((model.inputs[group, 0], torch.linspace(left, right, 129, dtype=torch.float64))))
        error = float((chart.predict(at) - model.predict(at[:, None])).norm(dim=1).max())
        maximum = max(maximum, error)
        models.append(chart)
    if not math.isfinite(maximum) or maximum > model.diagnostics["conversion_checked_tolerance"]:
        raise ArithmeticError("Support-partition component Bernstein model failed its checked tolerance.")
    model.diagnostics.update(
        restricted_conversion_checked_max_error=maximum,
        chart_polynomial_pieces=[len(chart.knots) - 1 for chart in models],
        conversion_profile="support_partition_vector_matern",
        conversion_rejection_policy_changed=True,
    )
    return models


def approximate_operator(
    inputs: torch.Tensor,
    values: torch.Tensor,
    operator: SmoothOperator,
    physical_predict: Callable[[Any], torch.Tensor],
    diagnostics: dict[str, Any],
) -> _Spline:
    """Compile fitted-function samples to empirically checked Bernstein pieces."""
    degree = 12
    nodes = np.cos(np.pi * np.arange(degree + 1) / degree)
    checks = np.cos(np.pi * (np.arange(2 * degree + 3) + 0.5) / (2 * degree + 3))
    response = values.numpy()
    amplitude = float(np.linalg.norm(response - response.mean(axis=0), axis=1).max())
    magnitude = float(np.linalg.norm(response, axis=1).max())
    tolerance = max(amplitude * 1e-10, magnitude * float(np.finfo(np.float64).eps) * 2048, float(np.finfo(np.float64).tiny))
    support = np.asarray(operator.support, dtype=np.float64)
    if support.ndim != 1 or not np.isfinite(support).all() or (np.abs(support) > 1).any():
        raise ValueError("Known support breakpoints must be finite and inside normalized bounds.")
    partition = np.unique(np.r_[-1.0, support, 1.0])
    pending = [(float(left), float(right), 0) for left, right in zip(partition[:-1], partition[1:], strict=True)][::-1]
    pieces = []
    maximum_error = 0.0
    physical_left, physical_right = float(inputs[0, 0]), float(inputs[-1, 0])
    observed = (inputs[:, 0].numpy() - physical_left) / (physical_right - physical_left) * 2 - 1
    required = np.unique(np.r_[observed, operator.support])
    validation_points = []
    while pending:
        left, right, depth = pending.pop()
        z = (left + right) / 2 + (right - left) / 2 * nodes
        node_values = operator.predict(z)
        coefficients = np.polynomial.chebyshev.chebfit(nodes, node_values, degree)
        at = (left + right) / 2 + (right - left) / 2 * checks
        at = np.unique(np.r_[left, right, at, required[(required >= left) & (required <= right)]])
        controls = _bernstein_controls(torch.from_numpy(coefficients), 1)
        controls[0] = torch.from_numpy(operator.predict(np.array([left]))[0])
        controls[-1] = torch.from_numpy(operator.predict(np.array([right]))[0])
        t = (at - left) / (right - left)
        design = np.column_stack([math.comb(degree, j) * t**j * (1 - t) ** (degree - j) for j in range(degree + 1)])
        approximation = design @ controls.numpy()
        error = float(np.linalg.norm(approximation - operator.predict(at), axis=1).max())
        if not math.isfinite(error):
            raise ArithmeticError("Operator-to-polynomial conversion produced nonfinite errors.")
        if error > tolerance:
            if depth >= 16:
                raise ArithmeticError("Operator-to-polynomial conversion did not meet its checked tolerance.")
            middle = left / 2 + right / 2
            pending.extend(((middle, right, depth + 1), (left, middle, depth + 1)))
            continue
        maximum_error = max(maximum_error, error)
        # Pin shared endpoints to one evaluation of the same operator;
        # rounding of separate fits must not create disconnected charts.
        pieces.append((left, right, controls))
        validation_points.extend(at.tolist())
        if len(pieces) + len(pending) > 1024:
            raise ArithmeticError("Operator conversion exceeded its polynomial-piece limit.")
    pieces.sort(key=lambda item: item[0])
    knots = torch.tensor([pieces[0][0], *[piece[1] for piece in pieces]], dtype=torch.float64)
    knots = physical_left + (knots + 1) / 2 * (physical_right - physical_left)
    knots[0], knots[-1] = physical_left, physical_right
    if not (knots[1:] > knots[:-1]).all():
        raise ArithmeticError("Polynomial pieces collapse at the represented physical input scale.")
    controls = torch.cat([pieces[0][2], *[piece[2][1:] for piece in pieces[1:]]])
    spline = _Spline(knots, degree, controls, torch.zeros(values.shape[1], dtype=torch.float64))
    actual = np.unique(
        np.r_[
            inputs[:, 0].numpy(),
            physical_left + (np.array(validation_points) + 1) / 2 * (physical_right - physical_left),
        ]
    )
    actual = np.clip(actual, physical_left, physical_right)
    expected = physical_predict(torch.from_numpy(actual[:, None]))
    actual_error = float((spline.predict(torch.from_numpy(actual)) - expected).norm(dim=1).max())
    if not math.isfinite(actual_error) or actual_error > tolerance:
        raise ArithmeticError("Actual Bernstein conversion failed its observed/support/verification-point check.")
    diagnostics.update(
        polynomial_pieces=len(pieces),
        initial_support_intervals=len(partition) - 1,
        initial_partition="known fitted-operator support knots only",
        subdivision_depth_is_per_seed_interval=True,
        conversion_checked_max_error=max(maximum_error, actual_error),
        conversion_checked_tolerance=tolerance,
        conversion_is_certified=False,
        conversion_oracle_rows=0,
    )
    return spline
