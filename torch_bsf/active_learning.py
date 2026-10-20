import itertools
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from torch_bsf.sampling import simplex_random
from torch_bsf.simploidal import sample_simploid, simploid_n_params


def _infer_device(model: nn.Module) -> torch.device:
    """Infer the device of *model* preferring an explicit ``model.device`` attribute,
    then its parameters, then its buffers, and finally falling back to CPU."""
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    # Try to infer device from parameters
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    # Try to infer device from buffers
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    # Model has neither parameters nor buffers; fall back to CPU
    return torch.device("cpu")


def suggest_next_points(
    models: Sequence[nn.Module],
    n_suggestions: int = 1,
    n_candidates: int = 1000,
    method: Literal["qbc", "density"] = "qbc",
    params: Optional[torch.Tensor] = None,
    n_params: Optional[int] = None,
    simploid_type: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Suggest points on the simplex or simploid where new data should be sampled.

    The candidates are drawn uniformly from the simploid of type ``simploid_type``
    (see :func:`torch_bsf.simploidal.sample_simploid`), or from the simplex of
    dimension ``n_params - 1`` for models without a type.

    Parameters
    ----------
    models : Sequence[nn.Module]
        An ensemble of models (e.g., from k-fold cross-validation).
        Each model must be callable with a tensor of shape
        ``(n_candidates, n_params)`` and return predictions.
        Accepts any :class:`~collections.abc.Sequence` of
        :class:`~torch.nn.Module` instances, including
        :class:`~torch.nn.ModuleList`.
    n_suggestions : int, default=1
        The number of points to suggest.
    n_candidates : int, default=1000
        The number of candidate points to evaluate.
    method : Literal["qbc", "density"], default="qbc"
        The method to use:
        - "qbc": Query-By-Committee. Suggests points where models disagree most.
        - "density": Suggests points that are furthest from existing training points.
    params : torch.Tensor, optional
        The existing training parameters. Required for method="density".
    n_params : int, optional
        The number of simplex parameters (input dimension).  When omitted,
        the value is inferred from ``models[0].n_params`` if that attribute
        exists.  When provided and ``models[0]`` exposes an ``n_params``
        attribute, the two values must agree; a ``ValueError`` is raised on
        mismatch.
    simploid_type : Sequence[int], optional
        The type of the simploid to sample.  When omitted, it is taken from
        ``models[0].simploid_type`` if that attribute exists (e.g., for
        :class:`~torch_bsf.simploidal.BezierSimploid` models), and the candidates are drawn
        from a simplex otherwise.  All models with a ``simploid_type`` must agree with it.
    seed : int, optional
        Random seed for drawing the candidates, which makes the suggestions reproducible.

    Returns
    -------
    torch.Tensor
        The suggested points in shape (n_suggestions, n_params).

    Examples
    --------
    >>> import torch_bsf
    >>> from torch_bsf.active_learning import suggest_next_points
    >>> from torch_bsf.simploidal import sample_simploid
    >>> t = sample_simploid((1, 1), 30, seed=0)
    >>> models = torch_bsf.fit_kfold(t, t[:, :1] * t[:, 2:3], degree=2, simploid_type=(1, 1), seed=0)
    >>> suggest_next_points(models, n_suggestions=3).shape
    torch.Size([3, 4])
    """
    if not models:
        raise ValueError("models must be a non-empty sequence of models")

    if simploid_type is None:
        simploid_type = getattr(models[0], "simploid_type", None)
    if simploid_type is not None:
        simploid_type = tuple(simploid_type)
        for i, model in enumerate(models):
            model_type = getattr(model, "simploid_type", None)
            if model_type is not None and tuple(model_type) != simploid_type:
                raise ValueError(f"models[{i}].simploid_type={model_type} does not match simploid_type={simploid_type}.")
        if n_params is None:
            n_params = simploid_n_params(simploid_type)
        elif n_params != simploid_n_params(simploid_type):
            raise ValueError(f"n_params={n_params} does not match simploid_type={simploid_type}.")

    model_n_params = getattr(models[0], "n_params", None)
    if n_params is None:
        n_params = model_n_params
        if n_params is None:
            raise ValueError("n_params must be provided when models do not have an 'n_params' attribute")
    elif model_n_params is not None and n_params != model_n_params:
        raise ValueError(f"n_params mismatch: explicit n_params={n_params} but models[0].n_params={model_n_params}")

    device = _infer_device(models[0])

    # Validate that all models share the same n_params and device
    for i, model in enumerate(itertools.islice(models, 1, None), start=1):
        model_n_params = getattr(model, "n_params", None)
        if model_n_params is not None and model_n_params != n_params:
            raise ValueError(f"models[{i}].n_params={model_n_params} does not match expected n_params={n_params}.")
        model_device = _infer_device(model)
        if model_device != device:
            raise ValueError(
                f"models[{i}] is on device '{model_device}' but models[0] is on "
                f"device '{device}'. All models in 'models' must be on the same device."
            )
    # Generate candidate points
    if simploid_type is not None:
        candidates = sample_simploid(simploid_type, n_candidates, seed=seed).to(device)
    else:
        candidates = simplex_random(n_params, n_candidates, seed=seed).to(device)

    if method == "qbc":
        # Calculate variance across models
        preds: list[torch.Tensor] = []
        for model in models:
            model.eval()
            with torch.no_grad():
                preds.append(model(candidates))

        # preds_stacked: (K, n_candidates, n_values)
        preds_stacked = torch.stack(preds)

        # Variance of predictions: (n_candidates, n_values)
        # Use unbiased=False so a single-model committee yields zero (not NaN).
        var = torch.var(preds_stacked, dim=0, unbiased=False)

        # Sum of variance across dimensions: (n_candidates,)
        uncertainty = torch.sum(var, dim=1)

        # Select top candidates
        _, indices = torch.topk(uncertainty, n_suggestions)
        return candidates[indices]

    elif method == "density":
        if params is None:
            raise ValueError("params must be provided for method='density'")

        # Calculate distance to nearest neighbor in parameter space
        # candidates: (C, M), params: (N, M)
        # We use squared Euclidean distance
        dist = torch.cdist(candidates, params.to(device))

        # min distance to any existing point: (C,)
        min_dist, _ = torch.min(dist, dim=1)

        # Select points with largest min_dist
        _, indices = torch.topk(min_dist, n_suggestions)
        return candidates[indices]

    else:
        raise ValueError(f"Unknown method: {method}")
