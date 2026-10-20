from argparse import ArgumentParser

import numpy as np


def reverse_logspace(num: int = 50, base: float = 10) -> np.ndarray:
    """Return numbers spaced evenly on a log scale.

    Parameters
    ----------
    num : integer, optional
        Number of samples to generate. Default is ``50``. Must be non-negative.
    base : array_like, optional
        The base of the log space.
        The step size between the elements in ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is ``10.0``.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    Examples
    --------
    >>> reverse_logspace(0)
    array([], dtype=float64)
    >>> reverse_logspace(1)
    array([0.])
    >>> reverse_logspace(2)
    array([0.        , 0.75974693])
    >>> reverse_logspace(3)
    array([0.        , 0.59537902, 0.87172948])
    >>> reverse_logspace(num=5, base=0.001)
    array([0.        , 0.00298406, 0.0148638 , 0.06215789, 0.25043908])
    >>> reverse_logspace(num=5, base=0.01)
    array([0.        , 0.01527158, 0.05363205, 0.14998921, 0.39202745])
    >>> reverse_logspace(num=5, base=0.1)
    array([0.        , 0.06498813, 0.16798738, 0.33123019, 0.5899526 ])
    >>> reverse_logspace(num=5, base=1)
    array([0. , 0.2, 0.4, 0.6, 0.8])
    >>> reverse_logspace(num=5, base=10)
    array([0.        , 0.4100474 , 0.66876981, 0.83201262, 0.93501187])
    >>> reverse_logspace(num=5, base=100)
    array([0.        , 0.60797255, 0.85001079, 0.94636795, 0.98472842])
    >>> reverse_logspace(num=5, base=1000)
    array([0.        , 0.74956092, 0.93784211, 0.9851362 , 0.99701594])
    """
    if base > 1.0:
        return 1.0 - (np.logspace(1.0, 0.0, num, endpoint=False, base=base) - 1.0) / (
            base - 1
        )
    elif base == 1.0:
        return np.linspace(0.0, 1.0, num, endpoint=False)
    elif base > 0:
        inv_base = 1.0 / base
        return (np.logspace(0.0, 1.0, num, endpoint=False, base=inv_base) - 1.0) / (
            inv_base - 1
        )
    else:
        raise ValueError("base must be > 0, but given: {base}")


def elastic_net_grid(
    n_lambdas: int = 102,
    n_alphas: int = 12,
    n_vertex_copies: int = 1,
    base: float = 10
) -> np.ndarray:
    """Return an array of 3D grid points on the standard 2-simplex, which is suitable for grid search for elastic net's hyperparameters.

    The grid consists of ``n_lambdas - 1`` regularization-strength slices (height levels),
    each containing ``n_alphas`` equally spaced points along the alpha axis with
    ``0 <= w_1 < 1``, together with the data-fidelity vertex at ``w_1 = 1`` and any
    additional vertex copies. When either ``n_lambdas == 0`` or ``n_alphas == 0``
    (i.e. the product ``n_lambdas * n_alphas == 0``), an empty ``(0, 3)`` array is
    returned. Otherwise the shape is
    ``((n_lambdas - 1) * n_alphas + 3 * n_vertex_copies - 2, 3)``.
    The first column (``w_1``) values are log-spaced: :func:`reverse_logspace` generates
    ``n_lambdas - 1`` values of ``w_1`` in ``[0, 1)`` and the vertex ``w_1 = 1`` is appended separately.
    The regularization strength is derived as ``λ = (1 - w_1) / w_1``.
    The second and third column values are spaced evenly over a specified interval.

    Parameters
    ----------
    n_lambdas : int, optional
        Number of samples to generate along the regularization-strength axis.
        :func:`reverse_logspace` generates ``n_lambdas - 1`` values of the
        data-fidelity weight ``w_1`` log-spaced in ``[0, 1)``, and the
        data-fidelity vertex (``w_1 = 1``, i.e. ``λ = 0``) is appended separately.
        Default is ``102``. Must be non-negative.
    n_alphas : int, optional
        Number of samples to generate along `alpha` axis.
        The values are equally spaced.
        Default is ``12``. Must be non-negative.
    n_vertex_copies : int, optional
        Controls how many additional copies of the three simplex vertices are
        appended to the grid as extra rows. In particular, the implementation
        appends extra vertex rows so that, for typical settings with
        ``n_lambdas >= 2`` and ``n_alphas >= 2``, each vertex appears
        ``n_vertex_copies`` times overall (including those already present in
        the main grid). This can be useful for k-fold cross-validation (one
        can choose ``n_vertex_copies >= k``). Default is ``1``. Must be a
        positive integer (>= 1).
    base : float, optional
        The base of the log space.
        The step size between the elements in ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is ``10.0``.

    Returns
    -------
    samples : ndarray
        An empty ``(0, 3)`` array when ``n_lambdas == 0`` or ``n_alphas == 0``;
        otherwise an array of shape ``((n_lambdas - 1) * n_alphas + 3 * n_vertex_copies - 2, 3)``.

    Examples
    --------
    >>> elastic_net_grid(0, 0)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(0, 1)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(1, 0)
    array([], shape=(0, 3), dtype=float64)
    >>> elastic_net_grid(1, 1)
    array([[1., 0., 0.]])
    >>> elastic_net_grid(1, 2)
    array([[1., 0., 0.]])
    >>> elastic_net_grid(2, 1)
    array([[0., 0., 1.],
           [1., 0., 0.]])
    >>> elastic_net_grid(2, 2)
    array([[0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    >>> elastic_net_grid(2, 3)
    array([[0. , 0. , 1. ],
           [0. , 0.5, 0.5],
           [0. , 1. , 0. ],
           [1. , 0. , 0. ]])
    >>> elastic_net_grid(3, 3)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ]])
    >>> elastic_net_grid(3, 3, 2)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])
    >>> elastic_net_grid(3, 3, 3)
    array([[0.        , 0.        , 1.        ],
           [0.        , 0.5       , 0.5       ],
           [0.        , 1.        , 0.        ],
           [0.75974693, 0.        , 0.24025307],
           [0.75974693, 0.12012654, 0.12012654],
           [0.75974693, 0.24025307, 0.        ],
           [1.        , 0.        , 0.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ],
           [1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])
    """
    if n_vertex_copies < 1:
        raise ValueError(f"n_vertex_copies must be >= 1, but given: {n_vertex_copies}")
    if n_lambdas < 0 or n_alphas < 0:
        raise ValueError(
            f"n_lambdas and n_alphas must be >= 0, but given: n_lambdas={n_lambdas}, n_alphas={n_alphas}"
        )
    if n_lambdas == 0 or n_alphas == 0:
        return np.empty((0, 3))
    w1_values = reverse_logspace(n_lambdas - 1, base)
    w2_values = np.linspace(0.0, 1.0 - w1_values, n_alphas, axis=1)
    n_vertices = 3 * n_vertex_copies - 2

    grids = np.zeros(((n_lambdas - 1) * n_alphas + n_vertices, 3))
    grids[:-n_vertices, 0] = np.repeat(w1_values, n_alphas)
    grids[:-n_vertices, 1] = w2_values.reshape(-1)  # faster than ravel and flatten
    grids[:-n_vertices, 2] = 1 - grids[:-n_vertices, 0] - grids[:-n_vertices, 1]
    grids[-n_vertices] = [1.0, 0.0, 0.0]  # last entry

    if n_vertices > 1:
        # fill remaining space with vertex copies
        grids[-n_vertices + 1:] = np.tile(np.eye(3), (n_vertex_copies - 1, 1))

    return grids


def _cli_main() -> None:
    """Entry point for ``python -m torch_bsf.model_selection.elastic_net_grid``."""
    parser = ArgumentParser(
        prog="python -m torch_bsf.model_selection.elastic_net_grid",
        description="Grid search for elastic net's hyperparameters",
    )
    parser.add_argument("--n_lambdas", help="Number of samples for lambda values", type=int, default=102)
    parser.add_argument("--n_alphas", help="Number of samples for alpha values", type=int, default=12)
    parser.add_argument("--n_vertex_copies", help="Number of copies of each vertex", type=int, default=1)
    parser.add_argument("--base", help="Base of the log space", type=float, default=10)
    args = parser.parse_args()

    grid = elastic_net_grid(args.n_lambdas, args.n_alphas, args.n_vertex_copies, base=args.base)
    print("\n".join(f"{p[0]:.17e},{p[1]:.17e},{p[2]:.17e}" for p in grid.tolist()))


if __name__ == "__main__":
    _cli_main()
