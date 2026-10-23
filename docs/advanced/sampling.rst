Sampling and grids
==================

All sampling is parameterized by the same space::

    from torch_bsf.sampling import sample, grid
    random_points = sample(space, 100, seed=7)
    sobol_points = sample(space, 100, method="sobol", seed=7)
    lattice = grid(space, resolution=10)

The returned Points include cell IDs. Sampling defaults to equal mass on
maximal cells, with uniform Dirichlet coordinates on each factor. Specify
``cell_weights={cell_id: weight}`` for another cell distribution. Sobol sampling
uses scrambled low-discrepancy coordinates within each cell; cell selection
remains categorical. Seeds use local generators and preserve global RNG state.
