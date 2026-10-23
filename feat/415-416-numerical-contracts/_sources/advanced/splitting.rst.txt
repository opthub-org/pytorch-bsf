Exact local subdivision
=======================

Subdivide a selected cell with an explicit edge or selection policy::

    from torch_bsf.splitting import EdgeSplit, LongestEdge, subdivide
    pieces = subdivide(model, cell=0, policy=EdgeSplit(0, 1, fraction=.4))
    child = pieces.models[0]
    parent_points = pieces.maps[0].to_parent(child_points)

``child(child_points)`` agrees with ``model(parent_points)``. The inverse map
returns child coordinates and a parent-row membership mask. ``LongestEdge``
chooses an edge from the selected model's vertex chords. Subdivision returns
local models and parameter maps; it does not automatically remesh neighbors.

``ResidualSplit(samples, resolution=10)`` selects the smallest sum of child
MSEs over a fraction grid. It requires equally weighted observations on the
selected cell. This groups existing residuals; it does not refit the children
or guarantee that approximation error decreases.
