"""Topology builders for simplex products and their glued parameter domains.

Compile a builder with ``BezierSpace.on_set`` before fitting. A compiled space
snapshots the topology; subsequent builder mutations cannot change its basis.
"""

from torch_bsf._kernels.sets import Simploid, SimploidalSet

__all__ = ["Simploid", "SimploidalSet"]

# These builders have one canonical public import path.
Simploid.__module__ = __name__
SimploidalSet.__module__ = __name__
