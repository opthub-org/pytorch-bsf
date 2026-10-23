Explicit ensembles and queries
==============================

Construct the committee explicitly from compatible fitted models::

    from torch_bsf import Ensemble
    from torch_bsf.active_learning import query_by_committee, farthest_points
    ensemble = Ensemble(validation.models)
    query = query_by_committee(ensemble, candidate_points, n=5)
    spread = farthest_points(candidate_points, observed_points, n=5)

Queries preserve cell IDs. Committee scores average output variance;
farthest-point selection compares coordinates within each cell and gives
unobserved cells priority. Refitting is always a separate operation.
