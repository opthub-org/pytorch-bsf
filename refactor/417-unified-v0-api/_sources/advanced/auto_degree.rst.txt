Cross-validation and degree search
==================================

Cross-validation returns evidence rather than an implicit ensemble::

    from torch_bsf.model_selection import KFold, cross_validate, select_degree
    from torch_bsf import Ensemble, fit
    validation = cross_validate(space, samples, cv=KFold(5, seed=7))
    ensemble = Ensemble(validation.models)
    search = select_degree(space, samples, degrees=[1, 2, 3], cv=KFold(5, seed=7))
    if search.success:
        refitted = fit(search.best_space, samples)

Every fold retains train/test indices, FitResult and held-out weighted errors.
The pooled MSE weights held-out observations; mean and standard deviation
of fold MSEs are also available. Failed folds remain visible and cannot win a
search. Every requested degree is evaluated on the same splits; no early
stopping or unreported full-data refit occurs.
