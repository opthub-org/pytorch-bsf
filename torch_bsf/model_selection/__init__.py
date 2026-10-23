"""Validation and degree search return explicit evidence rather than bare models."""

from torch_bsf.model_selection._validation import (
    CVResult,
    DegreeCandidate,
    DegreeSearchResult,
    FoldResult,
    KFold,
    cross_validate,
    select_degree,
)

__all__ = ["KFold", "FoldResult", "CVResult", "DegreeCandidate", "DegreeSearchResult", "cross_validate", "select_degree"]
