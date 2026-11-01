"""Verify the new vertex formula with independent original risk evaluations."""

from fractions import Fraction

import pytest

from tests.sampling import _exact_coefficients, _scalar_risk_sum
from tools.risk_research.vertex_single import tetrahedron_vertex_coefficient


@pytest.mark.parametrize("degree", range(1, 21))
def test_vertex_single_sum_matches_original_kernel_scalar_sum(degree):
    assert tetrahedron_vertex_coefficient(degree) == _scalar_risk_sum(4, degree, 1)


@pytest.mark.parametrize("degree", range(1, 7))
def test_vertex_single_sum_matches_independent_gram_propagation(degree):
    assert tetrahedron_vertex_coefficient(degree) == _exact_coefficients(4, degree)[0][0]


def test_vertex_single_sum_matches_new_independent_high_degree_holdout():
    # Generated at D260 by the pre-existing integer F^3 kernel oracle after
    # the candidate source was frozen; earlier exploration ended at D240.
    # Oracle source SHA256: 50a7adf30eb606fcb86a56427458ecb6a60e8db977228e71739e7068cc74f2d5.
    numerator = int(
        "506610173587353507785210798947230496462727379767993640021572819595049288"
        "963219541597775215108599226436399642883295588365677654527160723635716191"
        "872785814863442701608005842782904294693107917181273846307463067271401483"
        "885468059610445094090323646605201371905542391193587745815740211553309936"
        "2558964629175223142612410820536468889123333971151"
    )
    denominator = int(
        "118949415032999758778638929716321578975307797660780580285988619398901131"
        "484762196081621417691086383550345371844621947392521182018797014686574410"
        "087236822855727518809202813006476688210533480871223558760879815782731636"
        "894832848487937623617442104742341755351346736316399698875856878345130700"
        "04696844150028638655061109869744820767057868310612992000000"
    )
    assert tetrahedron_vertex_coefficient(260) == Fraction(numerator, denominator)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5])
def test_vertex_degree_must_be_positive_integer(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        tetrahedron_vertex_coefficient(invalid)
