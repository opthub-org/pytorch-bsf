A quadratic sufficient degree region, uniform in vertex count
===============================================================

Use the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. As in :doc:`skeleton_risk_fixed_vertex_tail`,
write :math:`R_{\rm ind}=(\sum_{s=1}^M\sqrt{a_{s-1}})^2` and
:math:`R_{\rm all}=\binom{D+M-1}{M-1}`. Common positive output and noise
factors are omitted.

.. prf:theorem:: A uniform quadratic winning degree region
   :label: thm-risk-quadratic-degree-region

   For every pair of integers :math:`M\geq4` and :math:`D\geq8M^2`,

   .. math::
      :label: risk-quadratic-degree-region

      R_{\rm all}(M,D)-R_{\rm ind}(M,D)
       >\frac{239}{3200}\frac{M}{(M-2)!}D^{M-2}>0.

   This is a simultaneous sufficient region. It does not identify the
   crossover or classify the remaining finite degrees at a given :math:`M`.

The result improves the constructive but very large :math:`H(M)` of
:doc:`skeleton_risk_fixed_vertex_tail`. It is compatible with the
complementary region :math:`M\geq9720D` in
:doc:`skeleton_risk_uniform_dimension_tail`. Neither condition asserts
victory throughout the intervening region. For :math:`M=4,5,6`, the new
thresholds are respectively :math:`128,200,288`; the existing special-case
classifications can give smaller sufficient thresholds.

The new threshold is smaller than the older construction for every
:math:`M\geq4`: its :math:`B_M\geq M` implies
:math:`t_M\geq12(M-2)!\geq M`, hence
:math:`H(M)\geq M^4>8M^2`. Individual coefficient majorants need not all
be smaller; the improvement comes from their joint binomial structure
and from retaining the facet's sharper estimate.

A direct diagonal majorant
----------------------------

For :math:`s\geq2`, :math:`q=M-s`, the positive-average formula of
:prf:ref:`thm-risk-nonvertex-positive-diagonals` and the absolutely
summable generating series of :doc:`skeleton_risk_vertex_general` give

.. math::

   0<\delta_q(j)\leq4^q,\qquad
   \sum_{k=0}^{D-s}\binom{k+s-2}{s-2}=\binom{D-1}{s-1}.

Indeed the diagonal is a rectangular partial sum of coefficients of
:math:`G(x,y)^q`, where :math:`\|G\|_1=4`; its absolute value is at most
:math:`4^q`. Therefore, for :math:`D\geq s`,

.. math::
   :label: risk-quadratic-direct-coefficient-majorant

   a_{s-1}(M,D)
    \leq\binom Ms^2\frac{(M-1)!\,2^{M-s}}{((s-1)!)^2}
                       D^{2s-M-1}.

Here :math:`(2D+s)_q\geq(2D)^q`,
:math:`(s)_q=(M-1)!/(s-1)!`, and
:math:`\binom{D-1}{s-1}\leq D^{s-1}/(s-1)!`.
At :math:`s=1`, exactly the same displayed bound follows from the vertex
majorant :eq:`risk-fixed-vertex-coarse-vertex`. On an inactive face
:math:`D<s`, its actual coefficient and its implementation majorant are
both zero. This avoids the repeated Cauchy--Schwarz factors in the older
recursive constants. The facet will use its sharper logarithmic estimate
instead of this direct bound.

Uniformly sum the lower skeletons
-----------------------------------

Set

.. math::

   b_0=\frac{D^{M-1}}{(M-1)!},\qquad
   p=\frac{M(M-1)}D\leq\frac18,\qquad
   X=\sum_{s=1}^{M-1}\sqrt{a_{s-1}(M,D)}.

For every nonfacet, put :math:`q=M-s\geq2`. Dividing
:eq:`risk-quadratic-direct-coefficient-majorant` by :math:`b_0` gives

.. math::

   \sqrt{a_{s-1}/b_0}
    \leq\binom Mq\frac{(M-1)!}{(M-q-1)!}
           \left(\frac{\sqrt2}{D}\right)^q
    \leq\frac{(\sqrt2p)^q}{q!}.

For :math:`q\geq2`, :math:`q!\geq2\,3^{q-2}`. With
:math:`x=\sqrt2p\leq3p/2<3`, it follows that

.. math::

   \sum_{s=1}^{M-2}\sqrt{a_{s-1}/b_0}
     \leq\sum_{q=2}^{\infty}\frac{x^q}{q!}
     \leq\frac{x^2}{2(1-x/3)}
     \leq\frac{p^2}{1-p/2}\leq\frac{2p}{15}.

For the facet, :eq:`risk-fixed-vertex-nonvertex-bound` is replaced by the
sharper facet estimate in :doc:`skeleton_risk_fixed_vertex_tail`:

.. math::

   \sqrt{a_{M-2}/b_0}
      \leq p\sqrt{\frac{(M-2)(1+\log D)}{2D}}.

The function :math:`(1+\log D)/D` decreases for :math:`D\geq1`, and
:math:`(1+\log(8M^2))/M` decreases for :math:`M\geq4`.
Consequently, throughout the asserted domain,

.. math::

   \frac{(M-2)(1+\log D)}{2D}
      <\frac{1+\log(8M^2)}{16M}
      \leq\frac{1+\log128}{64}<\frac{3}{32}
      <\left(\frac5{16}\right)^2.

The elementary :math:`\log128<5` can be checked without a numerical
logarithm: :math:`e^5>\sum_{k=0}^7 5^k/k!>128`.
Combining the facet and all nonfacets yields

.. math::

   T:=\frac{X}{\sqrt{b_0}}
     <\left(\frac5{16}+\frac2{15}\right)p
      =\frac{107}{240}p<\frac9{20}p.

Compare with the all-at-once gap
---------------------------------

Let :math:`b=a_{M-1}(M,D)=\binom{D-1}{M-1}\leq b_0`.
The positive elementary-symmetric expansion in
:doc:`skeleton_risk_fixed_vertex_tail` gives, for :math:`D\geq M`,

.. math::

   R_{\rm all}-b\geq\frac{M}{(M-2)!}D^{M-2}=p b_0.

On the other hand,

.. math::

   \frac{R_{\rm ind}-b}{b_0}
     =\frac{2\sqrt b X+X^2}{b_0}
     \leq2T+T^2
     <\left(\frac9{10}+\frac{81}{400}p\right)p
     \leq\frac{2961}{3200}p.

Subtracting proves :eq:`risk-quadratic-degree-region`. All estimates are
simultaneous in :math:`M,D`; no fixed-dimension asymptotic remainder or
finite scan has been substituted into a growing-dimension limit.

Exact specialization and limits
--------------------------------

The standard-library checkout tool records the region, rational margin,
direct majorants and proof/source fingerprints:

.. code-block:: console

   $ python -m tools.risk_research.quadratic_degree_tail --vertices 6 --output .research/skeleton-risk/quadratic6.json
   $ python -m tools.risk_research.quadratic_degree_tail --verify .research/skeleton-risk/quadratic6.json

It never calculates an actual risk or infers a theorem from a finite scan.
The normalized quantitative margin is relative to :math:`b_0` and applies
only for :math:`M\geq4,D\geq8M^2`. Below that domain the tool's coefficient
majorants remain valid, but they give no new classification.
Verification recomputes the complete record with exact JSON types and
rejects duplicate keys, extra fields, modified domains or changed sources.
Fingerprints canonicalize CRLF to LF; execution guards compare raw source
bytes before and after. Failed generation preserves an existing output.
Output paths resolving to any fingerprinted evaluator or proof source,
including symlink aliases, are rejected before evaluation or writing.
The destination is checked again after calculation and immediately before writing.
Independent Gram/scalar coefficient checks, integer-root comparisons and
frozen holdouts validate implementation; the proof establishes the
infinite region.
