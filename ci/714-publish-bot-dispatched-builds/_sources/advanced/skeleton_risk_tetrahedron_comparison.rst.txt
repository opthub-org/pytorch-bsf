Tetrahedron risk comparison at every degree
===========================================

Under the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`, the new positive vertex sum closes the
comparison on a tetrahedron at every degree. Here :math:`M=4` is the
number of vertices. The common positive noise and output factors cancel,
so the two normalized asymptotic risks are

.. math::

   R_{\mathrm{ind}}(D)=\left(\sum_{m=0}^3\sqrt{a_m(4,D)}\right)^2,
   \qquad R_{\mathrm{all}}(D)=\binom{D+3}3.

Inactive coefficients are zero. This fixed-vertex-count result is distinct
from :doc:`skeleton_risk_comparisons`, which fixes the degree and ranges
over every vertex count.

.. prf:theorem:: Complete tetrahedron comparison
   :label: thm-risk-tetrahedron-comparison

   For every integer :math:`D\geq1`, optimal inductive skeleton fitting on
   a tetrahedron has strictly smaller asymptotic risk than all-at-once
   fitting exactly when

   .. math::

      D\in\{1,2\}\quad\text{or}\quad D\geq8.

   All-at-once fitting has strictly smaller risk for
   :math:`D\in\{3,4,5,6,7\}`. Equality never occurs.
   The finite prefix in this theorem is computer-assisted.

.. prf:proof::

   We prove the comparison for :math:`D\geq625` by inequalities and
   certify every remaining degree with exact rational square-root
   enclosures. No monotonicity of the risk difference is assumed.

   **Vertex bound.** In
   :prf:ref:`thm-risk-tetrahedron-vertex-single`,
   :math:`G_0=1/6`. For :math:`j\geq1`, use
   :math:`H_j\leq j`, :math:`H_j^{(2)}\leq j` and drop the negative
   term to obtain

   .. math::

      0<G_j\leq\frac2{j+3}+\frac16<1.

   Vandermonde's identity gives
   :math:`Z_D=\sum_j\binom{D-1}j\binom{D+1}j=\binom{2D}{D-1}`.
   Since :math:`(D-1)!(D+1)!Z_D=(2D)!`, the vertex formula implies

   .. math::

      a_0(4,D)\leq
      \frac{576}{(2D+1)(2D+2)(2D+3)}
      \leq\frac{72}{D^3}.

   **Edge bound.** Fix one source-edge angular mode. Let :math:`p` be
   its tetrahedron propagation and :math:`g` its triangle propagation
   with the outside coordinate replaced by the sum of the two outside
   coordinates, as in the proof of
   :prf:ref:`thm-risk-codimension-two-modes`.
   Their traces agree on every facet, so :math:`p-g` is a
   full-simplex bubble and propagation orthogonality gives
   :math:`\langle p,p-g\rangle=0`. Hence
   :math:`\lVert p\rVert^2=\langle p,g\rangle\leq
   \lVert p\rVert\lVert g\rVert`, and
   :math:`\lVert p\rVert\leq\lVert g\rVert`.

   If :math:`r` is the sum of the two source-edge coordinates, its
   density is :math:`6r(1-r)` on the tetrahedron and :math:`2r` on the
   triangle. The conditional source-edge angular law is the same, and
   :math:`g` depends on the outside coordinates only through their sum.
   Therefore :math:`\lVert g\rVert^2` on the tetrahedron is at most
   three times its triangle norm. Dividing by the same source-edge
   norm and summing the modes, with the allocation factors
   :math:`\binom42^2=36` and :math:`\binom32^2=9`, gives

   .. math::

      a_1(4,D)\leq12a_1(3,D).

   For :math:`D\geq2`, the triangle edge formula gives

   .. math::

      a_1(3,D)=\frac{18}{2D+3}
       \left(H_{D-1}+H_{2D+2}-H_{D+3}\right)<9.

   Indeed :math:`H_{D-1}\leq D-1`, while the difference of the other
   two harmonic numbers is the sum of :math:`D-1` terms, each at most
   :math:`1/(D+4)`, and is smaller than one. Thus
   :math:`a_1(4,D)<108`.

   **Facet bound.** Reindexing the codimension-one norm sum with
   :math:`j=D-2-k` gives, for :math:`D\geq3`,

   .. math::

      a_2(4,D)=48\sum_{j=1}^{D-2}
       \frac{D-1-j}{j(2D+4-j)}
       \leq24H_{D-2}\leq24(1+\log D)\leq48\sqrt D.

   Here :math:`(D-1-j)/(2D+4-j)\leq1/2` and
   :math:`\log D\leq\sqrt D` for :math:`D\geq1`.

   **Infinite tail.** Put :math:`z=D^{1/4}` and
   :math:`X=\sqrt{a_0}+\sqrt{a_1}+\sqrt{a_2}`. If
   :math:`D\geq625`, then :math:`z\geq5` and the three bounds imply

   .. math::

      X<7z+11+\frac9{z^6}\leq\frac{19}2z.

   For the last inequality,
   :math:`11+9/z^6\leq11+9/5^6<25/2\leq(5/2)z`.
   The top coefficient is :math:`b=a_3(4,D)=\binom{D-1}3`, so
   :math:`\sqrt b\leq D^{3/2}/2` and

   .. math::

      \frac{R_{\mathrm{ind}}-b}{D^2}
       &=\frac{2\sqrt b\,X+X^2}{D^2}\\
       &<\frac{19}{2z}+\frac{361}{4z^6}
        \leq\frac{19}{10}+\frac{361}{62500}<2.

   In contrast,

   .. math::

      R_{\mathrm{all}}-b
       =\binom{D+3}3-\binom{D-1}3=2D^2+2.

   Therefore inductive fitting strictly wins for every
   :math:`D\geq625`.

   **Finite prefix.** For each :math:`1\leq D\leq624`, evaluate the
   vertex single sum, the proved tetrahedron edge recurrence
   representation, the harmonic facet formula, and the top coefficient
   over the rationals. Inactive coefficients are set to zero. For any
   nonnegative coefficient :math:`a=u/v`, let

   .. math::

      \ell=\operatorname{isqrt}
          \left(\left\lfloor\frac{u2^{2p}}v\right\rfloor\right).

   Then :math:`\ell/2^p\leq\sqrt a\leq(\ell+1)/2^p`, with equal
   endpoints when :math:`\ell^2v=u2^{2p}`. Sum the lower and upper
   endpoints separately and square them. Comparing these rational
   endpoints to the integer :math:`\binom{D+3}3` certifies a strict
   sign; an overlapping interval is never accepted as a result.
   All 624 comparisons are separated already at :math:`p=32`.
   The loss set is exactly :math:`\{3,4,5,6,7\}`, completing the proof.

Reproducing the finite certificate
-----------------------------------

The optional standard-library implementation emits every exact coefficient,
square-root enclosure and strict sign, together with the analytic tail
bound. From the repository root, run

.. code-block:: console

   python -m tools.risk_research.tetrahedron_comparison \
     --output .research/skeleton-risk/tetrahedron-comparison.json

The finite prefix uses ``Fraction`` and integer square roots. It does not
depend on floating-point sign decisions, a numerical large-degree fit, or
an unproved crossover pattern. The theorem concerns the normalized
asymptotic risks under the stated design assumptions; it makes no
finite-sample risk-ordering claim.
