An explicit winning degree tail at every fixed vertex count
===========================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. Write

.. math::

   R_{\rm ind}(M,D)=\left(\sum_{s=1}^M\sqrt{a_{s-1}(M,D)}\right)^2,
   \qquad R_{\rm all}(M,D)=\binom{D+M-1}{M-1}.

Inactive coefficients are zero. The common positive noise and output
factors cancel. The result concerns asymptotic statistical risk.

.. prf:theorem:: Constructive eventual victory at every fixed vertex count
   :label: thm-risk-fixed-vertex-degree-tail

   For every integer :math:`M\geq4`, the construction below returns an
   explicit integer :math:`H(M)\geq M` such that, for every integer
   :math:`D\geq H(M)`,

   .. math::

      R_{\rm ind}(M,D)<R_{\rm all}(M,D).

   The construction uses only rational arithmetic and integer roots.
   Its constants are deliberately conservative; it does not classify
   the finite prefix below :math:`H(M)` or locate an exact crossover.

The complementary sufficient region :math:`M\geq9720D` is proved in
:doc:`skeleton_risk_uniform_dimension_tail`. The two conditions describe
different directions; they do not classify their intervening region.

Coefficient bounds
--------------------

**Vertices.** The coefficients of
:math:`g(t)=-(1-t)\log(1-t)/t` are
:math:`g_0=1`, :math:`g_j=-1/(j(j+1))` for :math:`j\geq1`.
In the absolutely summable bivariate series

.. math::

   G(x,y)=g(x)+g(y)-g(xy),

the constant coefficient is one and the three nonconstant axis/diagonal
supports are disjoint. Thus :math:`\lVert G\rVert_1=4` and
:math:`\lVert G^q\rVert_1\leq4^q`.
The positive transformed diagonal :math:`\delta_q(j)` from
:doc:`skeleton_risk_vertex_general` is a rectangular partial sum of
these coefficients, so :math:`0<\delta_q(j)\leq4^q`.
The positive binomial weights in that chapter sum to :math:`Z_D`.
Consequently, at :math:`q=M-1`, for every :math:`D\geq1`,

.. math::
   :label: risk-fixed-vertex-coarse-vertex

   a_0(M,D)\leq V_M D^{-(M-1)},\qquad
   V_M=M^2(M-1)!2^{M-1}.

This coarse estimate is used only to prove a winning region; the sharper
vertex asymptotic and its explicit error remain available separately.

**Nonvertex faces with at least two outside coordinates.** For
:math:`s\geq2`, define rational constants

.. math::
   :label: risk-fixed-vertex-transfer-constants

   A_{s,0}&=\frac1{(s-1)!},\\
   A_{s,q}&=(2^q-1)\sum_{r=0}^{q-1}\binom qr
       \frac{(s+q-1)!}{2^{q-r}(s+r-1)!}A_{s,r},\\
   C_{s,q}&=\binom{s+q}s^2A_{s,q}.

For every :math:`q\geq0,D\geq s`,

.. math::
   :label: risk-fixed-vertex-nonvertex-bound

   a_{s-1}(s+q,D)\leq C_{s,q}D^{s-q-1}.

At :math:`q=0` this is
:math:`\binom{D-1}{s-1}\leq D^{s-1}/(s-1)!`.
For the induction step, apply the boundary-transfer inequality in
:doc:`skeleton_risk_codimension_three` before grouping the proper
outside subsets. Cauchy--Schwarz over their :math:`2^q-1` terms gives

.. math::

   \frac{a_{s-1}(s+q,D)}{\binom{s+q}s^2}
   \leq(2^q-1)\sum_{r=0}^{q-1}\binom qr
      \frac{(s+r)_{2D}}{(s+q)_{2D}}
      \frac{a_{s-1}(s+r,D)}{\binom{s+r}s^2}.

The embedding ratio satisfies

.. math::

   \frac{(s+r)_{2D}}{(s+q)_{2D}}
      =\frac{(s+q-1)!}{(s+r-1)!}
         \prod_{j=s+r}^{s+q-1}\frac1{2D+j}
      \leq\frac{(s+q-1)!}{2^{q-r}(s+r-1)!}D^{r-q}.

Insertion of the induction hypothesis proves the recursion and bound.
All quantities are zero on the inactive prefix :math:`D<s`.

**Facets.** Put :math:`s=M-1\geq3`, :math:`L(D)=1+\log D` and
:math:`F_M=sM^2/(2(s-2)!)`. The exact codimension-one formula, with
:math:`j=D-s-k+1`, gives

.. math::

   a_{s-1}(M,D)
     =sM^2\sum_{j=1}^{D-s+1}
        \frac{\binom{D-j-1}{s-2}}{j(2D+s+1-j)}
     \leq F_M D^{M-4}L(D).

Indeed, the numerator is at most
:math:`(D-j)^{s-2}/(s-2)!`, and
:math:`2D+s+1-j\geq2(D-j)>0` on the summation range.
Since :math:`s-3\geq0`, bound the remaining power by
:math:`D^{s-3}` and use :math:`H_{D-s+1}\leq L(D)`.
This is zero when :math:`D<s`.

Threshold and proof of the theorem
-----------------------------------

Let :math:`\lceil\sqrt{x}\rceil` mean the least integer whose square
is at least the positive rational :math:`x`. Set

.. math::

   B_M=\lceil\sqrt{V_M}\rceil+\lceil\sqrt{F_M}\rceil
             +\sum_{s=2}^{M-2}\lceil\sqrt{C_{s,M-s}}\rceil,
   \qquad c_M=\frac{M}{(M-2)!}.

The exponents of all the nonfacet bounds are at most :math:`M-5`;
the vertex exponent is also at most :math:`M-4` for :math:`M\geq4`.
Therefore, for :math:`D\geq M`,

.. math::

   X:=\sum_{s=1}^{M-1}\sqrt{a_{s-1}(M,D)}
      \leq B_M D^{(M-4)/2}\sqrt{L(D)}.

The elementary inequality :math:`L(D)\leq2\sqrt D` follows from
:math:`\log u\leq u-1` at :math:`u=\sqrt D`.
For :math:`b=\binom{D-1}{M-1}` one has
:math:`\sqrt b\leq D^{(M-1)/2}/\sqrt{(M-1)!}`. Hence

.. math::

   \frac{R_{\rm ind}-b}{D^{M-2}}
      =\frac{2\sqrt bX+X^2}{D^{M-2}}
      <3B_M D^{-1/4}+2B_M^2 D^{-3/2}.

Here :math:`2\sqrt2/\sqrt{(M-1)!}<3`.
Choose the integer

.. math::

   t_M=\max\left(1,
        \left\lceil\frac{12B_M}{c_M}\right\rceil,
        \left\lceil(8B_M^2/c_M)^{1/6}\right\rceil\right),
   \qquad H(M)=\max(M,t_M^4).

At every :math:`D\geq H(M)`, the upper bound is at most

.. math::

   \frac{3B_M}{t_M}+\frac{2B_M^2}{t_M^6}
      \leq\frac{c_M}4+\frac{c_M}4=\frac{c_M}2<c_M.

Finally, put :math:`n=M-1` and expand

.. math::

   R_{\rm all}-b
     =\frac{\prod_{j=1}^n(D+j)-\prod_{j=1}^n(D-j)}{n!}.

All the surviving elementary-symmetric terms have positive coefficients.
The first is :math:`2(1+\cdots+n)D^{n-1}/n!=c_MD^{M-2}`.
Thus :math:`R_{\rm all}-b\geq c_MD^{M-2}`,
which proves strict victory throughout the claimed tail.

Reproduction and limits
-------------------------

The optional standard-library checkout tool specializes this general
proof without calculating any risk coefficients:

.. code-block:: console

   $ python -m tools.risk_research.fixed_vertex_tail --vertices 6 --output .research/skeleton-risk/fixed6-tail.json
   $ python -m tools.risk_research.fixed_vertex_tail --verify .research/skeleton-risk/fixed6-tail.json

The certificate records every majorant, integer-root bound, threshold,
strict rational margin, and source/proof fingerprint. Verification
recomputes the full specialization and rejects altered fields. Hashes
canonicalize CRLF pairs to LF; the concurrent-edit guard compares the
original raw bytes. A failed computation preserves an existing output.

The proof establishes eventual victory for each fixed :math:`M`, with
an explicit but large threshold. Sharper special-case comparisons can
give substantially smaller thresholds. It gives neither the exact
finite loss set at a general vertex count nor a joint asymptotic formula
when :math:`M,D` both grow.
