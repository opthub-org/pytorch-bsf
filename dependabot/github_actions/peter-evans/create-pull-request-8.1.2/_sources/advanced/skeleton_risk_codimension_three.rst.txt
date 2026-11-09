Boundary transfer and a codimension-three edge bound
=====================================================

Under the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`, a trace-correct inclusion-exclusion lift
bounds every nonvertex coefficient by coefficients on smaller simplices.
For an edge on five vertices this yields a rational upper bound requiring
no degree-dependent angular sum. This is an inequality, not an exact
formula or an asymptotic equivalent for that coefficient.

A boundary lift for every nonvertex source face
------------------------------------------------

Let the source face have :math:`s\geq2` vertices and let
:math:`q\geq1` be the number of outside coordinates. Write
:math:`M=s+q` and :math:`(b)_n=b(b+1)\cdots(b+n-1)`.
At zero outside coordinates the top coefficient is

.. math::

   a_{s-1}(s,D)=\binom{D-1}{s-1},

with value zero when :math:`D<s`.

.. prf:theorem:: Nonvertex boundary-transfer inequality
   :label: thm-risk-nonvertex-boundary-transfer

   For every integer :math:`D\geq1`,

   .. math::
      :label: risk-nonvertex-boundary-transfer

      \sqrt{a_{s-1}(M,D)}
      \leq\binom Ms\sum_{r=0}^{q-1}
        \frac{\binom qr}{\binom{s+r}s}
        \sqrt{\frac{(s+r)_{2D}}{(M)_{2D}}
                   a_{s-1}(s+r,D)}.

   The source face is the same on both sides; only the number of outside
   coordinates changes. All coefficients in the inequality are zero on
   the inactive prefix :math:`D<s`. The statement here concerns
   nonvertex source faces; vertex boundary reductions are treated in
   :doc:`skeleton_risk_extensions`.

.. prf:proof::

   Assume :math:`D\geq s`, put :math:`n=D-s`, and fix one source-face
   angular mode :math:`Q_{k,\nu}` from
   :prf:ref:`thm-skeleton-scalar-sum`. Put :math:`L=n-k` and let
   :math:`r_I` denote the sum of the source coordinates. The common
   homogeneous angular factor is

   .. math::

      B_{k,\nu}(x_I)=\prod_{i\in I}x_i\,
                  r_I^k Q_{k,\nu}(x_I/r_I).

   It is a polynomial and is divisible by each source coordinate. For
   an outside-coordinate subset :math:`S\subseteq[q]`, define

   .. math::

      p_S=B_{k,\nu}(x_I)\sum_{h=0}^L
       (-1)^h\binom Lh\binom{n+k+3s-1}h
       r_I^{L-h}T_h(x_S),
      \qquad
      T_h(x_S)=\sum_{|b|=h}\frac{h!}{b!}
                       \prod_{j\in S}\frac{x_j^{b_j}}{b_j+1}.

   The empty-set convention is :math:`T_0=1,T_h=0` for :math:`h>0`.
   The scalar-mode construction proves that :math:`p_S` is the degree
   :math:`D` homogeneous propagation on the face :math:`I\cup S`.
   In particular, the coefficients do not depend on the number of
   coordinates omitted from that face. Its ambient lift is obtained by
   evaluating this same polynomial on the selected coordinates, without
   replacing any outside coordinate by a sum.

   Set :math:`p=p_{[q]}` and

   .. math::
      :label: risk-nonvertex-inclusion-exclusion

      b=\sum_{S\subsetneq[q]}(-1)^{q-|S|+1}p_S.

   On an outside facet :math:`x_j=0`, the polynomial :math:`p_S`
   restricts to :math:`p_{S\setminus\{j\}}`. Every proper subset of
   :math:`[q]\setminus\{j\}` receives two opposite coefficients in
   :math:`b`. The full remaining subset receives coefficient one, since
   :math:`S=[q]` was excluded. Thus :math:`b` and :math:`p` have the
   same trace on that facet. On every source facet both are zero because
   of :math:`B_{k,\nu}`. Their traces therefore agree on **every** facet.

   Consequently :math:`p-b` is a full-simplex bubble of degree at most
   :math:`D`. Propagation orthogonality yields
   :math:`\langle p,p-b\rangle=0`, and hence
   :math:`\lVert p\rVert\leq\lVert b\rVert`. This includes degrees
   smaller than :math:`M`, where no nonzero full bubble exists.

   Normalize the source-face bubble modes to have norm one on their
   source face. Regard the propagated polynomials for all these modes
   as one vector in the direct sum of their ambient :math:`L^2` spaces.
   The triangle inequality in this Hilbert space gives

   .. math::

      \left(\sum_\nu\lVert p_{[q],\nu}\rVert_M^2\right)^{1/2}
      \leq\sum_{S\subsetneq[q]}
        \left(\sum_\nu\lVert p_{S,\nu}\rVert_M^2\right)^{1/2}.

   If :math:`|S|=r`, the total mass of the coordinates in :math:`I\cup S`
   has the beta law with parameters :math:`s+r,M-s-r`. Its normalized
   angular coordinates are independent and uniform on that face.
   Homogeneity therefore gives the exact embedding identity

   .. math::
      :label: risk-nonvertex-embedding-norm

      \lVert p_{S,\nu}\rVert_M^2
        =\frac{(s+r)_{2D}}{(M)_{2D}}
                    \lVert p_{S,\nu}\rVert_{s+r}^2.

   Finally,

   .. math::

      a_{s-1}(s+r,D)=\binom{s+r}s^2
          \sum_\nu\lVert p_{S,\nu}\rVert_{s+r}^2.

   Grouping the :math:`\binom qr` subsets of each size proves the
   inequality. The inactive prefix is zero directly from the absence
   of source-face bubbles.

An elementary tetrahedron edge bound
-------------------------------------

The following explicit bound strengthens the coarse constant bound used
in :doc:`skeleton_risk_tetrahedron_comparison`.

.. prf:lemma:: An all-degree reciprocal-square edge bound
   :label: lem-risk-tetrahedron-edge-reciprocal-square

   For every integer :math:`D\geq2`,

   .. math::
      :label: risk-tetrahedron-edge-reciprocal-square

      0<a_1(4,D)<\frac{864}{(D+4)(D+5)}<\frac{864}{D^2}.

.. prf:proof::

   In :prf:ref:`thm-risk-codimension-two-modes`, put :math:`s=2`,
   :math:`j=L+1` and :math:`N=2D+3`. Then :math:`1\leq j\leq D-1`.
   The nonnegative mode bracket is at most :math:`H_{j+1}`: its other
   harmonic difference is nonpositive and the remaining corrections
   are nonnegative. Its denominator is
   :math:`j(j+1)(N-j)(N-j+1)`, with
   :math:`N-j\geq D+4` and :math:`N-j+1\geq D+5`.
   There is one angular mode at each :math:`k`. Including the allocation
   factor :math:`\binom42^2=36` gives

   .. math::

      a_1(4,D)\leq
       \frac{432}{(D+4)(D+5)}
          \sum_{j=1}^{D-1}\frac{H_{j+1}}{j(j+1)}.

   The finite sum has the exact telescoping evaluation

   .. math::
      :label: risk-edge-harmonic-telescope

      \sum_{j=1}^n\frac{H_{j+1}}{j(j+1)}
         =2-\frac{H_{n+1}+1}{n+1}<2.

   This proves the bound without an unquantified asymptotic remainder.

The five-vertex edge
---------------------

.. prf:theorem:: Five-vertex edge upper bound at every degree
   :label: thm-risk-five-vertex-edge-upper

   The edge coefficient is zero at :math:`D=1`. For every integer
   :math:`D\geq2`, it satisfies both bounds

   .. math::
      :label: risk-five-vertex-edge-upper

      0<a_1(5,D)&<\frac{30000}{D^2},\\
      a_1(5,D)&<300\left(
        \frac{24(D-1)}{(2D+2)(2D+3)(2D+4)}
        +\frac{108}{(2D+3)(2D+4)}
        +\frac{864}{(D+4)(D+5)(2D+4)}\right).

   Both are explicit all-degree bounds. Neither asserts the true decay
   exponent or leading asymptotic constant.

.. prf:proof::

   Apply :eq:`risk-nonvertex-boundary-transfer` at :math:`s=2,q=3`.
   The three embedding ratios are

   .. math::

      r_0=\frac{24}{(2D+2)(2D+3)(2D+4)},\quad
      r_1=\frac{12}{(2D+3)(2D+4)},\quad
      r_2=\frac4{2D+4}.

   The top coefficient at zero outside coordinates is :math:`D-1`,
   so the allocation factors simplify the transfer inequality to

   .. math::
      :label: risk-five-vertex-edge-root-upper

      \sqrt{a_1(5,D)}\leq10\left(
         \sqrt{r_0(D-1)}+\sqrt{r_1a_1(3,D)}
                +\tfrac12\sqrt{r_2a_1(4,D)}\right).

   The triangle edge formula gives :math:`a_1(3,D)<9` for
   :math:`D\geq2`, as proved in the tetrahedron comparison chapter.
   The three terms inside parentheses are, respectively, at most
   :math:`\sqrt3/D`, strictly smaller than :math:`3\sqrt3/D`, and
   strictly smaller than :math:`6\sqrt3/D`. For the last bound use
   :eq:`risk-tetrahedron-edge-reciprocal-square` and
   :math:`2D+4\geq8`. Their sum is smaller than
   :math:`10\sqrt3/D`, proving the first bound.

   Alternatively, square the three-term sum using
   :math:`(x+y+z)^2\leq3(x^2+y^2+z^2)` and retain the sharper
   denominator :math:`(D+4)(D+5)` in the tetrahedron bound. This gives
   the second, entirely rational bound. All displayed denominators are
   strictly positive on the active domain, and the strict triangle or
   tetrahedron inequalities ensure a strict final bound.

Evaluation and verification
-----------------------------

``tools.risk_research.codimension_three`` returns the smaller of the two
rational bounds and emits exact numerator/denominator JSON. Its operation
count per requested degree is bounded, although integer bit lengths grow.
``transfer_squared_weights(s,q,D)`` emits the exact rational weights under
square roots in the general transfer inequality; it does not evaluate the
risk coefficient. These tools add no production API or mandatory dependency.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.codimension_three --max-degree 100 --output .research/skeleton-risk/q3-upper.json

The optional SymPy lemma checker verifies the three outside-facet
inclusion-exclusion cancellations, the finite harmonic telescoping identity,
the rational inequality polynomials and the explicit constant. Its report
binds both source files by SHA256. The universal mode construction,
orthogonality and Hilbert-space argument are the proof above; the finite
polynomial/oracle tests are independent implementation checks.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.codimension_three_certificate --output .research/skeleton-risk/q3-lemmas.json
   $ .venv/bin/python -m tools.risk_research.codimension_three_certificate --verify .research/skeleton-risk/q3-lemmas.json --output .research/skeleton-risk/q3-rechecked.json

The tests independently integrate low-degree homogeneous propagated modes,
check all facet traces and the exact beta embedding factor, and compare
with scalar and Gram references. An altered constant, domain polynomial,
boundary cancellation or source-bound report is rejected. A fixed-state
exact q3 evaluation, its minimal order, and a uniform state bound over all
source sizes remain separate research questions.
