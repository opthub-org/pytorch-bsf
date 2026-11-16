Positive diagonal formulas for nonvertex risk
=============================================

Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`,
every nonvertex coefficient admits a positive construction using the same
univariate transformed diagonals as vertex risk.  At codimension three
these diagonals are explicit harmonic expressions.  This gives an exact
positive double sum for every source-face size, including the five-vertex
edge and six-vertex triangle, without a bivariate kernel convolution.
The number of terms still grows with the degree; no bounded classical
special-function expression, fixed-state bound or minimal order is claimed.

A Laguerre identity with arbitrary weight
-----------------------------------------

For integers :math:`L,\beta\geq0`, put

.. math::

   A_h=(-1)^h\binom Lh\binom{L+\beta}h,\qquad
   W_j=\binom Lj\binom{L+\beta}j,\qquad
   C=L!(L+\beta)!.

.. prf:lemma:: Arbitrary-weight binomial Gram diagonalization
   :label: lem-risk-nonvertex-gram-diagonal

   For arrays :math:`u,v` indexed by :math:`0,\ldots,L`, let
   :math:`\widehat u_j=\sum_{h=0}^j(-1)^h\binom jh u_h`, and similarly
   for :math:`v`.  Then

   .. math::
      :label: risk-nonvertex-gram-diagonal

      \sum_{h,l=0}^L A_hA_lh!l!(2L+\beta-h-l)!u_hv_l
       =C\sum_{j=0}^LW_j\widehat u_j\widehat v_j.

.. prf:proof::

   Define :math:`P_u(t)=\sum_{h=0}^LA_hh!u_ht^{L-h}`.  Binomial
   inversion and the finite Laguerre expansion give

   .. math::

      P_u(t)=(-1)^LL!\sum_{j=0}^L(-1)^j
           \binom{L+\beta}j\widehat u_jL_{L-j}^{(\beta)}(t).

   To check this identity, compare the coefficient of :math:`t^{L-h}`
   and use ordinary binomial inversion on :math:`u_h`; thus it also
   holds at :math:`L=0`.  Integrate :math:`P_uP_ve^{-t}t^\beta`
   on :math:`[0,\infty)`.  Its monomial moments are the factorials
   on the left of :eq:`risk-nonvertex-gram-diagonal`.  Laguerre
   orthogonality removes all unequal indices and gives

   .. math::

      L!^2\binom{L+\beta}j^2
          \frac{(L+\beta-j)!}{(L-j)!}
        =CW_j.

   This proves the identity, including all endpoints.

For a bivariate formal series :math:`G`, let

.. math::

   \mathscr T(G)(x,y)
      =\frac{G(-x/(1-x),-y/(1-y))}{(1-x)(1-y)}.

By linearity, the lemma evaluates the corresponding bivariate
functional as :math:`C\sum_jW_j[x^jy^j]\mathscr T(G)`.
Only coefficients through :math:`L` in each variable are used, so this
statement concerns finite sums and requires no analytic interchange.

One universal diagonal for every source face
---------------------------------------------

Use :math:`g(t)=-(1-t)\log(1-t)/t` and the positive sequences
:math:`\delta_q(j)` of :doc:`skeleton_risk_vertex_general`:

.. math::

   \delta_q(j)=[x^jy^j]
       \frac{(g(x)+g(y)-g(xy))^q}{(1-x)(1-y)}.

For convenience put :math:`\delta_0(j)=1`.  At every :math:`q\geq1`
the existing univariate convolution construction evaluates these
diagonals exactly, and they are strictly positive.

.. prf:theorem:: Positive nonvertex diagonal averages
   :label: thm-risk-nonvertex-positive-diagonals

   Let :math:`M\geq s\geq2`, :math:`q=M-s`, :math:`D\geq s`, and
   :math:`0\leq k\leq D-s`.  Define

   .. math::

      L&=D-s-k,&\beta&=3s+2k-1,\\
      Z_{L,\beta}&=\binom{2L+\beta}L,&
      \mathbb P(J=j)&=\frac{\binom Lj\binom{L+\beta}j}{Z_{L,\beta}}
               \quad(0\leq j\leq L).

   Every source-face angular mode of degree :math:`k` has propagated
   squared norm divided by its source norm equal to

   .. math::
      :label: risk-nonvertex-diagonal-norm

      \rho_{s,q,k,D}=\frac{(s)_q}{(2D+s)_q}\,
                              \mathbb E[\delta_q(J)].

   Consequently

   .. math::
      :label: risk-nonvertex-positive-diagonal-coefficient

      a_{s-1}(M,D)=\binom Ms^2\frac{(s)_q}{(2D+s)_q}
        \sum_{k=0}^{D-s}\binom{k+s-2}{s-2}
             \mathbb E[\delta_q(J)].

   The coefficient is zero when :math:`1\leq D<s`.

.. prf:proof::

   The radial coefficients in :prf:ref:`thm-skeleton-scalar-sum`
   are exactly :math:`A_h` with the displayed :math:`L,\beta`.
   Moreover :math:`2L+\beta=2D+s-1`, independent of :math:`k`.
   Apply :prf:ref:`lem-risk-nonvertex-gram-diagonal` to its outside
   kernel :math:`F^q`.  The formal substitution already proved in
   :doc:`skeleton_risk_vertex` gives
   :math:`F(-x/(1-x),-y/(1-y))=g(x)+g(y)-g(xy)`.
   Thus its transformed diagonal is precisely :math:`\delta_q(j)`;
   it does not depend on :math:`s,k` or :math:`D`.

   Vandermonde's identity gives :math:`\sum_jW_j=Z_{L,\beta}` and
   :math:`CZ_{L,\beta}=(2D+s-1)!`.  The remaining normalization
   in the scalar mode formula is

   .. math::

      \frac{(2D+s-1)!}{(s-1)!(M)_{2D}}
        =\frac{(s)_q}{(2D+s)_q}.

   This proves :eq:`risk-nonvertex-diagonal-norm`.  Summing the
   :math:`\binom{k+s-2}{s-2}` angular modes and multiplying by
   :math:`\binom Ms^2` proves the coefficient formula.  For
   :math:`D<s` there are no source bubbles.  For :math:`q=0`, every
   average is one and the formula recovers
   :math:`a_{s-1}(s,D)=\binom{D-1}{s-1}`.

The exact harmonic codimension-three family
--------------------------------------------

Write :math:`H_j=\sum_{r=1}^j1/r`,
:math:`H_j^{(2)}=\sum_{r=1}^j1/r^2`, with both zero at :math:`j=0`.
The tetrahedron vertex identity in :doc:`skeleton_risk_vertex` gives

.. math::
   :label: risk-nonvertex-q3-harmonic-diagonal

   \delta_3(j)=6\left(
      \frac{H_j(H_j-j/(j+2))}{(j+1)(j+2)(j+3)}
      +\frac{H_j^{(2)}}{(j+1)(j+3)}
      +\frac{2}{(j+1)^2(j+2)^2(j+3)}\right)>0.

At :math:`j=0` this is one.  At positive :math:`j`,
:math:`H_j\geq1>j/(j+2)` proves positivity term by term.

.. prf:corollary:: Every nonvertex codimension-three coefficient
   :label: cor-risk-nonvertex-codimension-three-exact

   For all integers :math:`s\geq2,D\geq s`,

   .. math::
      :label: risk-nonvertex-codimension-three-exact

      a_{s-1}(s+3,D)=
       \binom{s+3}s^2\frac{s(s+1)(s+2)}
          {(2D+s)(2D+s+1)(2D+s+2)}
       \sum_{k=0}^{D-s}\binom{k+s-2}{s-2}
       \frac{\displaystyle\sum_{j=0}^{D-s-k}
           \binom{D-s-k}j\binom{D+2s+k-1}j\delta_3(j)}
            {\binom{2D+s-1}{D-s-k}}.

   It is zero on the inactive prefix :math:`1\leq D<s`.
   In particular this evaluates :math:`a_1(5,D)` and
   :math:`a_2(6,D)` at every positive degree.

.. prf:proof::

   Substitute :math:`q=3` and
   :eq:`risk-nonvertex-q3-harmonic-diagonal` in
   :eq:`risk-nonvertex-positive-diagonal-coefficient`.
   Every denominator is a positive integer on the active domain.
   The formula has exactly :math:`(D-s+1)(D-s+2)/2` inner terms;
   harmonic tables and binomial weights have nonsingular updates.

Fixed-codimension large-degree behavior
----------------------------------------

The same positive formula determines the true leading order at every
fixed nonvertex source size and codimension at least two.

.. prf:theorem:: Nonvertex asymptotic at every fixed codimension
   :label: thm-risk-nonvertex-fixed-codimension-asymptotic

   For each fixed pair of integers :math:`s\geq2,q\geq2`,

   .. math::
      :label: risk-nonvertex-fixed-codimension-asymptotic

      a_{s-1}(s+q,D)\sim A_{s,q}D^{s-q-2},\qquad
      A_{s,q}=\binom{s+q}s^2
        \frac{(s)_q}{2^q(s-2)!}\frac{c_{q+1}}{q+1}>0.

   Here :math:`c_{q+1}` is the positive constant of
   :prf:ref:`thm-risk-general-vertex-asymptotic`.  In particular,

   .. math::
      :label: risk-nonvertex-codimension-three-asymptotic

      a_{s-1}(s+3,D)\sim
       \frac{s(s+1)^3(s+2)^3(s+3)^2}{24(s-2)!}
          (\zeta(2)-\zeta(3))D^{s-5}.

   Thus :math:`a_1(5,D)\sim3600(\zeta(2)-\zeta(3))D^{-3}` and
   :math:`a_2(6,D)\sim36000(\zeta(2)-\zeta(3))D^{-2}`.
   No finite-degree error bound or simultaneous growth of :math:`s,q,D`
   is asserted by this theorem.

.. prf:proof::

   First the univariate construction in
   :doc:`skeleton_risk_vertex_general` gives, for :math:`q\geq2`,

   .. math::

      \sum_{j\geq0}\delta_q(j)
       =\sum_{j\geq0}e_q(j)=\frac{c_{q+1}}{q+1}.

   Indeed all sequences in that construction are absolutely summable.
   The sum of :math:`\phi_q` is zero, by its generating function
   :math:`g(t)^q/(1-t)` at :math:`t=1`.  For :math:`k\geq1`,
   :math:`\sum_jb_k(j)=0`, so the sum of every convolution
   :math:`b_k*e_{q-k}` is zero.  Only the :math:`k=0` term remains.
   This also proves that the displayed sum is finite and positive.

   For summable domination we use the following exact inverse-factorial
   moment of the binomial weights.  Write :math:`B=L+\beta\geq L`.
   Vandermonde's identity gives

   .. math::

      \mathbb E\!\left[\frac1{(J+1)(J+2)}\right]
       &=\frac{(L+B+1)(L+B+2)}
              {(L+1)(L+2)(B+1)(B+2)},\\
      \mathbb E[(J+1)^{-2}]
       &\leq\frac8{(L+1)(L+2)}.

   For the first identity, replace
   :math:`\binom Bj/((j+1)(j+2))` by
   :math:`\binom{B+2}{j+2}/((B+1)(B+2))` and sum.
   For the inequality use
   :math:`(j+1)^{-2}\leq2/((j+1)(j+2))` and
   :math:`(L+B+i)/(B+i)\leq2` for :math:`i=1,2`.

   For each fixed :math:`q\geq2`, constants :math:`K,p` exist with

   .. math::

      0<\delta_q(j)\leq
           K\frac{(1+\log(j+1))^p}{(j+1)^2}\quad(j\geq0).

   At :math:`q=2`, this follows directly from
   :math:`\delta_2=-\phi_2+2\phi_1^2`, with :math:`p=1`.
   At :math:`q\geq3`, it follows from the explicit diagonal error
   :eq:`risk-general-diagonal-explicit-error`, with :math:`p=q-1`.
   Since :math:`0\leq J\leq L`, the inverse-square bound therefore gives

   .. math::

      \mathbb E[\delta_q(J)]\leq
          \frac{8K(1+\log(L+1))^p}{(L+1)(L+2)},

   a summable sequence in :math:`L`, independent of :math:`D`.

   Reindex the angular sum by :math:`L=D-s-k`.  For each fixed
   :math:`L`, :math:`B=2D+s-L-1` tends to infinity, and
   :math:`\mathbb P(J=L)=\binom BL/\binom{B+L}L\to1`.
   Thus its average tends to :math:`\delta_q(L)`.
   Also

   .. math::

      \frac{\binom{k+s-2}{s-2}}{D^{s-2}}
           \longrightarrow\frac1{(s-2)!},\qquad
      0\leq\frac{\binom{k+s-2}{s-2}}{D^{s-2}}
           \leq\frac1{(s-2)!}.

   Set terms outside :math:`0\leq L\leq D-s` to zero.  Dominated
   convergence and :math:`D^q(s)_q/(2D+s)_q\to(s)_q/2^q`
   prove :eq:`risk-nonvertex-fixed-codimension-asymptotic`.
   At :math:`q=3`, use
   :math:`c_4=48(\zeta(2)-\zeta(3))` and simplify.  At :math:`q=2`
   the theorem recovers the existing codimension-two asymptotic.

Evaluation and independent verification
----------------------------------------

``tools.risk_research.codimension_three_exact`` evaluates the harmonic
double sum using exact fractions.  Its generic nonvertex evaluator uses
the same theorem with the existing univariate :math:`\delta_q`
construction.  Neither path constructs :math:`F^q`.  The q3 path uses
quadratically many rational operations in :math:`D-s+1`, with growing
integer bit lengths.  The generic diagonal construction adds
:math:`O(q^2(D-s+1)+q(D-s+1)^2)` rational operations.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.codimension_three_exact --face-size 2 --degree 40 --output .research/skeleton-risk/q3-edge-exact.json
   $ .venv/bin/python -m tools.risk_research.codimension_three_exact --face-size 3 --degree 40 --output .research/skeleton-risk/q3-triangle-exact.json
   $ .venv/bin/python -m tools.risk_research.codimension_three_exact --face-size 2 --degree 40 --verify .research/skeleton-risk/q3-edge-exact.json --output .research/skeleton-risk/q3-edge-rechecked.json

Records bind the evaluator and its direct source dependencies.  SHA256
hashes normalize CRLF pairs to LF so that the same source record can be
checked on Windows.  The concurrent-edit guard compares the original raw
bytes and therefore still rejects line-ending edits during calculation.
Invalid inputs, changed records and calculation failures preserve the
previous destination through atomic output.  Recomputing a record checks
its exact value and provenance; the all-degree proof is the theorem above.

``nonvertex_leading_constant_enclosure(s,q,terms)`` scales the existing
vertex-constant interval by the exact positive rational factor

.. math::

   \frac{A_{s,q}}{C_{q+2}}=
     \frac{\binom{s+q}s^2(s)_q}
       {2(s-2)!(q+1)(q+2)^2(q+1)!}.

It encloses the leading constant for fixed :math:`s,q\geq2`; it does
not enclose the finite-degree coefficient by itself.

Tests compare every mode with independent polynomial moments at genuine
full-bubble degrees, recover existing q0/q1/q2 formulas, and compare small
values with the original scalar formula and separate Gram propagation.
Frozen larger cases are generated by the original integer outside-kernel
path, independently of this diagonal formula.  No finite comparison is
used in place of the all-parameter proof.
