Uniform construction for codimension two
=========================================

The previously evaluated edges, triangular faces and tetrahedral faces have
source-face sizes two, three and four. This chapter gives one construction
for every fixed source-face size. It uses the uniform-simplex assumptions of
:doc:`skeleton_risk_theory` and the mode formula in
:doc:`skeleton_risk_extensions`.

.. prf:theorem:: A uniform fifteen-state codimension-two construction
   :label: thm-risk-uniform-codimension-two-state

   For every integer :math:`s\geq2`, the coefficient
   :math:`a_{s-1}(s+2,D)` has the following constructive evaluation for every
   integer :math:`D\geq1`. It is zero if :math:`D<s`. On the active domain,
   set

   .. math::

      h=D+\lfloor s/2\rfloor,\qquad
      \epsilon=1+(s\bmod2),\qquad N=2h+\epsilon=2D+s+1.

   A fifteen-component state, independent of :math:`s`, has a rational
   transition from :math:`h` to :math:`h+1`. A finite algorithm depending on
   :math:`s` constructs a rational output row whose product with this state
   is :math:`a_{s-1}(s+2,D)`. The algorithm uses polynomial arithmetic,
   partial fractions with poles of order at most two, and at most
   :math:`\lfloor s/2\rfloor+1` explicit endpoint corrections. Its operation
   count for constructing the output row at fixed :math:`s` is independent
   of :math:`D`.

   State initialization uses only the fixed values :math:`h=1,N=3` or
   :math:`h=1,N=4`. State transitions are nonsingular for :math:`h\geq1`;
   output construction is nonsingular throughout :math:`D\geq s`.

This is an upper bound on state dimension, not a claim of minimality. The
eight-state formulas at :math:`s=3,4` remain useful smaller representations.
At :math:`s=1` the coefficient is the already evaluated triangle vertex;
the polynomial angular multiplicity used below applies to :math:`s\geq2`.
No production sampling API changes are involved.

The common state
-----------------

Let :math:`b_j=\binom Nj^{-1}`, and let :math:`H_r,H_r^{(2)}` have their
usual values, including zero at :math:`r=0`. Define

.. math::

   F_N&=\sum_{j=1}^N\frac{b_j}{j^2},&
   G_N&=\sum_{j=1}^N\frac{b_j}{j},&
   P_N&=\sum_{j=0}^N\frac{b_j}{j+1},\\
   T_1&=\sum_{j=1}^h\frac{b_j}{j},&
   T_2&=\sum_{j=1}^h\frac{b_j}{j^2},&
   U&=\sum_{j=1}^h\frac{b_j}{j+1},\qquad w=b_h.

The state is

.. math::
   :label: risk-uniform-codimension-two-state

   \bigl(H_N^2,H_NH_h,H_h^2,H_N,H_h,H_N^{(2)},H_h^{(2)},
         F_N,G_N,P_N,T_1,T_2,U,w,1\bigr)^{\mathsf T}.

The defining sums establish the identities below; evaluation advances the
fixed state and does not recompute them at each degree.

Write :math:`q=(N+1)(N+2)` and

.. math::

   W=\frac{(h+1)(N-h+1)}q\,w,\qquad
   S=\frac{N+1}{4}G_N-\frac12
       -\mathbf1_{\epsilon=2}\frac{h+1}{2(h+2)}w,\qquad
   J=\frac N2 S+\frac{(h+1)w-1}{2}.

Here :math:`S=\sum_{j=1}^h b_j` and
:math:`J=\sum_{j=1}^h j b_j`. The half-range updates are

.. math::
   :label: risk-uniform-half-transitions

   T_1'&=T_1-\frac{2N+3}{q}S+\frac Jq+\frac W{h+1},\\
   T_2'&=T_2-\frac{2N+3}{q}T_1+\frac Sq+\frac W{(h+1)^2},\\
   U'&=\frac{N+3}{N+1}U-\frac{2S}{N+1}+\frac Jq+\frac W{h+2},
   \qquad w'=W.

Primes denote :math:`N'=N+2,h'=h+1`. The full inverse-binomial updates
can be performed twice using

.. math::

   G_{N+1}&=G_N/2+1/(N+1),\\
   F_{N+1}&=F_N-G_N/(N+1)+1/(N+1)^2,\\
   P_{N+1}&=\frac{N+2}{N+1}P_N-\frac{G_N}{2}
                              -\frac1{N+1}+\frac1{N+2}.

Finally, :math:`H_N` increases by
:math:`a=1/(N+1)+1/(N+2)`, and :math:`H_h` by :math:`c=1/(h+1)`.
Their squares and product therefore have affine updates in the listed
quadratic basis. The second-order harmonic numbers increase by
:math:`1/(N+1)^2+1/(N+2)^2` and :math:`1/(h+1)^2`, respectively.
Every displayed denominator is strictly positive on the stated state domain.

.. prf:proof::

   The first two full inverse-binomial updates are already proved in
   :doc:`skeleton_risk_extensions`. The ratios

   .. math::

      \frac{b_{N+1,j}}{b_{N,j}}=\frac{N+1-j}{N+1},\qquad
      \frac{b_{N+2,j}}{b_{N,j}}
          =\frac{(N+1-j)(N+2-j)}q

   give the :math:`P_N` update and the three half-range updates by polynomial
   division by :math:`j,j^2,j+1`; the new endpoint is exactly :math:`W`.
   To express the zero-th and first half moments, observe

   .. math::

      \frac{N+1}{j}b_j=b_j+b_{j-1},\qquad
      (2j-N)b_j=(N-j)b_{j+1}-(N-j+1)b_j.

   Summing the first identity over :math:`1\leq j\leq N` gives
   :math:`\sum_{j=0}^N b_j=(N+1)G_N/2+1`. Symmetry and, when
   :math:`\epsilon=2`, subtraction of the middle term give :math:`S`.
   The second identity telescopes to the displayed expression for :math:`J`.
   This proves all the transitions and their initialization by the fixed
   defining sums at :math:`h=1`.

From arbitrary face size to polynomial moments
----------------------------------------------

Set

.. math::

   x&=j(N-j),\qquad \Delta=x(x+N+1),\\
   B_j&=H_{j+1}+H_{N-j+1}-H_N,\\
   g_N(j)&=\frac{B_j}{\Delta}
       -\frac{(N+1)(x-1)}{\Delta^2}
       -\frac{b_j}{x\Delta}.

Reindexing :eq:`risk-codimension-two-coefficient` by :math:`j=L+1`
and using :math:`d_s(k)=\binom{k+s-2}{s-2}` gives

.. math::
   :label: risk-uniform-polynomial-angular-weight

   a_{s-1}(s+2,D)
      =K_s\sum_{j=1}^{D-s+1}p(j)g_N(j),\quad
   K_s=\binom{s+2}s^2\,2s(s+1),\quad
   p(j)=\binom{D-j-1}{s-2}.

In the reduction, :math:`p` is the polynomial binomial product, including
its values at negative upper arguments. Extending the sum to :math:`h`
adds zero-weight terms at :math:`D-s+2,\ldots,D-1` and the explicit terms
at :math:`j=D+t`, :math:`0\leq t\leq\lfloor s/2\rfloor`, with weights

.. math::

   p(D+t)=(-1)^{s-2}\binom{t+s-2}{s-2}.

For such an endpoint, write :math:`j=h-r`. Its two harmonics differ from
:math:`H_h` by a fixed number of increments, while

.. math::

   b_{h-r}=w\prod_{i=1}^r\frac{N-h+i}{h-i+1}.

Consequently every correction is a rational row in the same state. On the
active domain :math:`h\geq s+\lfloor s/2\rfloor`, none of these denominators
can vanish.

Reduction of the harmonic part
------------------------------

The rational function :math:`p(j)/\Delta` has a polynomial part and only
simple poles at :math:`0,-1,N,N+1`. The pole part is a linear combination of

.. math::

   \left(\frac1j\mathbin\pm\frac1{N-j}\right)B_j,
   \qquad
   \left(\frac1{j+1}\mathbin\pm\frac1{N-j+1}\right)B_j.

For the symmetric parts, full-range summation and symmetry give the
half-range sums. The necessary full sums are

.. math::

   \sum_{j=1}^{N-1}\frac{B_j}{j}
      &=\frac{H_N^2-H_N^{(2)}}2+1-\frac1N
          +\frac{2H_N-1-1/N}{N+1},\\
   \sum_{j=1}^{N-1}\frac{B_j}{j+1}
      &=\frac{H_N^2-H_N^{(2)}}2
        +2\left(\frac1{N+1}+\frac1{N+2}\right)H_N
        +\frac2{(N+1)(N+2)}-\frac2{N+1}-1.

These follow from
:math:`\sum_{k=1}^r H_k/k=(H_r^2+H_r^{(2)})/2` and
:math:`\sum_{k=1}^{r-1}H_{r-k}/k=H_r^2-H_r^{(2)}`.
When :math:`N` is even, subtract the central summand before halving
the full symmetric sum. The central harmonic is
:math:`2H_{h+2}-H_N`, so this uses the listed state.

For an arbitrary constant :math:`H`, define

.. math::

   A(C,L;H)={}&\frac12\left(H_L^2+H_L^{(2)}
      -H_{C-1}^2-H_{C-1}^{(2)}
      +H_{C-L-1}^2+H_{C-L-1}^{(2)}\right)\\
      &+H_LH_{C-L-1}-H(H_L-H_{C-1}+H_{C-L-1}).

The following prefix identity handles both antisymmetric parts:

.. math::
   :label: risk-uniform-antisymmetric-prefix

   A(C,L;H)=\sum_{j=1}^L
        \left(\frac1j-\frac1{C-j}\right)(H_j+H_{C-j}-H).

Indeed both sides vanish at :math:`L=0`, and their differences at successive
:math:`L` agree by :math:`H_L-H_{L-1}=1/L`. In particular, the cross terms
telescope using

.. math::

   \sum_{j=1}^L\left(\frac{H_{C-j}}j-\frac{H_j}{C-j}\right)
        =H_LH_{C-L-1}.

For poles at :math:`j+1,N-j+1`, take :math:`C=N+2`,
:math:`L=h+1`, :math:`H=H_N`, and remove the :math:`j=1` term.
For poles at :math:`j,N-j`, take :math:`C=N,L=h,H=H_N`, and add the
rational correction

.. math::

   \sum_{j=1}^h\frac{(N+2)(N-2j)}\Delta.

This correction changes :math:`H_j+H_{N-j}` into
:math:`H_{j+1}+H_{N-j+1}`. At the half-range endpoints, every harmonic index
in these formulas is :math:`N` or :math:`h` plus a fixed integer. Hence
the formulas use only the quadratic harmonic components of the state.

It remains to sum :math:`Q(j)B_j` for a polynomial :math:`Q`. Construct a
polynomial :math:`R` with :math:`R(0)=0` and
:math:`R(j)-R(j-1)=Q(j)`. Its coefficients follow by triangular elimination,
dividing only by positive integers. Summation by parts gives

.. math::

   \sum_{j=1}^h Q(j)B_j=R(h)B_h
      -\sum_{j=1}^{h-1}R(j)
          \left(\frac1{j+2}-\frac1{N-j+1}\right).

The remaining sums are rational, with fixed simple poles. Their partial
fractions evaluate using harmonic numbers and polynomial antidifferences.
This completes the harmonic reduction for every polynomial weight.

Rational and inverse-binomial parts
-----------------------------------

The rational part :math:`p(j)(N+1)(x-1)/\Delta^2` has only the same four
poles, of order at most two. Its sum therefore uses only polynomials,
first-order harmonics and second-order harmonics.

The inverse-binomial part has denominator
:math:`j^2(j+1)(N-j)^2(N-j+1)`. After partial fractions, its proper part
uses :math:`T_1,T_2,U` and the reflected half sums. Reflection expresses
these through :math:`G_N,F_N,P_N`, the same half sums and, for even
:math:`N`, the central term. For example,

.. math::

   \sum_{j=1}^h\frac{b_j}{N-j}
     =G_N-\frac1N-T_1
        -\mathbf1_{\epsilon=2}\frac{w}{h+2}.

For a polynomial part :math:`Q`, introduce the operator

.. math::

   (\mathcal LR)(j)=(j+1)R(j+1)-(N-j+1)R(j).

If :math:`R(j)=j^k`, its leading term is :math:`2j^{k+1}`. Triangular
elimination thus gives, for every polynomial :math:`Q`, a polynomial
:math:`R` and a constant :math:`c` with
:math:`Q=\mathcal LR+c`. The ratio :math:`b_{j+1}/b_j=(j+1)/(N-j)` yields

.. math::

   \sum_{j=1}^h Q(j)b_j
      =cS+(h+1)wR(h+1)-R(1).

Thus even arbitrarily high polynomial moments introduce no new state.
Polynomial degrees grow with :math:`s`; the number of residual sequences
does not.

Completion of the construction and its scope
---------------------------------------------

.. prf:proof::

   Apply the three reductions above to the polynomial in
   :eq:`risk-uniform-polynomial-angular-weight`, subtract its finite endpoint
   corrections, and multiply by :math:`K_s`. Every step is finite polynomial
   arithmetic or an explicit partial-fraction identity. Harmonic products
   are at most quadratic in :math:`H_N,H_h`; all inverse-binomial residuals
   are in the listed state. This constructs the rational output row.

   Partial fractions divide only by differences of the four pole locations,
   hence by nonzero powers of :math:`N,N+1,N+2` and constants. Polynomial
   antidifferences divide by positive integers; inverse-binomial polynomial
   reduction divides by two. The endpoint increments divide by positive
   shifts of :math:`h` on the active domain. All formulas are therefore
   nonsingular. The inactive prefix is explicitly zero. Together with the
   established rational transitions this proves the theorem for every
   :math:`s\geq2`, rather than only the finitely many tested face sizes.

The construction is implemented by
``tools.risk_research.codimension_two_family`` using only standard-library
exact fractions. ``family_coefficients(s, maximum)`` returns a complete
degree prefix, while ``family_coefficient(s, D)`` advances the state and
constructs only the requested output row. The number of rational operations
per degree at fixed :math:`s` is bounded; numerator and denominator bit
lengths still increase with :math:`D`.

The fifteen rational output rows obtained by advancing the state, together
with one further row, are linearly dependent over :math:`\mathbb Q(D)`.
They therefore also give a constructive scalar recurrence of order at most
fifteen. This chapter supplies the common state construction, not a claim
that the scalar order is minimal or that an eliminated leading coefficient
has no exceptional indices. An explicit scalar operator must check those
indices separately.

Reproduction and independent verification
------------------------------------------

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.codimension_two_family --face-size 5 --max-degree 50 --output .research/skeleton-risk/family-m7.json

The optional native-Sage construction verifier checks the emitted rational
identities over :math:`\mathbb Q(h)(j)`, polynomial antidifferences, boundary
weights and positivity of all output and transition denominators. It also
uses a separate dense linear solve for polynomial inverse-binomial reduction.

.. code-block:: console

   $ python -m tools.risk_research.codimension_two_family_certificate --face-size 6 --output .research/skeleton-risk/family-m8-proof.json

Run that command in the prepared Sage environment. The report records the
output row, both source hashes and the exact residuals. Native regressions
are selected by ``BSF_SAGE_PYTHON`` as for the earlier recurrence verifiers.

Independent tests compare all fifteen transitions to the defining sums in
both parities, polynomial kernel moments through degree eight, original
scalar coefficients and low-degree Gram propagation, and the original
angular formula at larger degrees and new face sizes. The committed fresh
holdout artifact includes faces larger than the first construction examples.
These finite tests check the implementation; the reductions above establish
the all-size, all-degree theorem.
