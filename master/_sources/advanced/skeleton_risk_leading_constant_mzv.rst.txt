Finite MZV constructions for leading constants
=================================================

This chapter constructs a finite expression for every fixed-index constant
in :doc:`skeleton_risk_vertex_general`. The index convention matters:
:math:`q=M-1` at a vertex. It is not an input face codimension. The
sampling, evaluation and noise assumptions of the existing asymptotic
risk theory are retained.

.. prf:theorem:: A terminating finite convergent-MZV construction
   :label: thm-risk-leading-constant-finite-mzv

   For every fixed integer :math:`q\geq3`, the constant

   .. math::

      c_q=q\sum_{j=0}^{\infty}e_{q-1}(j)

   has an effectively constructible finite expression over the rationals
   in the unit :math:`1` and convergent multiple zeta values

   .. math::

      \zeta(2,w_1,\ldots,w_d),\qquad
      w_i\in\{1,2\},\quad 2+\sum_iw_i\leq q-1.

   Here the indices use strict descending summation:

   .. math::

      \zeta(s_1,\ldots,s_k)
       =\sum_{n_1>\cdots>n_k\geq1}
         n_1^{-s_1}\cdots n_k^{-s_k}.

   The algorithm terminates by decreasing harmonic-word depth and does
   not require assigning a value to a divergent sum. Consequently, for
   each fixed :math:`M=q+1\geq4`, the vertex leading constant is

   .. math::

      C_M=\frac{M^2q!}{2^{q-2}}c_q.

   The theorem does not assert a minimal MZV basis, linear independence,
   a classical-zeta reduction at arbitrary weight, or a bound uniform in
   growing :math:`M,D`. The rational unit is allowed in the construction;
   its cancellation at every index is not needed or asserted.

.. prf:proof::

   **Finite harmonic words.** Set :math:`H_N(\varnothing)=1`, and, for
   a nonempty word :math:`w=(s,v)`, set

   .. math::

      H_N(w)=\sum_{N\geq n_1>\cdots>n_k\geq1}
                    n_1^{-w_1}\cdots n_k^{-w_k},\qquad
      H_n(s,v)-H_{n-1}(s,v)=n^{-s}H_{n-1}(v).

   In particular :math:`H_0(w)=0` for nonempty :math:`w`.
   Write :math:`E_i` for an elementary symmetric polynomial and use
   :math:`(n)_a=n(n+1)\cdots(n+a-1)`. The proved formula
   :eq:`risk-general-phi-elementary` gives, for :math:`n\geq1`,

   .. math::

      \phi_a(n-1)=\frac{a!(a-1)!}{(n)_a}
        \sum_{l=0}^{a-1}(-1)^l
          E_{a-1-l}\left(1,\frac12,\ldots,\frac1{a-1}\right)
          H_{n-1}(\underbrace{1,\ldots,1}_{l}).

   This follows by separating the positive fixed entries from the
   negative reciprocal entries in that elementary polynomial.

   The product :math:`H_N(1^l)H_N(1^k)` has an exact finite expansion
   into words of ones and twos. For each :math:`0\leq t\leq\min(l,k)`,
   every word with :math:`t` twos and :math:`l+k-2t` ones has coefficient
   :math:`\binom{l+k-2t}{l-t}`. Indeed, :math:`t` coincident indices
   occur in both strict chains, and the remaining ones choose which
   chain they belong to. This partitions the finite product without
   taking any limit.

   Taking :math:`a+b=q-1` in the definition of :math:`e_{q-1}` therefore
   expresses :math:`c_q` as a finite rational combination of sums

   .. math::

      S(f,w)=\sum_{n\geq1} f(n)H_{n-1}(w),

   where the word entries are in :math:`\{1,2\}`, its weight is at most
   :math:`q-3`, and :math:`f` is a proper rational function whose poles
   are nonpositive integers, each of order at most two. Initially its
   denominator is :math:`(n)_a(n)_b`, so :math:`f=O(n^{-2})`.

   **Rational telescoping.** Write its exact partial fractions as

   .. math::

      f(n)=\sum_{t\geq0}\sum_{p=1}^{2}
                 \frac{c_{t,p}}{(n+t)^p}.

   Only finitely many coefficients occur. The decay :math:`O(n^{-2})`
   implies :math:`\sum_tc_{t,1}=0`. Define

   .. math::

      A=\sum_tc_{t,2},\qquad
      G(n)=\sum_{t,p}c_{t,p}\sum_{u=0}^{t-1}(n+u)^{-p}.

   Empty inner sums are zero. Telescoping the inner sums proves the
   finite rational identity

   .. math::

      f(n)=G(n+1)-G(n)+\frac{A}{n^2}.

   All its coefficients are rational. For a kernel
   :math:`\prod_t(n+t)^{-m_t}`, :math:`m_t\in\{1,2\}`, its highest
   pole coefficient at :math:`-t` is
   :math:`B_t=\prod_{u\ne t}(u-t)^{-m_u}`. If :math:`m_t=2`, its
   simple-pole coefficient is
   :math:`-B_t\sum_{u\ne t}m_u/(u-t)`. For :math:`m_t=1`, it is
   just :math:`B_t`. These formulas follow by evaluating the analytic
   factor and its derivative at :math:`-t`; matching principal parts
   determines the proper rational function. No symbolic-algebra package
   or numerical coefficient fit is necessary.

   **Decrease the word depth.** Put
   :math:`S_N(f,w)=\sum_{n=1}^Nf(n)H_{n-1}(w)`. Finite summation by
   parts gives, for nonempty :math:`w=(s,v)`,

   .. math::

      S_N(f,w)={}&A H_N(2,w)+G(N+1)H_N(w)-G(1)H_0(w)\\
                 &+S_N(h,v),\qquad
      h(n)=-\frac{G(n+1)}{n^s}.

   For the empty word the last sum is absent and the boundary at one
   is :math:`-G(1)`. Since :math:`G(n)=O(n^{-1})` and :math:`s\geq1`,
   :math:`h=O(n^{-2})`. Its poles again have order at most two: those
   from :math:`G(n+1)` have shifts at least one, while :math:`n^{-s}`
   has shift zero and :math:`s\in\{1,2\}`. The rational class is thus
   invariant and the remaining word is strictly shorter.

   For any such word of depth :math:`d`, positivity and comparison with
   all-one words show

   .. math::

      0\leq H_N(w)\leq\frac{(1+\log N)^d}{d!}\quad(N\geq1).

   Every sum :math:`S(f,w)` is absolutely convergent, and all upper
   boundary terms tend to zero, bounded by a constant times
   :math:`(1+\log N)^d/N`. Taking limits in the finite identity yields
   :math:`A\zeta(2,w)`, a rational lower boundary when appropriate,
   and a strictly shorter instance of the same problem. After at most
   the original depth plus one instances the recursion stops. Each
   output MZV is convergent and has weight at most :math:`q-1`.

   Finally the original positive integral identifies this expression
   with the positive constant :math:`c_q`; the stated normalization
   of :math:`C_M` is the existing fixed-vertex normalization.

Rational enclosures and symbolic work limits
---------------------------------------------

The standard-library tool ``tools.risk_research.leading_constant_mzv``
implements these finite algebraic steps using ``Fraction``. For example,

.. doctest::

   >>> from fractions import Fraction
   >>> from tools.risk_research.leading_constant_mzv import leading_constant_expression
   >>> leading_constant_expression(3) == {(2,): Fraction(6)}
   True
   >>> leading_constant_expression(4) == {(2,): Fraction(48), (2, 1): Fraction(-48)}
   True

At indices five and six the resulting expressions agree with the
corresponding finite harmonic telescopes. Using the established
low-weight identities, the fifth-index expression gives
:math:`900\zeta(4)-600\zeta(3)-120\zeta(2)`. These low-index
checks are calibration, not a proof for arbitrary index.

For a convergent word :math:`(a,w)` with :math:`a\geq2` and
:math:`d=\operatorname{depth}(w)`, a direct numerical enclosure is

.. math::

   H_T(a,w)\leq\zeta(a,w)\leq H_T(a,w)
    +\frac1{T^{a-1}}\sum_{k=0}^{d}
       \frac{(2+\log T)^k}{k!(a-1)^{d-k+1}},\qquad T\geq1.

To prove the tail, bound the inner word by
:math:`(1+\log n)^d/d!` and integrate on :math:`[n-1,n]`.
There :math:`n^{-a}\leq x^{-a}` and
:math:`1+\log n\leq2+\log x`. Repeated integration by parts gives
the displayed expression. Replacing :math:`2+\log T` with
:math:`2+\operatorname{bit\_length}(T)` makes it rational.
Signed coefficients choose the appropriate interval endpoints.
Positivity from the original integral permits intersecting the final
interval with the nonnegative half-line.

The enclosure can be very broad because this representation can have
large cancellations and its generic MZV tails decay slowly. No numerical
speed or accuracy advantage over the original defining series is claimed.
The finite symbolic expression, its numerical truncation uncertainty,
and the original independent evaluation are distinct objects.

Expression size, arithmetic count and integer bits
--------------------------------------------------

These are three separate costs. The following bounds concern the
symbolic coefficient construction, before numerical MZV truncation,
and disregard the implementation's operational cutoff.

**Expression size.** Let :math:`F_0=0`, :math:`F_1=1` and
:math:`F_{j+2}=F_{j+1}+F_j`. There are :math:`F_{m+1}` words over
:math:`\{1,2\}` of weight :math:`m`, including the empty word at weight
zero. Hence there are

.. math::

   \sum_{m=0}^{q-3}F_{m+1}=F_q-1

possible nonconstant terms :math:`\zeta(2,w)`, and at most :math:`F_q`
terms after allowing the rational unit. Every index tuple has length
at most :math:`q-2`. This bounds the number and lengths of the stored
words, not their linear independence or the bits needed for coefficients.

**Arithmetic operations.** Put :math:`Q=q+1`. Count an integer or
rational addition, subtraction, multiplication or division as one
operation, including negation, and evaluate factorials, binomial
coefficients and nonnegative integer powers by elementary products and
divisions. Loop control, work-counter bookkeeping, dictionary operations,
tuple construction, sorting and GCD normalization are outside this
arithmetic model. A deliberately
loose bound for the symbolic construction is

.. math::

   A_q=128Q^8 2^q.

Here is a loop count that also accounts for recomputation in the current
implementation. ``_initial`` has at most :math:`Q^3` choices of
``left``, ``a`` and ``b``. Each ``_ones_product`` enumerates at most
:math:`2^q` words, and each initial kernel has at most :math:`2Q`
partial-fraction entries. ``_phi_polynomial`` uses at most :math:`Q^2`
inner iterations and is recomputed for each ``a`` on the right side.
Together these cost at most :math:`64Q^6 2^q` arithmetic operations.
Throughout ``_transform``, pole shifts remain below :math:`q` and pole
orders remain at most two. Thus kernels and antiderivatives each have
at most :math:`2Q` entries. There is one child per recursive call,
at most :math:`Q` levels per initial word, and at most :math:`Q` terms
to merge per level. Telescoping and division by the next harmonic index,
including the two-factor partial fractions, therefore cost at most
:math:`64Q^5 2^q` for all initial words and the final accumulation.
The sum is below :math:`A_q`. This is a unit-cost algebraic bound,
not a count of Python instructions, GCD steps or elapsed seconds.

**Integer bits.** A separate bound follows by tracking numerator and
denominator lengths. All integer inputs to the coefficient arithmetic,
including the factorial and binomial values, fit within
:math:`B_0=4Q^2` bits. For example, :math:`q!\leq q^q` and
:math:`\binom{q}{j}\leq2^q`. If both rational operands have numerator
and denominator lengths at most :math:`B`, unreduced addition or
subtraction has lengths at most :math:`2B+1`; multiplication and
division need at most :math:`2B`. Reduction cannot increase these
lengths. Iterating :math:`B\mapsto2B+1` therefore gives the explicit
coarse bound

.. math::

   B_q=(4Q^2+1)2^{A_q}-1

for coefficient numerators and denominators, including intermediate
rational results in this model. It is intentionally enormous. It is
an existence bound, not a polynomial bit-complexity claim or a bound on
Python's internal temporary buffers, CPU time or peak resident memory.
The term-count bound alone gives no such bit or memory guarantee.

Small-index observations can be reproduced independently of these
coarse all-index bounds:

.. doctest::

   >>> rows = []
   >>> for q in range(3, 11):
   ...     expression = leading_constant_expression(q)
   ...     rows.append((q, len(expression),
   ...                  max(abs(c.numerator).bit_length() for c in expression.values()),
   ...                  max(c.denominator.bit_length() for c in expression.values())))
   >>> rows
   [(3, 1, 3, 1), (4, 2, 6, 1), (5, 4, 10, 1), (6, 7, 14, 1), (7, 12, 19, 1), (8, 20, 24, 2), (9, 33, 29, 3), (10, 54, 34, 4)]

Each row reports ``(q, terms, maximum numerator bits, maximum denominator
bits)`` for the final nonzero coefficients. These finite observations
do not bound intermediate coefficients or establish a growth law.

Numerical truncation is an additional cost. With prefix length
:math:`K`, ``_harmonic_values`` collects at most :math:`QF_q` distinct
suffixes and performs one division and one addition for each nonempty
suffix at each of :math:`K` steps. The powers in these constructed words
are at most two. This gives at most :math:`3KQF_q` arithmetic operations
for that update loop when the possible squaring is included. Tail
formation and interval accumulation add further operations. The bound
:math:`B_q` above covers symbolic coefficients only; numerical prefix
denominators and their arithmetic also depend on :math:`K`.

Selected symbolic and numerical steps consume ``_Budget`` work units,
with a default limit of 200,000 and an operational maximum of 1,000,000.
This counter charges selected loops and some word lengths; it omits
other arithmetic, factorial/binomial internals, GCD work and integer
bit costs. It is neither :math:`A_q` nor an elapsed-time or memory
budget. Exceeding the work limit raises a controlled error without
returning a partial expression. The mathematical theorem is not
restricted to the indices that fit that operational budget, and the
tool makes no efficiency or minimal-basis claim.

Records and verification scope
------------------------------

.. code-block:: console

   python -m tools.risk_research.leading_constant_mzv --constant-index 6 --terms 37 --work-limit 180000 --output .research/skeleton-risk/mzv.json
   python -m tools.risk_research.leading_constant_mzv --verify .research/skeleton-risk/mzv.json

The new record stores all generation options, ordered exact coefficients,
the index convention, signed intervals, normalization and normalized
source hashes. Verification restores nondefault parameters and compares
the entire typed record. It does not change the existing three schemas
handled by ``saved_records``. Duplicate keys, extra fields, changed types,
changed coefficients or sources are rejected.

The output directory is managed by the caller. The writer fixes the
canonical parent and original leaf and protects evaluator/proof sources;
it does not provide a sandbox against replacement of the canonical
directory inode by another process.

After freezing the algebra and runtime, unused indices and finite
prefixes are checked against the original elementary-symmetric series
and positive integral, with source, environment and input preservation.
Finite checks verify implementation and normalization; they do not
replace the all-index termination and convergence proof above. The
result concerns leading constants as the degree tends to infinity with
vertex count fixed, not a finite-sample risk or a finite-degree remainder.
