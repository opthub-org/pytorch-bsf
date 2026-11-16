Coefficient heights in the finite MZV construction
==================================================

The finite construction in :doc:`skeleton_risk_leading_constant_mzv`
already applies to every fixed integer index :math:`q\geq3`.
This chapter bounds its symbolic rational coefficients, including stored
intermediate rational results. Numerical MZV prefixes are a separate cost.
The algorithm and its existing operational work limit are unchanged.

.. prf:theorem:: A common denominator and polynomial coefficient-bit bound
   :label: thm-risk-leading-constant-height

   Put :math:`Q=q+1`, :math:`L=q!`, and let
   :math:`b=\operatorname{bit\_length}(q)`. Throughout the coefficient
   arithmetic of ``leading_constant_mzv._construct(q, budget)``, every
   stored rational denominator divides :math:`L^{5Q}`. Every stored
   rational numerator and denominator has at most

   .. math::

      B_q=5qQb+8Q(b+1)+2

   bits, where the sign is excluded and zero has bit length zero.
   The unreduced integer products and sums for a single rational
   addition, subtraction, multiplication or nonzero division have at
   most :math:`2B_q+1` bits in the elementary arithmetic model.

   This does not bound the number of words by a polynomial: the previous
   :math:`F_q` term bound and :math:`128Q^8 2^q` unit-cost arithmetic
   bound still apply. It does not bound Python instruction counts,
   internal GCD/factorial/binomial buffers, peak RSS, or the rational
   arithmetic of numerical MZV prefixes.

.. prf:proof::

   **Initial factors.** All shifts are nonnegative integers below
   :math:`q`, and pole orders are one or two. Differences of distinct
   shifts are nonzero integers of absolute value at most :math:`q`.
   In ``_partial_fractions`` the highest-pole coefficient is a product
   of at most :math:`2q` such reciprocal factors. Its denominator divides
   :math:`L^{2q}` and its absolute value is at most one. The logarithmic
   derivative is a sum of at most :math:`q` terms of size at most two,
   with denominator dividing :math:`L`. Thus every initial partial
   fraction, including the simple coefficient of a double pole, has
   denominator dividing :math:`L^{2q+1}` and absolute value at most
   :math:`2q`. These statements also hold during the partial products
   and partial sums.

   Each intermediate elementary symmetric polynomial in
   ``_phi_polynomial(a)`` has denominator dividing :math:`(a-1)!`.
   A term is a reciprocal product of a subset of the distinct integers
   :math:`1,\ldots,a-1`, and its denominator divides that factorial.
   At the end, multiplication by :math:`a!(a-1)!` gives an integer.
   The sum of the absolute coefficients is

   .. math::

      a!(a-1)!\prod_{j=1}^{a-1}(1+1/j)=(a!)^2\leq L^2.

   The finite product-of-harmonic-words coefficient is a binomial
   integer at most :math:`2^q`. So is :math:`\binom{q-1}{\ell}`.
   For each fixed word and pole, ``_initial`` adds at most :math:`q^3`
   contributions. All its accumulated coefficients and partial sums
   therefore have denominator dividing :math:`L^{2q+1}` and absolute
   value at most

   .. math::

      I_q=2q^5 4^q L^4.

   The elementary-polynomial intermediates and integer prefactors also
   satisfy this absolute bound.

   **Recursion.** Suppose the current pole coefficients have common
   denominator dividing :math:`L^E` and absolute value at most :math:`U`.
   ``_telescope`` changes no denominator. Its residual and each
   antiderivative coefficient sum at most :math:`q` coefficients, giving
   absolute bound :math:`qU`.

   ``_shift_divide`` multiplies these by partial fractions with exactly
   two poles, at zero and :math:`h\in\{1,\ldots,q\}`, both of order at
   most two. Directly from the partial-fraction formulas, their
   denominators divide :math:`h^3`, and their absolute values are at most
   two. Consequently the new denominator divides :math:`L^{E+3}`.
   There are at most :math:`2q` antiderivative entries contributing to
   each new coefficient, so its absolute value, including partial
   accumulations, is at most :math:`4q^2U`.

   Each recursive call removes the first harmonic index. The initial
   word length is at most :math:`q-3`. At the empty word, the final
   boundary divides antiderivative coefficients by :math:`(1+h)^p`,
   where :math:`1+h\leq q` and :math:`p\leq2`, adding at most two to
   the denominator exponent and giving absolute bound :math:`2q^2U`.
   Hence throughout a branch the denominator exponent is at most
   :math:`2q+1+3(q-3)+2=5q-6<5Q`.

   With :math:`T_q=8q^3`, the bound
   :math:`I_qT_q^{q+1}` covers all branch coefficients, residuals,
   boundaries and intermediate products. Residual keys within a branch
   have strictly decreasing weights, so copying the recursively returned
   dictionary does not add different levels to the same key. Finally,
   there are fewer than :math:`2^q` initial words. Even a partial final
   accumulation has absolute value at most

   .. math::

      H_q=2^q I_q(8q^3)^{q+1}.

   Since :math:`\log_2L\leq qb` and
   :math:`\log_2H_q\leq8Q(b+1)`, a reduced denominator is at most
   :math:`2^{5qQb}`, and a numerator is at most this denominator times
   :math:`H_q`. The displayed :math:`B_q` includes spare bits for zero
   and equality at a power of two. Rational reduction cannot increase
   either length. A single elementary rational operation on operands
   with at most :math:`B_q` bits uses integer products of at most
   :math:`2B_q` bits and their sums of at most :math:`2B_q+1` bits.

The old generic bound was
:math:`(4Q^2+1)2^{128Q^8 2^q}-1`. The new bound is strictly smaller
for every :math:`q\geq3`: using :math:`b\leq q`,
:math:`B_q\leq16Q^3`, whereas the old exponent alone exceeds
:math:`128Q^8`, hence exceeds :math:`16Q^3`.
This bounds bits of each coefficient, not the exponential expression
size or the bit complexity of the whole construction.

Records and reproducibility
----------------------------

``tools.risk_research.leading_constant_height`` computes the integer
bounds without constructing the enormous old power. Its saved record
also constructs the original finite expression with the original
selected-work budget and reports the largest *final* coefficient bits.
Those measurements are not measurements of intermediate coefficients;
the theorem above covers intermediates. The final denominator-divisibility
check is an implementation check, not a proof of the theorem.

Records bind the source and proof bytes, restore the index and work
limit, and compare the complete typed JSON record. The mathematical
index domain has no upper bound; operational budgets can reject an
input without returning a partial record. Numerical intervals and
finite-degree errors are outside this record's scope.
