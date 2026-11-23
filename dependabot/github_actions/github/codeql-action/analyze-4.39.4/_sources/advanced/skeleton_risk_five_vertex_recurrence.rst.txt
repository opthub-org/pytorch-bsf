Five-vertex vertex degree recurrence
=====================================

Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`, the
harmonic single sum in :doc:`skeleton_risk_five_vertex` admits an explicit
scalar recurrence. The construction uses eight rational telescoping
identities. It does not fit finitely many coefficient values.

.. prf:theorem:: Certified five-vertex vertex degree recurrence
   :label: thm-risk-five-vertex-recurrence

   Put :math:`a_D=a_0(5,D)`. The artifact
   ``tools/risk_research/examples/five_vertex_vertex_recurrence_order17.json``
   gives integer polynomials :math:`p_0,\ldots,p_{17}`, each of degree 188,
   such that

   .. math::

      \sum_{i=0}^{17}p_i(D)a_{D+i}=0\qquad(D=1,2,\ldots).

   Every coefficient of :math:`p_{17}` is strictly positive. There are
   no exceptional positive integer indices. The artifact supplies the
   seventeen initial values :math:`a_1,\ldots,a_{17}`, starting with
   :math:`5/3,5/8,5/24`, so the recurrence determines every degree.
   Minimality of its order or polynomial degree is not asserted.

The coefficient arrays are in ascending powers of :math:`D`. They are
stored in full rather than reproduced as thousands of large integers
in this chapter. The following certificate construction independently
checks the full arrays over :math:`\mathbb Q(D)`.

The triangular harmonic state
-------------------------------

Use the notation from :doc:`skeleton_risk_five_vertex` and order the row
of harmonic monomials as

.. math::

   T(j)=\left(H_j^3,H_j^2,H_jH_j^{(2)},H_j^{(3)},A_j,
                  H_j,H_j^{(2)},1\right)^{\mathsf T}.

With :math:`t=(j+1)^{-1}`, its shift matrix :math:`S(j)` is defined by
the following increments of its eight components:

.. math::

   T(j+1)-T(j)=\left(
      3tH_j^2+3t^2H_j+t^3,\;
      2tH_j+t^2,\;
      tH_j^{(2)}+t^2H_j+t^3,\;
      t^3,\;t^2H_j,\;t,\;t^2,\;0
   \right)^{\mathsf T}.

Thus :math:`T(j+1)=S(j)T(j)`. Write the known diagonal weight as
:math:`\Psi_j=b(j)T(j)`; the eight rational entries of :math:`b` are
exactly :eq:`risk-five-vertex-diagonal` in this ordering.

Normalization to the largest shift
------------------------------------

For a candidate order :math:`r`, set :math:`N=D+r-1` and

.. math::

   w_{N,j}&=\binom Nj\binom{N+2}j,\\
   U_m(j)&=(N-j)^{\underline m}(N+2-j)^{\underline m},\\
   B_m&=N^{\underline m}(N+2)^{\underline m},\\
   P(j)&=\sum_{m=0}^{r}\lambda_m(N)U_m(j).

Falling factorials use a nonnegative integer number of factors. The
weight for degree :math:`D+i`, divided by :math:`w_{N,j}`, is
:math:`U_{r-i}(j)/B_{r-i}`. This identity also respects the different
supports: :math:`U_m` vanishes at the extra upper indices.

Put :math:`A(j)=(N-j)(N+2-j)`. A row certificate :math:`c(j)` must satisfy

.. math::
   :label: risk-five-vertex-certificate-row

   \frac{A(j)}{(j+1)^2}c(j+1)S(j)-c(j)=P(j)b(j).

Multiplication by :math:`w_{N,j}T(j)` then gives a telescoping difference.

Eight polynomial equations
----------------------------

Choose :math:`c_i(j)=j^2Q_i(j)/E_i(j)`, with denominator types

.. math::

   E_{\mathrm L}(j)&=(j+1)(j+2)(j+3),\\
   E_{\mathrm M}(j)&=(j+1)^2(j+2)^2(j+3),\\
   E_{\mathrm H}(j)&=(j+1)^3(j+2)^2(j+3).

The types in state order are
:math:`(\mathrm L,\mathrm M,\mathrm L,\mathrm L,\mathrm L,
\mathrm M,\mathrm M,\mathrm H)`. Let
:math:`\Omega_i=\operatorname{lcm}(E_i(j),E_i(j+1))`.
After the previous :math:`Q_l` are known, equation :math:`i` of
:eq:`risk-five-vertex-certificate-row` becomes

.. math::

   \mathcal L_iQ_i=\Omega_i\left(Pb_i
      -A\sum_{l<i}\frac{Q_l(j+1)}{E_l(j+1)}S_{li}(j)\right).

The three polynomial difference operators are

.. math::

   \mathcal L_{\mathrm L}Q
    &=A(j)(j+1)Q(j+1)-j^2(j+4)Q(j),\\
   \mathcal L_{\mathrm M}Q
    &=A(j)(j+1)^2Q(j+1)-j^2(j+3)(j+4)Q(j),\\
   \mathcal L_{\mathrm H}Q
    &=A(j)(j+1)^3Q(j+1)-j^2(j+2)(j+3)(j+4)Q(j).

On :math:`j^m`, their leading degrees are :math:`m+2,m+3,m+4` and
their leading coefficients are :math:`m-2N-5,m-2N-7,m-2N-8`.
Triangular polynomial reduction therefore leaves at most
:math:`(2,3,2,2,2,3,3,4)` remainder coefficients, twenty-one in total.

Reduce :math:`U_0,\ldots,U_{17}` separately over
:math:`\mathbb Q(N)`. Their remainder columns have a one-dimensional
kernel. Its vector supplies :math:`\lambda_0,\ldots,\lambda_{17}`.
The verifier reconstructs the resulting eight :math:`Q_i`, checks all
polynomial remainders are zero, and separately substitutes the rational
row into :eq:`risk-five-vertex-certificate-row`. The certificate degrees
are :math:`(32,34,33,33,34,35,36,35)`.

Domain, boundaries and scalar coefficients
--------------------------------------------

All :math:`j`-poles of the row lie at :math:`-1,-2,-3`, outside the
integer summation range. The quotient degrees are at most :math:`2r+2`.
For :math:`N=D+r-1`, the largest possible leading coefficient of any
triangular reduction is at most :math:`-2D-1<0`; the other two types
are bounded still farther below zero. The verifier additionally checks
every one of the 280 parameter denominators directly: after the shift
:math:`N=D+16`, each has nonnegative coefficients and is nonzero.
Thus none vanishes at any :math:`D\geq1`.

The lower boundary vanishes through the common factor :math:`j^2` at
zero. The upper boundary is :math:`w_{N,N+1}c(N+1)T(N+1)=0`:
the weight vanishes and the remaining factors are finite.
There is no missing endpoint term.

Finally put :math:`d_D=(2D+4)!/((D-1)!(D+1)!)`. The proved single sum is
:math:`a_D=600\sum_jw_{D-1,j}\Psi_j/d_D`. Therefore the operator entries
before common-denominator clearing are

.. math::

   p_i(D)=\lambda_{r-i}(D+r-1)B_{r-i}(D+r-1)
                      \frac{d_{D+i}}{d_D},

up to a shared nonzero scalar. Primitive normalization produces the
recorded integer arrays. The seventeen initial fractions are checked
against the existing harmonic formula. The strict positivity of all
189 coefficients of :math:`p_{17}` proves the absence of exceptional
positive integers independently of finite root searches.

Reproduction and independent checks
------------------------------------

Use the prepared Sage environment to derive or verify the full operator:

.. code-block:: console

   $ python -m tools.risk_research.vertex_five_recurrence --verify tools/risk_research/examples/five_vertex_vertex_recurrence_order17.json --report .research/skeleton-risk/five-vertex-proof.json
   $ python -m tools.risk_research.vertex_five_recurrence --output .research/skeleton-risk/five-vertex-rederived.json

``recurrence_coefficients(maximum_degree)`` evaluates the certified
artifact with ordinary integer and fraction arithmetic and does not
import Sage. It is an evaluator, not a replacement for the symbolic
verification of an untrusted or edited operator.

Tests compare the recurrence with the independently proved harmonic
sum through degree 48 and at new degrees 263, 277, 293 and 307, as well
as the original scalar and rational Gram references at small degrees.
Optional native Sage tests rederive every coefficient, reject altered
operator/initial/domain data, and round-trip both CLI actions and their
source-bound proof reports. Finite comparisons check the implementation;
the eight exact identities and domain argument prove all degrees.
