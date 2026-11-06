Tetrahedron vertex degree recurrence
====================================

The positive harmonic sum in :doc:`skeleton_risk_vertex` admits an explicit
order-ten recurrence in the degree. Its polynomial coefficients have degree
at most 69. The recurrence holds at every integer :math:`D\geq1`, has no
exceptional integer indices, and determines the coefficient from its first
ten values. This supplies a fixed-order recurrence; it does not evaluate
the coefficient by a bounded number of classical special functions.

.. prf:theorem:: A certified degree recurrence
   :label: thm-risk-tetrahedron-vertex-recurrence

   Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`,
   write :math:`a_D=a_0(4,D)`. The eleven integer polynomials recorded in
   :file:`tools/risk_research/examples/tetrahedron_vertex_recurrence_order10.json`
   satisfy

   .. math::

      \sum_{i=0}^{10}p_i(D)a_{D+i}=0\qquad(D\geq1).

   The JSON rows give coefficients in ascending powers of :math:`D`.
   Their maximum degree is 69, and :math:`p_{10}(D)\ne0` at every
   positive integer. The same artifact records the ten exact initial
   values, starting with :math:`a_1=8/5` and :math:`a_2=16/35`.

The proof below constructs a rational telescoping certificate from these
polynomials. The executable verifier rebuilds this certificate and checks
four rational identities exactly. Finite sequence agreement is an
additional regression check, rather than the proof of the recurrence.

Four polynomial difference equations
-------------------------------------

Recall the proved single sum

.. math::

   a_D=\frac{576}{d_D}\sum_{j=0}^{D-1}
      \binom{D-1}j\binom{D+1}j G_j,
   \qquad d_D=\frac{(2D+3)!}{(D-1)!(D+1)!},

where

.. math::

   G_j=\frac{H_j^2}{(j+1)(j+2)(j+3)}
      +\frac{H_j^{(2)}}{(j+1)(j+3)}
      -\frac{jH_j}{(j+1)(j+2)^2(j+3)}
      +\frac{2}{(j+1)^2(j+2)^2(j+3)}.

For a candidate order :math:`r`, put :math:`N=D+r-1` and
:math:`w_{N,j}=\binom Nj\binom{N+2}j`. With
:math:`x^{\underline m}=x(x-1)\cdots(x-m+1)`, define

.. math::

   U_m(j)=(N-j)^{\underline m}(N+2-j)^{\underline m},\qquad
   B_m=N^{\underline m}(N+2)^{\underline m}.

The elementary shift identity is

.. math::

   w_{N,j}U_m(j)=B_m
        \binom{N-m}j\binom{N+2-m}j.

It holds on :math:`0\leq j\leq N`, including the zero terms outside
the shorter support. For :math:`P(j)=\sum_{m=0}^r\lambda_m U_m(j)`,
we seek a telescope for :math:`w_{N,j}P(j)G_j`.

Set :math:`A(j)=(N-j)(N+2-j)` and introduce two polynomial operators
over :math:`\mathbb Q(N)`:

.. math::

   L_2 q&=A(j)(j+1)q(j+1)-j^2(j+3)q(j),\\
   L_3 q&=A(j)(j+1)^2q(j+1)-j^2(j+2)(j+3)q(j).

Four certificate polynomials :math:`q_1,q_2,q_3,q_4` must satisfy

.. math::
   :label: risk-vertex-polynomial-certificate

   L_2q_1&=P,\\
   L_2q_2&=(j+2)P,\\
   L_3q_3&=-j(j+1)P
      -2A(j)(j+1)(j+2)q_1(j+1),\\
   L_3q_4&=2P-A(j)(j+2)\bigl(q_1(j+1)+q_2(j+1)\bigr)
      -A(j)(j+1)q_3(j+1).

These equations can be solved by polynomial reduction. The leading terms
of :math:`L_2(j^m)` and :math:`L_3(j^m)` are respectively

.. math::

   (m-2N-4)j^{m+2},\qquad (m-2N-5)j^{m+3}.

Successively eliminating the highest powers therefore leaves remainders
of degrees below :math:`2,2,3,3`. Their ten coefficients depend linearly
on the :math:`r+1` unknowns :math:`\lambda_m`. At :math:`r=10` the
resulting :math:`10\times11` matrix has an exact nonzero kernel. The
implementation computes that kernel over :math:`\mathbb Q(N)` without
using sequence values. This establishes an order bound for this
certificate construction; it does not claim that order ten is minimal
among all recurrences for :math:`a_D`.

The rational telescope and its boundaries
-----------------------------------------

Let :math:`h_j=(H_j^2,H_j^{(2)},H_j,1)^T` and let :math:`b(j)` be the
four rational coefficients of :math:`G_j=b(j)h_j` above. Define the row

.. math::

   c(j)=j^2\left(
      \frac{q_1(j)}{(j+1)(j+2)},
      \frac{q_2(j)}{(j+1)(j+2)},
      \frac{q_3(j)}{(j+1)^2(j+2)},
      \frac{q_4(j)}{(j+1)^2(j+2)}\right).

The harmonic transition :math:`h_{j+1}=T(j)h_j` is

.. math::

   T(j)=\begin{pmatrix}
      1&0&2/(j+1)&1/(j+1)^2\\
      0&1&0&1/(j+1)^2\\
      0&0&1&1/(j+1)\\
      0&0&0&1
   \end{pmatrix}.

Substitution of :eq:`risk-vertex-polynomial-certificate` gives the four
rational identities

.. math::

   \frac{A(j)}{(j+1)^2}c(j+1)T(j)-c(j)=P(j)b(j).

Multiplication by :math:`w_{N,j}h_j` telescopes the desired summand.
The lower boundary vanishes because :math:`c(0)=0`. At the upper
boundary :math:`w_{N,N+1}=0`, and :math:`c(N+1)` is finite, so the
sum is zero.

There are no hidden poles on the summation domain. Polynomial reduction
uses indices :math:`m\leq2r-1` and :math:`N=D+r-1\geq r`, hence
:math:`m-2N-4<0` and :math:`m-2N-5<0`. The factors :math:`B_m`
are positive. The normalizing ratio

.. math::

   \frac{d_{D+i}}{d_D}
     =\frac{(2D+4)_{2i}}{(D)_i(D+2)_i}

has positive factors as well. The certificate denominators in
:math:`j` involve only :math:`j+1,j+2`.
The verifier additionally expands every certificate parameter denominator
after :math:`N=D+r-1` and checks that its coefficients are nonnegative
and not all zero. This independently certifies positivity for
:math:`D\geq1`.

Finally set

.. math::

   p_i(D)=\lambda_{r-i}(D+r-1)B_{r-i}(D+r-1)
         \frac{d_{D+i}}{d_D}.

Clearing the common denominator and integer content gives the recorded
polynomials and proves the theorem. Conversely, the verifier starts
with supplied :math:`p_i`, recovers :math:`\lambda_m` through this
formula, and checks the telescope directly. It therefore detects an
incorrect coefficient array independently of its provenance metadata.

Integer domain and reproduction
---------------------------------

The leading polynomial has one sign variation in its nonzero
coefficients. Its exact evaluations satisfy
:math:`p_{10}(4)<0<p_{10}(5)`. Descartes' rule and the intermediate
value theorem place its only positive real root strictly inside
:math:`(4,5)`. Thus it has no positive integer zero; recurrence
evaluation needs no additional initial values beyond degrees one
through ten.

With the optional Sage environment, reproduce both the derivation and
the symbolic certificate using

.. code-block:: console

   python -m tools.risk_research.vertex_degree_recurrence \
     --output /tmp/tetrahedron-vertex.json --report /tmp/vertex-proof.json
   python -m tools.risk_research.vertex_degree_recurrence \
     --verify tools/risk_research/examples/tetrahedron_vertex_recurrence_order10.json

Ordinary Python can evaluate the certified recurrence without Sage:

.. code-block:: python

   from fractions import Fraction
   from tools.risk_research.vertex_degree_recurrence import recurrence_coefficients

   values = recurrence_coefficients(300)
   assert values[1] == Fraction(8, 5)

Regression checks compare recurrence values with the previously proved
positive sum at degrees one through 48 and at 121, 241, 260, 271, 277,
290 and 300. The degrees above 260 were checked after the operator was
frozen. These checks do not replace the symbolic certificate.
