An all-degree sufficient region for large vertex counts
========================================================

Under the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`, put

.. math::

   R_{\mathrm{ind}}(M,D)=\left(\sum_{s=1}^{\min(D,M)}
          \sqrt{a_{s-1}(M,D)}\right)^2,
   \qquad R_{\mathrm{all}}(M,D)=\binom{M+D-1}{D}.

The common positive noise and output factors are omitted. The following
bound covers every positive integer degree without generating a
degree-specific dimension polynomial. It is a sufficient region, not a
claim about the exact crossover or the shape of the intervening loss set.

.. prf:theorem:: A linear sufficient vertex threshold at every degree
   :label: thm-risk-uniform-dimension-tail

   For all integers :math:`D\geq1` and :math:`M\geq9720D`,

   .. math::
      :label: risk-uniform-dimension-tail

      R_{\mathrm{ind}}(M,D)<\frac45R_{\mathrm{all}}(M,D).

   In particular, optimal inductive fitting has strictly smaller
   asymptotic risk throughout this region. The constant 9720 is
   conservative and is not asserted to be optimal.

.. prf:proof::

   **A positive kernel majorant.** The coefficients of the outside
   kernel :math:`F` in :prf:ref:`lem-skeleton-kernel` are nonnegative,
   and coefficientwise

   .. math::

      F(x,y)\preceq(1-x-y)^{-1}.

   Products preserve this ordering. With :math:`q=M-s` and
   :math:`t=h+l`, this gives

   .. math::

      K_q(h,l)\leq\frac{(q)_t}{h!l!}.

   Initially suppose only :math:`M\geq4D`. For every active size
   :math:`1\leq s\leq D`, the scalar formula has
   :math:`0\leq t\leq2(D-s)`. Every factor of :math:`(q)_t`
   is at most :math:`M+2D`, including the empty product at :math:`t=0`.
   Therefore

   .. math::
      :label: risk-uniform-kernel-majorant

      (q)_t\leq M^t\left(1+\frac{2D}{M}\right)^{2D}
          \leq M^t\left(\frac94\right)^D.

   **A bound for one skeleton.** Set :math:`n=D-s`. In
   :prf:ref:`thm-skeleton-scalar-sum`, the two binomial upper indices
   defining :math:`A_{k,h}` add to :math:`2D+s-1`. The binomial
   theorem bounds each binomial by two to its upper index, so

   .. math::

      |A_{k,h}A_{k,l}|\leq2^{4D+2s-2}=16^D4^{s-1}.

   Also :math:`\binom Ms\leq M^s/s!` and
   :math:`(M)_{2D}\geq M^{2D}`. The factorial from the scalar sum
   satisfies

   .. math::

      \frac{(2D+s-t-1)!}{(s-1)!}
          =\prod_{j=s}^{2D+s-t-1}j\leq(3D)^{2D-t}.

   Since :math:`2n-t\geq0` and :math:`M\geq4D\geq3D`,

   .. math::

      M^{t-2n}(3D)^{2D-t}
         =(3D)^{2s}\left(\frac{3D}{M}\right)^{2n-t}
         \leq(3D)^{2s}.

   For each angular index :math:`k`, there are at most :math:`D^2`
   pairs :math:`(h,l)`. The multiplicities obey

   .. math::

      \sum_{k=0}^{D-s}d_s(k)=\binom{D-1}{s-1}.

   This is the hockey-stick identity when :math:`s\geq2`; at
   :math:`s=1` only :math:`k=0` contributes and both sides are one.
   Apply the triangle inequality to the exact scalar formula; its
   original nonnegativity follows because it is a risk coefficient.
   The factors :math:`h!l!` cancel with the kernel majorant. Combining
   the displayed bounds gives, simultaneously for all :math:`M\geq4D`,

   .. math::
      :label: risk-uniform-skeleton-majorant

      0\leq a_{s-1}(M,D)\leq
      B_{D,s}:=D^2 36^D4^{s-1}\binom{D-1}{s-1}
              \frac{(3D)^{2s}}{(s!)^2}.

   **Sum over all skeletons.** The sequence
   :math:`(3D)^s/s!` increases for :math:`1\leq s\leq D`.
   Moreover,

   .. math::

      \sum_{s=1}^D4^{s-1}\binom{D-1}{s-1}=5^{D-1}.

   There are exactly :math:`D` active sizes because :math:`M\geq4D`.
   Cauchy--Schwarz and :eq:`risk-uniform-skeleton-majorant` imply

   .. math::
      :label: risk-uniform-risk-majorant

      R_{\mathrm{ind}}(M,D)\leq D\sum_{s=1}^DB_{D,s}
      \leq U_D:=D^3 36^D5^{D-1}
                    \left(\frac{(3D)^D}{D!}\right)^2.

   **Elementary degree bounds.** For every positive integer :math:`D`,

   .. math::

      D^D<3^D D!,\qquad D^3\leq4\,2^D.

   For the first, the ratio of successive terms of
   :math:`D^D/(3^DD!)` is :math:`(1+1/D)^D/3<1`.
   Indeed the binomial expansion bounds :math:`(1+1/D)^D` by
   :math:`\sum_{j=0}^D1/j!<3`; the tail from :math:`j=2` is bounded
   by the geometric series :math:`\sum_{j=2}^{\infty}2^{1-j}=1`.
   The initial term at :math:`D=1` is :math:`1/3<1`.
   For the second, check :math:`D=1,2,3,4` directly. Starting at
   :math:`D=4`, the ratio of successive cubes is at most
   :math:`(5/4)^3<2`, which proves the rest by induction.

   Consequently, with strictness retained from the factorial inequality,

   .. math::

      D!U_D
      &=\frac{D^3}{5}\,1620^D\frac{D^{2D}}{D!}\\
      &<\frac{D^3}{5}(4860D)^D
       \leq\frac45(9720D)^D.

   Finally :math:`R_{\mathrm{all}}(M,D)=(M)_D/D!\geq M^D/D!`.
   If :math:`M\geq9720D`, this bound and
   :eq:`risk-uniform-risk-majorant` prove
   :eq:`risk-uniform-dimension-tail`, including :math:`D=1`.

Exact specialization and checks
--------------------------------

The optional standard-library tool evaluates :math:`B_{D,s}`, the
closed rational bound :math:`U_D`, and the sufficient threshold.
It does not construct a dimension polynomial or calculate an actual
risk coefficient. For example:

.. code-block:: console

   $ python -m tools.risk_research.uniform_dimension_tail --degree 31 --output .research/skeleton-risk/uniform-tail31.json
   $ python -m tools.risk_research.uniform_dimension_tail --verify .research/skeleton-risk/uniform-tail31.json

The JSON record includes the exact strict margin below four fifths of
the all-at-once lower bound at :math:`M=9720D`, both applicable vertex
domains, and source hashes. The ``source_hash_policy`` field is
``sha256-crlf-to-lf-v1``: replace each CRLF pair by LF before hashing,
and retain every other byte. This makes fingerprints portable between
LF and CRLF checkouts. The verifier rejects changed bounds, domains,
hash policies, content fingerprints and unrecognized fields.
The before/after source guard compares the original raw bytes, so even
a line-ending-only edit during certification fails. Output replacement
is atomic and happens only after arithmetic and source-consistency checks.
Large exact numerators are serialized in local decimal chunks, without
changing Python's process-wide integer-string conversion limit.

Independent finite checks use the original scalar kernel and risk
formula, the previously counted dimension polynomials, and a degree-31
holdout after freezing the stated theorem. They check the implementation
and the intermediate bounds. The all-degree and infinite-vertex claims
follow from the proof above, not from these finite checks.
