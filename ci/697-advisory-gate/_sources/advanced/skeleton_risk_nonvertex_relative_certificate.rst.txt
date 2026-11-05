One certificate for a requested finite-degree relative accuracy
================================================================

Keep the uniform-simplex design, evaluation distribution and noise
assumptions of :doc:`skeleton_risk_theory`. Write :math:`s\geq2` for the
number of vertices of the source face, :math:`q\geq2` for its outside
codimension, and :math:`M=s+q`. The quantity being enclosed is the
coefficient :math:`a=a_{s-1}(s+q,D)` of sample-size-asymptotic risk.
It is not a finite-sample risk.

The general construction already gives the positive constant

.. math::

   A_{s,q}=\frac{\binom{s+q}s^2(s)_q}{2^q(s-2)!(q+1)}c_{q+1},
   \qquad c_{q+1}\geq q+1.

The constant index here is :math:`q+1`, while q itself denotes the
outside codimension. The lower bound follows from the positive diagonal
mass :math:`c_{q+1}/(q+1)=\sum_{j\geq0}d_j\geq d_0=1` in
:doc:`skeleton_risk_nonvertex_general_error`.

.. prf:theorem:: Relative-accuracy composition at a fixed degree
   :label: thm-risk-nonvertex-relative-certificate

   Let :math:`0<\varepsilon<1` be rational and :math:`D\geq s`.
   Suppose the proved general finite-degree error evaluator supplies
   :math:`0\leq r<\varepsilon` such that

   .. math::

      \left|\frac a{A_{s,q}D^{s-q-2}}-1\right|\leq r.

   Given a rational constant enclosure
   :math:`0<A_L\leq A_{s,q}\leq A_U`, define

   .. math::
      :label: risk-nonvertex-relative-composition

      L&=A_L(1-r)D^{s-q-2},\quad
      U=A_U(1+r)D^{s-q-2},\\
      \widehat a&=\frac{2LU}{L+U},\quad
      \varepsilon_{\rm cert}=\frac{U-L}{U+L},\quad
      \delta_A=\frac{A_U-A_L}{A_U+A_L}.

   Then :math:`0<L\leq a\leq U` and

   .. math::

      \left|\frac{\widehat a}a-1\right|
      \leq\varepsilon_{\rm cert}
      =\frac{\delta_A+r}{1+\delta_A r}.

   Put :math:`\tau=(\varepsilon-r)/(1-\varepsilon r)>0`.
   One dyadic enclosure of :math:`c_{q+1}` with width at most
   :math:`2(q+1)\tau` suffices for
   :math:`\varepsilon_{\rm cert}\leq\varepsilon`, after intersecting
   its lower endpoint with the known bound :math:`c_{q+1}\geq q+1`.

.. prf:proof::

   Positivity gives the interval for a. On :math:`[L,U]`,
   :math:`\widehat a/a-1` decreases with a. Its values at the two
   endpoints are respectively :math:`(U-L)/(U+L)` and its negative.
   This proves the true-value-denominator relative error; the arithmetic
   midpoint would not give this bound. Substitution of L and U gives
   the formula in :math:`\delta_A,r`. Since :math:`r<\varepsilon<1`,
   elementary rearrangement shows that this formula is at most
   :math:`\varepsilon` exactly when :math:`\delta_A\leq\tau`.

   Let :math:`[c_L,c_U]` be the dyadic interval after the lower clamp.
   The positive scale defining A cancels in :math:`\delta_A`, and
   :math:`c_L+c_U\geq2(q+1)`. Its width at most :math:`2(q+1)\tau`
   therefore implies :math:`\delta_A\leq\tau`.
   :doc:`skeleton_risk_mzv_dyadic` supplies width at most
   :math:`2^{-b}` for every requested positive integer b. Choosing b
   once so that :math:`2^{-b}\leq2(q+1)\tau` proves the assertion.

Research evaluator and refusal contract
-----------------------------------------

``tools.risk_research.nonvertex_relative_certificate`` computes the
analytic bound before constructing an MZV expression. It chooses b
using integer bit lengths and at most one rational comparison, then
charges symbolic construction, cutoff selection and numerical evaluation
to one shared additive work counter. The separate analytic selected-input
counter bounds accepted scalar inputs. Neither counter is a CPU-time,
RSS or complete integer-bit-complexity bound.

The CLI accepts ``--face-size``, ``--outside``, ``--degree``,
``--relative-target``, ``--analytic-work-limit`` and
``--numeric-work-limit``. The target must lie strictly between zero and
one. Both resource policies and input bit lengths are checked before
diagonal or MZV construction. For an inactive face :math:`D<s`, the
coefficient is exactly zero: its record has null relative-error fields,
and performs no analytic or constant evaluation.

If :math:`r\geq\varepsilon`, the tool conservatively refuses: this
bound cannot certify the requested target. It does not assert that the
actual coefficient error is too large. This also refuses the equality
case even if an exactly known constant could certify it. Work-limit
refusals likewise do not refute the mathematical stopping theorem.
No failure is serialized as a success certificate; the CLI exits
nonzero, reports its reason on stderr and preserves any existing output.
The first version handles one fixed degree and does not automatically
choose among sharper codimension-specific methods or search an all-tail
threshold.

Success records use the new schema
``skeleton-risk-nonvertex-relative-certificate-v1``. They include both
budgets, actual additive work, selected precision and cutoff, constant
and coefficient intervals, the analytic bound, harmonic-center
approximation and certified relative error. Replay restores every saved
option and compares the entire typed record. It uses the source union of
the wrapper, proof, both evaluators and their recorded dependencies.
Fingerprints normalize CRLF pairs to LF; raw-byte guards still detect
line-ending changes during calculation. Sources, Git HEAD, tree and
complete checkout status are checked across the composition and replay;
the CLI also guards the raw saved input across verification. These guards
detect changes but are not a sandbox or protection against transient
changes restored between snapshots.

Use ``checkout_harness.run_python`` for an initially clean committed
checkout and separately retained outputs. Its finally-block guards also
cover failed commands and timeouts. A certificate does not encode the
checkout commit or status, so replay from another unchanged clean commit
with the same bound sources remains possible. Historical schemas and
their evaluator sources are unchanged.

For example, from the repository root, with output outside the checkout::

   python -m tools.risk_research.nonvertex_relative_certificate \
     --face-size 3 --outside 4 --degree 1208925819614629174706193 \
     --relative-target 1/100 --analytic-work-limit 190000 \
     --numeric-work-limit 180000 --output /tmp/nonvertex-relative.json
   python -m tools.risk_research.nonvertex_relative_certificate \
     --verify /tmp/nonvertex-relative.json

The evaluator requires only the Python standard library. Finite checks
of its output validate implementation and bookkeeping; the composition
proof and the two component theorems establish interval inclusion.
