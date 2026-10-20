The Category of Simploidal Sets
===============================

Simplicial sets are presheaves on the simplex category :math:`\Delta`, and a
morphism of simplicial sets is a family of maps that commutes with all face
and degeneracy operators.  This page sets up the same framework for the
simploidal sets of :doc:`simploidal` :cite:p:`peltier2009simploidal,peltier2018simploidal`
and proves that **the category of simploidal sets is a topos**.  The argument
is standard topos theory :cite:p:`maclane1992sheaves`; the only part specific
to simploids is the description of the indexing category.

Throughout, :math:`[k] = \{0, 1, \ldots, k\}`, and :math:`\Delta` is the
category of the ordered sets :math:`[k]` (:math:`k \ge 0`) and monotone maps.

The simploid category
---------------------

A type :math:`\mathbf a = (a_1, \ldots, a_n)` with every :math:`a_i \ge 1`
(and :math:`n \ge 0`; the empty type :math:`()` is a vertex) stands for the
simploid :math:`\Delta^{\mathbf a} = \Delta^{a_1} \times \cdots \times \Delta^{a_n}`,
whose vertex set is

.. math::

   V(\mathbf a) = [a_1] \times \cdots \times [a_n].

.. prf:definition:: Simploid category
   :label: def-simploid-category

   Let :math:`\mathbf a = (a_1, \ldots, a_n)` and
   :math:`\mathbf b = (b_1, \ldots, b_m)` be types.  A *simploidal map*
   :math:`f: \mathbf a \to \mathbf b` is a map :math:`f: V(\mathbf a) \to V(\mathbf b)`
   of the form

   .. math::

      f(x)_l =
      \begin{cases}
         g_l(x_{k_l}) & (l \in L), \\
         v_l & (l \notin L),
      \end{cases}

   where :math:`L \subseteq \{1, \ldots, m\}`, the map
   :math:`L \to \{1, \ldots, n\}`, :math:`l \mapsto k_l` is strictly
   increasing, every :math:`g_l: [a_{k_l}] \to [b_l]` is a **non-constant**
   monotone map, and every :math:`v_l \in [b_l]` is a vertex.

   The *simploid category* :math:`\mathbb S` has the types as objects and the
   simploidal maps as morphisms, composed as maps of sets.

In words: every factor of the target either follows exactly one factor of the
source through a monotone map, or is constant; different target factors
follow different source factors, and the order of the factors is kept.  A
source factor that no target factor follows is projected away.

Geometrically, :math:`f` extends affinely to the map
:math:`\Delta^{\mathbf a} \to \Delta^{\mathbf b}` that sends
:math:`(\mathbf u^1, \ldots, \mathbf u^n)` to the point whose :math:`l`-th
factor is :math:`\sum_x u^{k_l}_x \mathbf e_{g_l(x)}` for :math:`l \in L` and the
vertex :math:`\mathbf e_{v_l}` otherwise.  These are exactly the maps between
simploids that faces and degeneracies generate (:prf:ref:`lem-generation`).

.. prf:lemma::
   :label: lem-small-category

   :math:`\mathbb S` is a small category.

.. prf:proof::

   The identity of :math:`V(\mathbf a)` is simploidal (:math:`L` is everything,
   :math:`k_l = l`, :math:`g_l = \mathrm{id}`; :math:`\mathrm{id}_{[a_l]}` is
   non-constant because :math:`a_l \ge 1`).  Let
   :math:`f: \mathbf a \to \mathbf b` and :math:`h: \mathbf b \to \mathbf c` be
   simploidal, with data :math:`(L, k, g, v)` and :math:`(L', k', g', v')`.
   For a factor :math:`p` of :math:`\mathbf c`,

   * if :math:`p \notin L'`, then :math:`(h \circ f)(x)_p = v'_p` is constant;
   * if :math:`p \in L'` and :math:`l = k'_p \notin L`, then
     :math:`(h \circ f)(x)_p = g'_p(v_l)` is constant;
   * if :math:`p \in L'` and :math:`l = k'_p \in L`, then
     :math:`(h \circ f)(x)_p = (g'_p \circ g_l)(x_{k_l})`, where
     :math:`g'_p \circ g_l` is monotone; it is either constant or
     non-constant.

   So :math:`h \circ f` is simploidal, with :math:`L''` the set of :math:`p` of
   the last case with :math:`g'_p \circ g_l` non-constant and
   :math:`p \mapsto k_{k'_p}`, a composite of strictly increasing maps, as its
   index map.  Composition of maps of sets is associative and unital, and
   there are countably many types, each :math:`V(\mathbf a)` being finite.

The faces and degeneracies of simploids are morphisms of :math:`\mathbb S`.
Write :math:`\delta_j: [k - 1] \to [k]` for the monotone injection that skips
:math:`j`, and :math:`\sigma_j: [k + 1] \to [k]` for the monotone surjection
that hits :math:`j` twice.

.. prf:definition:: Coface and codegeneracy maps
   :label: def-generators

   Let :math:`\mathbf a = (a_1, \ldots, a_n)`.

   * For :math:`1 \le i \le n` and :math:`j \in [a_i]`, the *coface*
     :math:`\delta^i_j` is the simploidal map into :math:`\mathbf a` from the
     type :math:`\partial^i \mathbf a` obtained by decreasing :math:`a_i` by one
     and deleting it if it becomes zero.  It applies :math:`\delta_j` to the
     :math:`i`-th coordinate when :math:`a_i \ge 2`, and inserts the constant
     coordinate :math:`1 - j` as the :math:`i`-th one when :math:`a_i = 1`.
   * For :math:`1 \le i \le n` and :math:`j \in [a_i]`, the *codegeneracy*
     :math:`\sigma^i_j: (a_1, \ldots, a_i + 1, \ldots, a_n) \to \mathbf a`
     applies :math:`\sigma_j` to the :math:`i`-th coordinate.
   * For :math:`0 \le i \le n`, the *codegeneracy*
     :math:`\sigma^i_{-1}: (a_1, \ldots, a_i, 1, a_{i+1}, \ldots, a_n) \to \mathbf a`
     forgets the inserted coordinate.

   All other coordinates are kept.  For a presheaf :math:`X` on
   :math:`\mathbb S` (see below), :math:`d^i_j = X(\delta^i_j)` and
   :math:`s^i_j = X(\sigma^i_j)` are the face and degeneracy operators of
   :doc:`simploidal`.

Indeed, :math:`d^i_j` removes vertex :math:`j` of the :math:`i`-th factor (an
edge factor disappears, leaving its other vertex :math:`1 - j`), :math:`s^i_j`
repeats vertex :math:`j` of the :math:`i`-th factor, and :math:`s^i_{-1}`
inserts a degenerate edge after the :math:`i`-th factor.  In ``torch_bsf``,
``face(i, j)`` is :math:`d^{i+1}_j`, ``degeneracy(i, j)`` is
:math:`s^{i+1}_j` for :math:`j \ge 0`, and ``degeneracy(i, -1)`` is
:math:`s^i_{-1}`.  The ``degeneracy_map`` of a degenerate simploid records
the composite codegeneracy from its type to the type of its base: for each
factor, the factor of the base it follows and the monotone surjection, or
``-1`` when the factor is forgotten.

.. prf:lemma:: Generation
   :label: lem-generation

   Every morphism of :math:`\mathbb S` is a composite of cofaces and
   codegeneracies.  More precisely, every :math:`f: \mathbf a \to \mathbf b`
   factors as :math:`f = \mu \circ \varepsilon`, where :math:`\varepsilon` is a
   composite of codegeneracies and :math:`\mu` a composite of cofaces.

.. prf:proof::

   Let :math:`(L, k, g, v)` be the data of :math:`f`, and
   :math:`L = \{l_1 < \cdots < l_r\}`.  In :math:`\Delta`, factor each
   :math:`g_{l_t}` as a surjection :math:`\pi_t: [a_{k_{l_t}}] \to [c_t]`
   followed by an injection :math:`\iota_t: [c_t] \to [b_{l_t}]`; then
   :math:`c_t \ge 1` because :math:`g_{l_t}` is not constant.  With
   :math:`\mathbf c = (c_1, \ldots, c_r)`, let

   * :math:`\varepsilon: \mathbf a \to \mathbf c` be the simploidal map with
     :math:`\varepsilon(x)_t = \pi_t(x_{k_{l_t}})`, and
   * :math:`\mu: \mathbf c \to \mathbf b` the simploidal map with
     :math:`\mu(y)_{l_t} = \iota_t(y_t)` and :math:`\mu(y)_l = v_l` for
     :math:`l \notin L`.

   Then :math:`f = \mu \circ \varepsilon`.  Both maps act factor by factor, so
   it suffices to realize each factor with generators that only touch that
   factor.  Recall that in :math:`\Delta` every surjection is a composite of
   maps :math:`\sigma_j` and every injection one of maps :math:`\delta_j`
   :cite:p:`maclane1998categories`.

   * :math:`\varepsilon`: on a kept factor :math:`k_{l_t}`, write
     :math:`\pi_t` as a composite of maps :math:`\sigma_j`, i.e. of
     codegeneracies :math:`\sigma^{k_{l_t}}_j`.  A factor :math:`k` that is
     not kept is first collapsed to :math:`[1]` by a surjection
     :math:`[a_k] \to [1]` (codegeneracies :math:`\sigma^k_j`) and then
     forgotten by a codegeneracy :math:`\sigma^{k-1}_{-1}`.
   * :math:`\mu`: on a factor :math:`l_t`, write :math:`\iota_t` as a
     composite of maps :math:`\delta_j`, i.e. of cofaces :math:`\delta^l_j`
     with :math:`a_l \ge 2`.  A constant factor :math:`l \notin L` is inserted
     by a coface :math:`\delta^l_{1 - w}` (into an edge factor, at vertex
     :math:`w \in [1]`), followed by cofaces along an injection
     :math:`[1] \to [b_l]` sending :math:`w` to :math:`v_l`, which exists
     because :math:`b_l \ge 1`.

Simploidal sets as presheaves
-----------------------------

.. prf:definition:: Simploidal sets
   :label: def-simploidal-set

   A *simploidal set* is a presheaf on :math:`\mathbb S`, i.e. a functor
   :math:`X: \mathbb S^{\mathrm{op}} \to \mathbf{Set}`; :math:`X_{\mathbf a}`
   is its set of simploids of type :math:`\mathbf a`.  A *morphism of
   simploidal sets* is a natural transformation.  They form the category
   :math:`\mathbf{sSpd} = \mathbf{Set}^{\mathbb S^{\mathrm{op}}}`.

The *standard simploid* of type :math:`\mathbf a` is the representable
presheaf :math:`\Delta^{\mathbf a} = \mathbb S(-, \mathbf a)`, the
analogue of the standard simplex :math:`\Delta^n = \Delta(-, [n])`.  By the
Yoneda lemma, the simploids of type :math:`\mathbf a` of :math:`X` are the
morphisms :math:`\Delta^{\mathbf a} \to X`.

.. prf:corollary::
   :label: cor-morphisms

   A family of maps :math:`\phi_{\mathbf a}: X_{\mathbf a} \to Y_{\mathbf a}`
   between simploidal sets is a morphism if and only if it commutes with all
   face operators :math:`d^i_j` and all degeneracy operators :math:`s^i_j`.

.. prf:proof::

   Naturality with respect to every generator implies naturality with respect
   to their composites, which are all morphisms by :prf:ref:`lem-generation`.

So :math:`\mathbf{sSpd}` is the analogue of the category of simplicial sets
that the corollary describes: graded sets with face and degeneracy operators,
and the maps that commute with them.

.. prf:remark:: The axiomatic definition

   :cite:t:`peltier2018simploidal` (Definition 1) define a simploidal set
   axiomatically, as graded sets with operators :math:`d^i_j` and
   :math:`s^i_j` that satisfy a list of equations.  Such data are exactly
   the presheaves on the category :math:`\mathbb P` *presented* by the cofaces
   and codegeneracies as generators and those equations as relations: a
   functor out of a presented category is the same as an assignment of the
   generators that satisfies the relations.  :math:`\mathbb P` is small, so
   the theorem below applies to it as well; therefore the category of
   simploidal sets is a topos with either definition.

   The equations hold in :math:`\mathbb S`, since they are relations among
   the faces and degeneracies of standard simploids, so there is a functor
   :math:`\mathbb P \to \mathbb S` that is the identity on objects.  It is
   full by :prf:ref:`lem-generation`.  It is faithful precisely when the
   equations are a complete presentation of :math:`\mathbb S`, the analogue
   of the simplicial identities presenting :math:`\Delta`; the topos
   property does not depend on this.

The topos theorem
-----------------

.. prf:theorem::
   :label: thm-topos

   For every small category :math:`\mathcal C`, the presheaf category
   :math:`\widehat{\mathcal C} = \mathbf{Set}^{\mathcal C^{\mathrm{op}}}`
   is an elementary topos.  In particular, the category :math:`\mathbf{sSpd}`
   of simploidal sets is a topos (indeed a Grothendieck topos).

.. prf:proof::

   We write :math:`y c = \mathcal C(-, c)` for the Yoneda embedding and use the
   Yoneda lemma: elements :math:`x \in X(c)` correspond to morphisms
   :math:`\hat x: y c \to X`, with :math:`\hat x_{c'}(f) = X(f)(x)`.

   *Finite limits.*  Limits of presheaves exist and are computed objectwise,
   :math:`(\lim X_i)(c) = \lim X_i(c)`, because limits in :math:`\mathbf{Set}`
   commute with the restriction maps.

   *Exponentials.*  For presheaves :math:`X, Y`, put
   :math:`Y^X(c) = \widehat{\mathcal C}(y c \times X, Y)`, with
   :math:`Y^X(f)(\theta) = \theta \circ (y f \times X)` for
   :math:`f: c' \to c`.  For every presheaf :math:`Z` there is a bijection,
   natural in :math:`Z`,

   .. math::

      \widehat{\mathcal C}(Z \times X, Y) \cong \widehat{\mathcal C}(Z, Y^X).

   It sends :math:`\phi` to :math:`\psi` with
   :math:`\psi_c(z) = \phi \circ (\hat z \times X)`; conversely :math:`\psi`
   goes to :math:`\phi_c(z, x) = \psi_c(z)_c(\mathrm{id}_c, x)`.  Starting from
   :math:`\phi`, we get back
   :math:`\phi \circ (\hat z \times X)` evaluated at
   :math:`(\mathrm{id}_c, x)`, which is :math:`\phi_c(z, x)` since
   :math:`\hat z_c(\mathrm{id}_c) = z`.  Starting from :math:`\psi`, the
   naturality of :math:`\psi` gives, for :math:`f: c' \to c` and
   :math:`x \in X(c')`,

   .. math::

      \psi_c(z)_{c'}(f, x)
      = \bigl(Y^X(f)\,\psi_c(z)\bigr)_{c'}(\mathrm{id}_{c'}, x)
      = \psi_{c'}(Z(f) z)_{c'}(\mathrm{id}_{c'}, x)
      = \phi_{c'}(Z(f) z, x)
      = \bigl(\phi \circ (\hat z \times X)\bigr)_{c'}(f, x),

   so the two assignments are mutually inverse.  Hence
   :math:`\widehat{\mathcal C}` is cartesian closed.

   *Subobject classifier.*  A *sieve* on :math:`c` is a set :math:`S` of
   morphisms with codomain :math:`c` such that :math:`g \in S` implies
   :math:`g \circ h \in S` for every composable :math:`h`.  Let
   :math:`\Omega(c)` be the set of sieves on :math:`c`, with
   :math:`\Omega(f)(S) = f^* S = \{h \mid f \circ h \in S\}`, and let
   :math:`\top: 1 \to \Omega` pick the maximal sieve :math:`t_c` of all
   morphisms into :math:`c`.  Note that :math:`f \in S` if and only if
   :math:`f^* S = t_{c'}`, because a sieve containing an identity is maximal.

   Let :math:`A \subseteq X` be a subpresheaf (every monomorphism of presheaves
   is objectwise injective, hence isomorphic to one).  Define

   .. math::

      \chi_c(x) = \{f: c' \to c \mid X(f)(x) \in A(c')\}.

   This is a sieve because :math:`A` is closed under restriction, and
   :math:`\chi` is natural:
   :math:`f^* \chi_c(x) = \{h \mid X(h) X(f) x \in A\} = \chi_{c'}(X(f) x)`.
   Moreover :math:`\chi_c(x) = t_c` if and only if
   :math:`\mathrm{id}_c \in \chi_c(x)`, i.e. :math:`x \in A(c)`, so
   :math:`A` is the pullback of :math:`\top` along :math:`\chi`.
   Conversely, if :math:`\chi'` is any morphism along which :math:`A` is the
   pullback of :math:`\top`, then for :math:`f: c' \to c`,

   .. math::

      f \in \chi'_c(x)
      \iff f^* \chi'_c(x) = t_{c'}
      \iff \chi'_{c'}(X(f) x) = t_{c'}
      \iff X(f) x \in A(c')
      \iff f \in \chi_c(x),

   so :math:`\chi' = \chi`.  Thus :math:`\Omega` is a subobject classifier.

   A category with finite limits, exponentials and a subobject classifier is
   an elementary topos.  :math:`\mathbb S` is small by
   :prf:ref:`lem-small-category`, so :math:`\mathbf{sSpd}` is a topos.
   Presheaf categories are also the Grothendieck toposes of sheaves for the
   trivial topology :cite:p:`maclane1992sheaves`.

For simploidal sets, :math:`\Omega_{\mathbf a}` is the set of sieves on the
type :math:`\mathbf a`.  For example, every type :math:`\mathbf b` has a
morphism to every other type (a constant one), so a sieve on the vertex type
:math:`()` is empty or maximal: a vertex has just the two truth values
"false" and "true".  Higher simploids have many more truth values, such as
"true on this face only".

Remarks
-------

**Simplicial and cubical sets.**  The full subcategory of :math:`\mathbb S`
on the types of length at most one is isomorphic to :math:`\Delta`
(:math:`()` is :math:`[0]` and :math:`(a)` is :math:`[a]`): a simploidal map
:math:`(a) \to (b)` is either constant or a non-constant monotone map, i.e.
any monotone map.  Likewise, the full subcategory on the types
:math:`(1, \ldots, 1)` is the cube category with faces and degeneracies only
(no connections, symmetries or reversals) :cite:p:`grandis2003cubical`,
because the only non-constant monotone map :math:`[1] \to [1]` is the
identity.  Restricting a simploidal set to these subcategories gives its
underlying simplicial and cubical sets; as restrictions along functors
between small categories, these functors have both a left and a right
adjoint (left and right Kan extension), so they are the inverse image parts
of essential geometric morphisms.

**Two different products.**  The product of the topos is computed type by
type, :math:`(X \times Y)_{\mathbf a} = X_{\mathbf a} \times Y_{\mathbf a}`,
and it is **not** the cartesian product of simploids.  For instance,
:math:`\Delta^{(1)} \times \Delta^{(1)} \ne \Delta^{(1, 1)}`: the former has
the "diagonal" edge :math:`(\mathrm{id}, \mathrm{id})` of type :math:`(1)`,
nine edges in total including degenerate ones, whereas a simploidal map
:math:`(1) \to (1, 1)` follows the single source factor with at most one
target factor, so the square :math:`\Delta^{(1, 1)}` has only eight.  The
product of simploids is instead the concatenation of types,
:math:`\mathbf a \cdot \mathbf b`, which is a functor
:math:`\mathbb S \times \mathbb S \to \mathbb S` (apply two simploidal maps
side by side) and makes :math:`\mathbb S` strict monoidal with unit
:math:`()`.  By Day convolution :cite:p:`day1970closed` it extends to a
monoidal structure :math:`\otimes` on :math:`\mathbf{sSpd}` with
:math:`\Delta^{\mathbf a} \otimes \Delta^{\mathbf b} = \Delta^{\mathbf a \cdot \mathbf b}`;
this is the product that Bézier simploids model.

**Conditions that break the topos structure.**  Data structures often
restrict to simploidal sets with extra properties, e.g. that the faces of a
simploid are pairwise distinct.  Such full subcategories are typically not
closed under the colimits of :math:`\mathbf{sSpd}` (gluing two faces of one
simploid is a colimit that violates the condition), so the theorem does not
apply to them; it concerns the category of all simploidal sets.
