References
==========

The bibliography supports the mathematical background and the application
formulations discussed in this documentation. Use the links from each
chapter to identify the result relevant to its assumptions. A theorem
about the reference optimization problem does not by itself give an error
bound for a finite-data Bézier fit.

Fitting, approximation and statistical behavior
-----------------------------------------------

:cite:t:`kobayashi2019bezier` introduces Bézier simplex fitting for Pareto
fronts and its approximation perspective. :cite:t:`tanaka2020asymptotic`
studies statistical risk. Read these alongside :doc:`whatis` and
:doc:`advanced/auto_degree`: expressiveness at a sufficiently large degree
and accuracy from a particular finite sample are distinct questions.

:cite:t:`hikima2025stochastic` concerns objective-based stochastic
optimization of a Bézier representation. Its setting is discussed in
:doc:`advanced/sgd`; it should be distinguished from linear least-squares
regression on known parameter-response pairs.

Topology and domain construction
--------------------------------

:cite:t:`mizota2021unconstrained` provides the unconstrained strongly convex
results stated in :doc:`whatis`. Check smoothness, feasibility and objective
assumptions before transferring a statement to a constrained application.
Weak simpliciality permits a many-to-one parameterization; it is stronger
to claim that the image is homeomorphic to a simplex.

:cite:t:`hamada2018data,hamada2020test` discuss topology assessment from
data under their stated assumptions. A sampled picture or failure to
detect a hole is not independently a proof of simplex topology.

For product cells and their faces, see
:cite:t:`peltier2009simploidal,peltier2018simploidal` and the geometric
introduction in :doc:`advanced/simploidal`. The mathematical continuation
in :doc:`advanced/simploidal_category` uses the category-theory background
of :cite:t:`maclane1998categories,maclane1992sheaves`.

Shape-preserving interpolation on simplicial meshes
(:doc:`advanced/interpolation`) generalizes the monotone piecewise cubic
interpolation of :cite:t:`fritsch1980monotone`. Its face controls use the
degree elevation of Bézier triangles described by :cite:t:`farin2002curves`.

Application sources and reproducible examples
---------------------------------------------

Each :doc:`application <applications>` links its formulation to relevant
literature and names the script used for its figures. These scripts are
small numerical illustrations of specified targets. Consult the page's
parameter range, validation set and feasibility checks when interpreting
its reported error; a reference to a broader application does not make
the toy experiment an implementation of that complete system.

Bibliography
------------

.. bibliography::
