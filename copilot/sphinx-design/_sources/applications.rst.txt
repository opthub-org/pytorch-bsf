Applications
============

This section explores the practical applications of Bézier simplex fitting. By exploiting the continuous relationships between optimal solutions and competing objectives, Bézier simplices provide a powerful framework for multi-objective optimization and parametric modeling.

The most prominent and elegant application of this technique is the continuous approximation of the multi-parameter regularization path in Elastic Net and other sparse modeling methods. We detail how a Bézier simplex can efficiently capture these paths, enabling comprehensive model selection without repeatedly re-training models over a discrete grid.

Furthermore, we present several other potential applications across diverse domains. While these examples arise from different fields, they share a crucial mathematical property: their underlying formulations are strongly convex optimization problems. As established in the theoretical foundations of this method, strong convexity guarantees that their Pareto sets admit a simplex-like topology. This inherently makes them perfectly suited for Bézier simplex fitting.

.. toctree::
   :maxdepth: 1

   applications/elastic_net
   applications/portfolio
   applications/smart_grids
   applications/federated_learning
   applications/model_predictive_control
   applications/communications
   applications/supply_chain
   applications/medical_imaging
   applications/facility_location
