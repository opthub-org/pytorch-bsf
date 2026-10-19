.. PyTorch-BSF documentation master file, created by
   sphinx-quickstart on Sun Jun 27 02:31:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch-BSF!
=======================

**Fit smooth, high-dimensional manifolds to your data — from a single GPU to a multi-node cluster.**

PyTorch-BSF brings `Bézier simplex fitting`_ to PyTorch.
A Bézier simplex is a high-dimensional generalization of the Bézier curve: it can model an arbitrarily complex point cloud as a smooth parametric hyper-surface in any number of dimensions.
This makes it a natural tool for representing **Pareto fronts** in multi-objective optimization, interpolating scattered observations, and fitting geometric structures in high-dimensional spaces.

The project is on `GitHub`_.

.. _Bézier simplex fitting: whatis.html

.. image:: _static/bezier-simplex.png
   :width: 49%
   :alt: A Bézier simplex and its control points

.. image:: _static/bezier-simplex-fitting.png
   :width: 49%
   :alt: A Bézier simplex that fits to a dataset

.. toctree::
   :maxdepth: 2
   :caption: User guide

   whatis
   quickstart
   advanced
   applications
   faq
   references

.. toctree::
   :maxdepth: 2
   :caption: API reference

   modules


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/opthub-org/pytorch-bsf
