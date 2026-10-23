Visualization
=============

Install ``pytorch-bsf[plot]`` and use one plotting function::

    from torch_bsf.plotting import plot_model
    ax = plot_model(result.model, resolution=20, coordinates=(0, 1, 2))

Choose two or three output coordinates. Maximal cells share the same axes;
domains above dimension two are displayed through their 2D faces. Optional
control nets show shared coefficient geometry.
