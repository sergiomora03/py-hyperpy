Class
=======

.. currentmodule:: hyperpy.core

.. _class-models:

Class models
----------------

The class :class:`models` buils a model from a set of parameters.

.. autoclass:: hyperpy.core.models
   :members:
   :undoc-members:

The fact, all parameters for build model are (default):

   - initnorm=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
   - min_layers:int=1,
   - max_layers:int=13,
   - min_units:int=4,
   - max_units:int=128

and at the moment we can manipulate the model with the following methods:

.. autofunction:: hyperpy.core.models.BuildModelSimply

.. autofunction:: hyperpy.core.models.BuildModel

The difference between th two methods is the first use the same activation function for all layers, the second use different activations funcion for each layer.


.. _class-optimizers:

Class optimizers
-----------------

The class :class:`optimizers` build optimizers for the model.

.. autoclass:: hyperpy.core.optimizers
   :members:
   :undoc-members:

At the moment, we can select between:

.. autofunction:: hyperpy.core.optimizers.optimizerAdam
.. autofunction:: hyperpy.core.optimizers.optimizerRMSprop
.. autofunction:: hyperpy.core.optimizers.optimizerSGD

And if we want that the model is trained with several optimizers, we can use the method:

.. autofunction:: hyperpy.core.optimizers.buildOptimizer