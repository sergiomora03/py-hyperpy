Classes
=======

.. _class-models:

Class models
----------------

.. currentmodule:: hyperpy.core

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

.. currentmodule:: hyperpy.core

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


.. _class-trainers:

Class trainers
--------------

.. currentmodule:: hyperpy.core


The class :class:`trainers` build trainers for the model.

.. autoclass:: hyperpy.core.trainers
   :members:
   :undoc-members:

The final idea, is to select by several type of trainers. By the way, at moment have onle one trainer:

.. autofunction:: hyperpy.core.trainers.trainer

.. _class-run:

Class run
---------

.. currentmodule:: hyperpy.core

To run a study, you could call ``hy.run(feat_X, Y)`` function:

.. autoclass:: hyperpy.core.run
   :members:
   :undoc-members:

.. autofunction:: hyperpy.core.run.buildStudy

.. autofunction:: hyperpy.core.run.objective

.. _class-results:

Class results
---------

.. currentmodule:: hyperpy.core

To read results from a study, you could call ``hy.results(study)`` function:

.. autoclass:: hyperpy.core.results
   :members:
   :undoc-members:

.. autofunction:: hyperpy.core.results.results