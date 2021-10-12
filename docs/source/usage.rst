Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

create study:
----------------

The library **hyperpy** function by study. This study represent several running
of an diferents neural networks, to find the best fit. To run a study, you could
call ``hy.run(feat_X, Y)`` function:

.. autofunction:: hy.run(feat_X, Y):

The ``Feat_X`` parameter should be the feature to train the model. And ``"Y"``
represents the target in dataset. However, :py:func:`hy.run(feat_X, Y)`
at the moment just run clasification problems and run study with doble cross validation.


For example:

>>> import hyperpy as hy
>>> running=hy.run(feat_X, Y)
>>> study = running.buildStudy()

Then the study return the structure of the neural netowork and the accuracy.