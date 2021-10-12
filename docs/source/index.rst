Welcome to HyperPy's documentation!
===================================

.. image:: ../../img/logo.png
   :alt: HyperPy Logo

**HyperPy** (py-hyperpy in PyPi) is a Python library for build an automatic hyperparameter optimization.

You can install `hyperpy` with pip:

.. code-block:: console

   # pip install py-hyperpy


Example
----------------

Read library:

.. code-block:: console

   import hyperpy as hy

Run the optimization:

.. code-block:: console

   running=hy.run(feat_X, Y)
   study = running.buildStudy()


See the results:

.. code-block:: console

   print("best params: ", study.best_params)
   print("best test accuracy: ", study.best_value)
   best_params, best_value = hy.results.results(study)


.. note::

   - The function `hy.run()` return a `Study` object. And only needs: Features, target. In the example: best test accuracy = 0.7407407164573669
   - *feat_X*: features in dataset
   - *Y*: target in dataset

.. note::

   At moment only solves binary clasification problems.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api