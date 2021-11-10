.. image:: ../../img/logo.png
   :alt: HyperPy Logo

Welcome to HyperPy's documentation!
===================================

**HyperPy** (py-hyperpy in PyPi) is a Python library for build an automatic hyperparameter optimization.

You can install `hyperpy` with pip:

.. code-block:: console

   # pip install py-hyperpy


Example
----------------

Read library:

.. code-block:: python

   import hyperpy as hy

Run the optimization:

.. code-block:: python

   running=hy.run(feat_X, Y)
   study = running.buildStudy()


See the results:

.. code-block:: python

   print("best params: ", study.best_params)
   print("best test accuracy: ", study.best_value)
   best_params, best_value = hy.results.results(study)


.. note::

   - The function `hy.run()` return a `Study` object. And only needs: Features, target. In the example: best test accuracy = 0.7407407164573669
   - *feat_X*: features in dataset
   - *Y*: target in dataset

.. warning::

   At moment only solves binary clasification problems.

.. note::

   This project is under active development.

**Citing PyCaret**\ :

If you’re citing PyCaret in research or scientific paper, please cite this page as the resource. PyCaret’s first stable release 1.0.0 was made publicly available in April 2020. 

pycaret.org. PyCaret, April 2020. URL https://pycaret.org/about. PyCaret version 1.0.0.

A formatted version of the citation would look like this::

    @Manual{PyCaret,
      author  = {Mora, Sergio},
      title   = {HyperPy: An automatic hyperparameter optimization framework in Python},
      year    = {2021},
      month   = {October},
      note    = {HyperPy version 0.0.5},
      url     = {https://py-hyperpy.readthedocs.io/en/latest/}
    }

Contents
--------

.. toctree::

   install
   usage
   core
   util