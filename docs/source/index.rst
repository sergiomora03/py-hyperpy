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

import library:

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

   This project is active development.

**Citing HyperPy**\ :

If you’re citing HyperPy in research or scientific paper, please cite this page as the resource. HyperPy’s first stable release 0.0.5 was made publicly available in October 2021. 
py-hyperpy.readthedocs. HyperPy, October 2021. URL https://py-hyperpy.readthedocs.io/en/latest/. HyperPy version 0.0.5.

A formatted version of the citation would look like this::

    @Manual{HyperPy,
      author  = {Mora, Sergio},
      title   = {HyperPy: An automatic hyperparameter optimization framework in Python},
      year    = {2021},
      month   = {October},
      note    = {HyperPy version 0.0.5},
      url     = {https://py-hyperpy.readthedocs.io/en/latest/}
    }

We are appreciated that HyperPy has been increasingly referred and cited in scientific works. See all citations here: https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=hyperpy&btnG=


**Key Links and Resources**\ :

* `Release Notes <https://github.com/sergiomora03/py-hyperpy/releases>`_
* `Example Notebooks <https://github.com/sergiomora03/py-hyperpy/tree/master/examples>`_
* `Blog Posts <https://github.com/sergiomora03/py-hyperpy/tree/master/resources>`_
* `Contribute <https://github.com/sergiomora03/py-hyperpy/blob/master/CONTRIBUTING.md>`_
* `More about HyperPy <https://py-hyperpy.readthedocs.io/en/latest/>`_


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   install
   usage
   tutorials
   contribute
   modules

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api/classification
   api/regression
   api/nlp
   api/datasets