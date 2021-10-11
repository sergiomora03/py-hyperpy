# hyperapy
HyperaPy: An automatic hyperparameter optimization framework

![PyPI - Status](https://img.shields.io/pypi/status/Bannerquery) ![GitHub top language](https://img.shields.io/github/languages/top/sergiomora03/BannerQuery) ![GitHub](https://img.shields.io/github/license/sergiomora03/BannerQuery) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Bannerquery) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/BannerQuery)

![](img/logo.svg)

# Description

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/sergiomora03/BannerQuery/issues) 

Library for automatic hyperparameter optimization. Build on top of Optuna to perform hyperparameter optimization with low code.

This library corresponds to part of the work of [Sergio A. Mora Pardo](https://sergiomora03.github.io/)

👶 Our current version: [![PyPI version](https://badge.fury.io/py/BannerQuery.svg)](https://badge.fury.io/py/BannerQuery) 

# Installation

![GitHub Release Date](https://img.shields.io/github/release-date/sergiomora03/BannerQuery) ![GitHub last commit](https://img.shields.io/github/last-commit/sergiomora03/BannerQuery)

You can install ```hyperpy``` with pip:

```
# pip install hyperpy
```

# Example

Import the library:

```py
import hyperapy.core as hy
from hyperapy.util import ExampleConfig # Just for example
```

Reading data:

```py
data=ExampleConfig()
train, test, sub = data.readData()
```

Extract features:

```py
feat_X = train.filter(['Pclass','Age', 'SibSp', 'Parch','Fare']).values
Y = train.Survived.values
```

Run the optimization:

```py
running=hy.run(feat_X, Y)
study = running.buildStudy()
```

See the results:

```py
print("best params: ", study.best_params)
print("best test accuracy: ", study.best_value)
best_params, best_value = hy.results.results(study)
```

**NOTE**
best test accuracy -> 'Adam':  0.7407407164573669


# Documentation

Documentation is available at [hyperpy](https://hyperapy.readthedocs.io/en/latest/)

Working on tutorial, meanwhile explore documentation.

# Development ![GitHub issues](https://img.shields.io/github/issues/sergiomora03/BannerQuery) ![GitHub issues](https://img.shields.io/github/issues-raw/sergiomora03/BannerQuery) 

Source code is available at https://github.com/sergiomora03/BannerQuery

https://packaging.python.org/tutorials/creating-documentation/

# Welcome, I've been expecting you.
![](./img/wellcome.svg)

# Contact

<div class="github-card" data-github="sergiomora03" data-width="400" data-height="150" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

---

<a href="https://www.buymeacoffee.com/sergiomorapardo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" style="height: 51px !important;width: 217px !important;" ></a>