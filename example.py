#!/usr/bin/python3
#-*- coding: utf-8 -*-

# Author: Sergio A. Mora Pardo <sergiomora823@gmail.com>
# Project: hyperapy

import hyperpy.core as hy
from hyperpy.util import ExampleConfig

data=ExampleConfig()
train, test, sub = data.readData()


feat_X = train.filter(['Pclass','Age', 'SibSp', 'Parch','Fare']).values
Y = train.Survived.values

running=hy.run(feat_X, Y)
study = running.buildStudy()

print("best params: ", study.best_params)
print("best test accuracy: ", study.best_value)
best_params, best_value = hy.results.results(study)

# NOTA
# best test accuracy -> 'Adam':  0.7407407164573669
