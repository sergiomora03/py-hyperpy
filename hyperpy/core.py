#!/usr/bin/python3
#-*- coding: utf-8 -*-

# Author: Sergio A. Mora Pardo <sergiomora823@gmail.com>
# Project: hyperapy

import os
import keras
import optuna
import plotly
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential, load_model
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, GlobalAveragePooling2D,
                          MaxPooling2D)

warnings.filterwarnings("ignore")
np.random.seed(1)
tf.random.set_seed(1)


# TODO : add a function to plot calibration of model
# TODO : add smart feature engeneering to the model
# TODO : add diferent type of trainers. Ex: cros validation, doble cross validation, etc.
# TODO : cronstruir varios tipos de topologías predeterminadas para las redes neuronales.

class models:
    def __init__(self,initnorm=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)) -> None:
        self.initnorm=initnorm

    def BuildModelSimply(trial:optuna.Trial, self) -> None:
        n_layers = trial.suggest_int('n_layers', 1, 13)
        activation_selected = trial.suggest_categorical(f"activation_units_layer_{i}", ["selu", "sigmoid", "tanh"])
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int(f"n_units_layer_{i}", 4, 128, log=True)
            model.add(Dense(num_hidden, activation=activation_selected, kernel_initializer=self.initnorm, bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=self.initnorm, bias_initializer='zeros'))
        return model

    def BuildModel(trial:optuna.Trial, self) -> None:
        n_layers = trial.suggest_int('n_layers', 1, 13)
        model = Sequential()
        for i in range(n_layers):
            activation_selected = trial.suggest_categorical(f"activation_units_layer_{i}", ["selu", "sigmoid", "tanh"])
            num_hidden = trial.suggest_int(f"n_units_layer_{i}", 4, 128, log=True)
            model.add(Dense(num_hidden, activation=activation_selected, kernel_initializer=self.initnorm, bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=self.initnorm, bias_initializer='zeros'))
        return model

class optimizers:
    def optimizerAdam(trial:optuna.Trial) -> None:
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        beta_1 = trial.suggest_loguniform('beta_1', 0.0001, 0.9)
        beta_2 = trial.suggest_loguniform('beta_2', 0.0001, 0.9)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        return optimizer

    def optimizerRMSprop(trial:optuna.Trial) -> None:
        learning_rate = trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
        decay = trial.suggest_float("decay", 0.85, 0.99)
        momentum = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum, rho=decay)
        return optimizer

    def optimizerSGD(trial:optuna.Trial) -> None:
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        return optimizer


    def buildOptimizer(trial:optuna.Trial) -> None:
        kwargs = {}
        optimizer_options = ["RMSprop", "Adam", "SGD"]
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        if optimizer_selected == "RMSprop":
            kwargs["learning_rate"] = trial.suggest_float(
                "rmsprop_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
            kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adam":
            kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
            kwargs["beta_1"] = trial.suggest_loguniform('beta_1', 0.0001, 0.9)
            kwargs["beta_2"] = trial.suggest_loguniform('beta_2', 0.0001, 0.9)
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = trial.suggest_float(
                "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)
        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
        return optimizer

class trainers():
    def __init__(self,trial,feat_X,Y,verbose:int=0,model:models=models, optimizer:optimizers=optimizers, type:str="Build", initnorm=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)) -> None:
        self.trial=trial
        self.feat_X=feat_X
        self.Y=Y
        self.verbose=verbose
        self.model=model
        self.optimizer=optimizer
        self.type=type
        self.initnorm=initnorm

    def trainer(self,save:bool=False) -> None: # trainer Method define how to train Neural Network. This works by maximizing the test data set (Exactitud de Validación)
        table = PrettyTable(["Exac_E", "Exac_V", "Exac_P", "Epoca", "Optimizer"])
        err_p = 999
        print('printing...',self.feat_X, self.Y)
        for i in range(0,3,1):
            r = i^3
            CE_x, CV0_x, CE_y, CV0_y = train_test_split(self.feat_X, self.Y, test_size = 0.3, random_state = r)
            CV_x, CP_x, CV_y, CP_y = train_test_split(CV0_x, CV0_y, test_size = 0.5, random_state = r)
            epocas = self.trial.suggest_categorical('epocas', [100, 200, 300])
            model = self.model.BuildModel(self.trial, self)
            if self.type=="Adam":
                optimus = self.optimizer.optimizerAdam(self.trial)
            elif self.type=="RMSprop":
                optimus = self.optimizer.optimizerRMSprop(self.trial)
            elif self.type=="SGD":
                optimus = self.optimizer.optimizerSGD(self.trial)
            else:
                optimus = self.optimizer.buildOptimizer(self.trial)
            model.compile(loss='binary_crossentropy', optimizer=optimus, metrics=['accuracy'])
            history=model.fit(x=CE_x, y=CE_y, epochs=epocas, validation_data=(CV_x, CV_y), verbose=0, shuffle=False)
            #print(history.history)
            min_err=np.min(history.history['val_loss'])
            best_epoc=np.where(history.history['val_loss'] == min_err)[0]
            model.fit(x=CE_x, y=CE_y, epochs=best_epoc[0], validation_data=(CV_x, CV_y), verbose=0, shuffle=False)
            train_metrics = model.evaluate(x=CE_x, y=CE_y, verbose=0)
            valid_metrics = model.evaluate(x=CV_x, y=CV_y, verbose=0)
            test_metrics = model.evaluate(x=CP_x, y=CP_y, verbose=0)
            accu_e = train_metrics[1]
            loss_e = train_metrics[0]
            accu_v = valid_metrics[1]
            loss_v = valid_metrics[0]
            accu_p = test_metrics[1]
            loss_p = test_metrics[0]
            if save:
                if (loss_p < err_p):
                    pathr = (os.path.join(os.getcwd(),'model',f'{self.study_name}-partseed_{str(r)}.h5'))
                    model.save(pathr)
                    err_p = loss_p
            print('Epoca= '+str(best_epoc[0])+' , accu_v1='+str(accu_v) +' , accu_v2='+str(accu_p) + ' , Optimizer=' + str(optimus.get_config()["name"])) if self.verbose > 0 else None
            table.add_row([np.round(accu_e,4), np.round(accu_v,4), np.round(accu_p,4), best_epoc[0], optimus.get_config()["name"]])
        print(table) if self.verbose > 0 else None
        return model, CV_x, CV_y

class run():

    def __init__(self,feat_X,Y,study_name:str='First try', direction:str="maximize", n_trials=10) -> None:
        self.features=feat_X
        self.target=Y
        self.study_name=study_name
        self.direction=direction
        self.n_trials=n_trials

    def objective(self, trial):
        model, CV_x, CV_y = trainers(trial,self.features,self.target).trainer()
        model.save(os.path.join(os.getcwd(),'model',f'{self.study_name}-trial={trial.number}.h5'))
        evaluate = model.evaluate(x=CV_x, y=CV_y, verbose=0)
        return evaluate[1]

    def buildStudy(self):
        study=optuna.create_study(study_name=self.study_name, direction=self.direction)
        study.optimize(func=self.objective,n_trials=self.n_trials,n_jobs=-1,show_progress_bar=True)
        return study

class results:
    def results(study):
        print("best params: ", study.best_params)
        print("best test accuracy: ", study.best_value)
        return study.best_params, study.best_value

