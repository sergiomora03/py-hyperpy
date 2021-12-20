Core
=======

.. _models:

Models
------------

class models:
    """
     Class to build a model with a given topology
    """
    def __init__(self,initnorm=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)) -> None:
        self.initnorm=initnorm

    def BuildModelSimply(trial:optuna.Trial, self) -> keras.models.Model:
        """
        BuildModelSimply Standar model

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: sequential model
        :rtype: keras.models.Model.Sequential
        """
        n_layers = trial.suggest_int('n_layers', 1, 13)
        activation_selected = trial.suggest_categorical(f"activation_units_layer_{i}", ["selu", "sigmoid", "tanh"])
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int(f"n_units_layer_{i}", 4, 128, log=True)
            model.add(Dense(num_hidden, activation=activation_selected, kernel_initializer=self.initnorm, bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=self.initnorm, bias_initializer='zeros'))
        return model

    def BuildModel(trial:optuna.Trial, self) -> keras.models.Model:
        """
        BuildModel Standar model

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: sequential model
        :rtype: keras.models.Model
        """
        n_layers = trial.suggest_int('n_layers', 1, 13)
        model = Sequential()
        for i in range(n_layers):
            activation_selected = trial.suggest_categorical(f"activation_units_layer_{i}", ["selu", "sigmoid", "tanh"])
            num_hidden = trial.suggest_int(f"n_units_layer_{i}", 4, 128, log=True)
            model.add(Dense(num_hidden, activation=activation_selected, kernel_initializer=self.initnorm, bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=self.initnorm, bias_initializer='zeros'))
        return model

class optimizers:
    """
     class to build a model optimizer
    """
    def optimizerAdam(trial:optuna.Trial) -> keras.optimizers.Adam:
        """
        optimizerAdam method to build a model optimizer with Adam

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: optimizer
        :rtype: keras.optimizers.Adam
        """
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        beta_1 = trial.suggest_loguniform('beta_1', 0.0001, 0.9)
        beta_2 = trial.suggest_loguniform('beta_2', 0.0001, 0.9)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        return optimizer

    def optimizerRMSprop(trial:optuna.Trial) -> keras.optimizers.RMSprop:
        """
        optimizerRMSprop method to build a model optimizer with RMSprop

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: optimizer
        :rtype: keras.optimizers.RMSprop
        """
        learning_rate = trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
        decay = trial.suggest_float("decay", 0.85, 0.99)
        momentum = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum, rho=decay)
        return optimizer

    def optimizerSGD(trial:optuna.Trial) -> keras.optimizers.SGD:
        """
        optimizerSGD method to build a model optimizer with SGD

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: optimizer
        :rtype: keras.optimizers.SGD
        """
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        return optimizer

    def buildOptimizer(trial:optuna.Trial) -> None:
        """
        buildOptimizer method to build a model optimizer

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :return: optimizer
        :rtype: keras.optimizers
        """
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
    """
    trainers class to build a model trainer
    """
    def __init__(self,trial,feat_X,Y,verbose:int=0,model:models=models, optimizer:optimizers=optimizers, type:str="Build", initnorm=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)) -> None:
        """
        __init__ method to build a model trainer

        :param trial: trial to build the model
        :type trial: optuna.Trial
        :param feat_X: features to train the model
        :type feat_X: pandas.DataFrame
        :param Y: target to train the model
        :type Y: pandas.Series
        :param verbose: verbose, defaults to 0
        :type verbose: int, optional
        :param model: model to train, defaults to models
        :type model: keras.models, optional
        :param optimizer: optimizer to train model, defaults to optimizers
        :type optimizer: keras.optimizers, optional
        :param type: type build, defaults to "Build"
        :type type: str, optional
        :param initnorm: type of normalization, defaults to keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
        :type initnorm: keras.initializers, optional
        """
        self.trial=trial
        self.feat_X=feat_X
        self.Y=Y
        self.verbose=verbose
        self.model=model
        self.optimizer=optimizer
        self.type=type
        self.initnorm=initnorm

    def trainer(self,save:bool=False) -> None:
        """
        trainer trainer Method define how to train Neural Network. This works by maximizing the test data set (Exactitud de Validación).

        :param save: save model, defaults to False
        :type save: bool, optional
        :return: model, cv_x, cv_y
        :rtype: keras.models, pandas.DataFrame, pandas.Series
        """
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
    """
    run class is used to run the experiment.
    """
    def __init__(self,feat_X,Y,study_name:str='First try', direction:str="maximize", n_trials=10) -> None:
        """
        __init__ class is used to initialize the run class.

        :param feat_X: features of the data set.
        :type feat_X: pandas.DataFrame
        :param Y: labels of the data set.
        :type Y: pandas.Series
        :param study_name: string, defaults to 'First try'
        :type study_name: str, optional
        :param direction: string minimize or maximize, defaults to "maximize"
        :type direction: str, optional
        :param n_trials: trial numbers in study, defaults to 10
        :type n_trials: int, optional
        """
        self.features=feat_X
        self.target=Y
        self.study_name=study_name
        self.direction=direction
        self.n_trials=n_trials

    def objective(self, trial):
        """
        objective function is used to define the objective function.

        :param trial: trial object
        :type trial: optuna.trial.Trial
        :return: objective function
        :rtype: float
        """
        model, CV_x, CV_y = trainers(trial,self.features,self.target).trainer()
        model.save(os.path.join(os.getcwd(),'model',f'{self.study_name}-trial={trial.number}.h5'))
        evaluate = model.evaluate(x=CV_x, y=CV_y, verbose=0)
        return evaluate[1]

    def buildStudy(self):
        """
        buildStudy function is used to build the study.

        :return: study
        :rtype: optuna.study.Study
        """
        study=optuna.create_study(study_name=self.study_name, direction=self.direction)
        study.optimize(func=self.objective,n_trials=self.n_trials,n_jobs=-1,show_progress_bar=True)
        return study

class results:
    """
    results class is used to get the results of the study.
    """
    def results(study):
        """
        results function is used to get the results of the study.

        :param study: study object
        :type study: optuna.study.Study
        :return: results
        :rtype: pandas.DataFrame
        """
        print("best params: ", study.best_params)
        print("best test accuracy: ", study.best_value)
        return study.best_params, study.best_value

