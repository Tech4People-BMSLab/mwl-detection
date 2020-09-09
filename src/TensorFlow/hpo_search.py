"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-08-10 15:12:44
@Last Modified by:   Tenzing Dolmans
@Last Modified time: 2020-08-10 16:14:37
@Description: Script for HPO search using the optuna toolbox.
Currently varies lr, momentum, dropout, and model type.
"""
import os

import numpy as np
import optuna
import joblib
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from trainer import Trainer  # noqa
from Models.mlp_base import HeadClass as MlpClass  # noqa
from Models.small_mlp_base import HeadClass as s_MlpClass  # noqa
from Models.literature_base import HeadClass as LitClass  # noqa
from Models.small_literature_base import HeadClass as s_LitClass  # noqa


def cross_validation(num_epochs, batch_size, dropout_rate,
                     min_lr, max_lr, min_mom, max_mom, model):
    """Call the trainer and perform k-fold cross-validation.
    Returns mean results of all performed folds."""
    n_split = 5
    dropout_rate = 0.2
    trainer = Trainer()
    train_dataset = trainer.get_dataset()

    folds = []
    for k in range(n_split):
        folds.append(train_dataset.shard(num_shards=n_split, index=k))

    for i in range(n_split):
        results = []
        print('Starting with fold %i of %i' % (i + 1, n_split))

        trainer = Trainer(
            num_epochs, batch_size, dropout_rate,
            min_lr, max_lr, min_mom, max_mom)

        if model == 'LIT':
            trainer.model = LitClass(dropout_rate)
        elif model == "S_MLP":
            trainer.model = s_MlpClass(dropout_rate)
        elif model == "S_LIT":
            trainer.model = s_LitClass(dropout_rate)

        folds_train = folds[:n_split - 1 - i] + folds[n_split - i:]
        train_dataset_fold = folds_train[0]

        for j in range(n_split - 2):
            train_dataset_fold = train_dataset_fold.concatenate(
                folds_train[j + 1])

        test_dataset_fold = folds[n_split - 1 - i]
        returns = trainer.training_loop(train_dataset_fold, test_dataset_fold)
        objective = returns[-1]
        results.append(objective)
    return np.mean(results)


def objective(trial):
    """Define parameters which to search over and
    optimise the indicated objective."""
    num_epochs = 25
    batch_size = 8
    joblib.dump(study, study.folder + study.study_name + '.pkl')

    min_lr = trial.suggest_loguniform('min_lr', 1e-8, 0.005)
    max_lr = trial.suggest_loguniform('max_lr', min_lr, 0.1)
    min_mom = trial.suggest_uniform('min_mom', 0.7, 0.95)
    max_mom = trial.suggest_uniform('max_mom', min_mom, 0.99)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0., 0.2)
    model = trial.suggest_categorical('model',
                                      ['MLP', 'LIT', 'S_MLP', 'S_LIT'])

    mean_accuracy_trial = cross_validation(
        num_epochs, batch_size, dropout_rate,
        min_lr, max_lr, min_mom, max_mom, model)
    return mean_accuracy_trial


if __name__ == "__main__":
    study_name = 'Name_of_study'
    folder = 'Location/of/output'
    if os.path.isfile(folder + study_name + '.pkl'):
        study = joblib.load(folder + study_name + '.pkl')
    else:
        study = optuna.create_study(
            direction='minimize', study_name=study_name)
    study.folder = folder

    study.optimize(objective, n_trials=10)
    joblib.dump(study, study.folder + study.study_name + '.pkl')
