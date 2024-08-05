import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import dill
import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    EarlyStoppingCallback,
    FillMissing,
    Normalize,
    RegressionBlock,
    TabularPandas,
)
from fastai.tabular.all import accuracy as fai_accuracy
from fastai.tabular.all import cont_cat_split, rmse, tabular_config, tabular_learner
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from pyxtend import struct
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from fastai_train import load_and_prepare_fastai_data, prepare_fastai_data, train_fastai_with_optuna
from log_config import logger


def train_model(params: Dict[str, Any], model_class, X_train, X_test, y_train, y_test, problem_type: str, target: str):
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:
        if model_class == tabular_learner:
            # might remove this
            # FastAI specific training
            logger.warn("FastAI not set up for hyperopt")
        else:
            # Sklearn and XGBoost training
            model = model_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if problem_type == 'classification':
                loss = 1 - accuracy_score(y_test, preds)
            else:
                loss = mean_squared_error(y_test, preds)

        logger.info(f"Achieved loss: {loss}")
        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return {'loss': np.inf, 'status': STATUS_FAIL, 'error': str(e)}


def run_hyperopt(model_name: str, X_train, y_train, X_test, y_test, problem_type: str, num_trials: int = 50, target=''):
    logger.info(f"Starting Hyperopt optimization for {model_name}")
    model_info = model_spaces[model_name]
    trials = Trials()

    def objective(params):
        try:
            logger.info(f"Starting objective function with params: {params}")
            model_class = model_info['classifier'] if problem_type == 'classification' else model_info['regressor']
            result = train_model(params, model_class, X_train, X_test, y_train, y_test, problem_type, target)
            logger.info(f"Finished objective function. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return {'loss': np.inf, 'status': STATUS_FAIL}

    try:
        best = fmin(
            fn=objective,
            space=model_info['hyperopt_space'],
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            show_progressbar=False,
        )

        best_hyperparams = space_eval(model_info['hyperopt_space'], best)
        logger.info(f"Best hyperparameters for {model_name}: {best_hyperparams}")

        logger.info(f"Completed {len(trials.trials)} trials for {model_name}")

        best_score = min([t['result']['loss'] for t in trials.trials if t['result']['status'] == STATUS_OK])
        logger.info(f"Best score achieved for {model_name}: {best_score}")

    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        return None

    return best_hyperparams


def train_and_evaluate_best_params(
    model_name: str, best_hyperparams: Dict[str, Any], X_train, y_train, X_test, y_test, problem_type: str
):
    logger.info(f"Training final {model_name} model with best hyperparameters")
    model_info = model_spaces[model_name]

    if model_name == 'fastai_tabular':
        # FastAI specific training and evaluation
        dls = X_train.dataloaders(bs=best_hyperparams['bs'])

        learn = tabular_learner(dls)
        learn.fit_one_cycle(5)
        test_dl = learn.dls.test_dl(X_test)
        preds, _ = learn.get_preds(dl=test_dl)
        if problem_type == 'regression':
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            logger.info(f"Final model performance - MSE: {mse}, R2 Score: {r2}")
            return {'model': model_name, 'mse': mse, 'r2': r2, **best_hyperparams}, learn
        else:
            accuracy = accuracy_score(y_test, preds.argmax(dim=1))
            f1 = f1_score(y_test, preds.argmax(dim=1), average='weighted')
            logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
            return {'model': model_name, 'accuracy': accuracy, 'f1': f1, **best_hyperparams}, learn
    else:
        # Sklearn and XGBoost training and evaluation
        model_class = model_info['classifier'] if problem_type == 'classification' else model_info['regressor']
        model = model_class(**best_hyperparams)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if problem_type == 'regression':
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            logger.info(f"Final model performance - MSE: {mse}, R2 Score: {r2}")
            return {'model': model_name, 'mse': mse, 'r2': r2, **best_hyperparams}, model
        else:
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
            return {'model': model_name, 'accuracy': accuracy, 'f1': f1, **best_hyperparams}, model
