from typing import Any, Dict, Tuple, Union

import dill
import numpy as np
import optuna
import pandas as pd
from fastai.tabular.all import *
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from log_config import logger


def load_and_prepare_fastai_data(
    path: str,
    target: str,
    problem_type: str,
):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

    data = prepare_fastai_data(df, target, problem_type)

    return data


def prepare_fastai_data(df: pd.DataFrame, target: str, problem_type: str):
    """
    This splits the data
    """
    continuous_vars, categorical_vars = cont_cat_split(df, dep_var=target, max_card=20)

    preprocessing = [Categorify, FillMissing, Normalize]

    if problem_type == 'regression':
        y_block = RegressionBlock()
    else:
        y_block = CategoryBlock()

    # Assuming 'Date' is the name of your date column
    def date_splitter(df):
        """
        The splitter returns two lists - one of all the training data indices and one of all the validation data indices
        """
        train_mask = df['Date'] < pd.to_datetime('1/1/2024')
        train_indices = df.index[train_mask].tolist()
        val_indices = df.index[~train_mask].tolist()
        return train_indices, val_indices

    date_splits = date_splitter(df)

    data = TabularPandas(
        df,
        procs=preprocessing,
        cat_names=categorical_vars,
        cont_names=continuous_vars,
        y_names=target,
        y_block=y_block,
        splits=date_splits,
    )

    return data


def fastai_objective(trial, data, problem_type):
    # Define the hyperparameters to optimize
    layers = trial.suggest_categorical('layers', [[200, 100], [500, 200], [1000, 500, 200]])
    ps = trial.suggest_float('ps', 0, 0.5)
    bs = trial.suggest_categorical('bs', [32, 64, 128, 256])

    # Create the DataLoaders
    dls = data.dataloaders(bs=bs)

    # Create and train the model
    config = tabular_config(ps=ps, embed_p=trial.suggest_float('embed_p', 0, 0.5))
    learn = tabular_learner(
        dls, layers=layers, config=config, metrics=accuracy if problem_type == 'classification' else rmse
    )
    learn.fit_one_cycle(5)

    # Evaluate the model
    if problem_type == 'classification':
        preds, _ = learn.get_preds(dl=dls.valid)
        acc = accuracy_score(dls.valid.y.items, preds.argmax(dim=1))
        return -acc  # Optuna minimizes the objective, so we return negative accuracy
    else:
        preds, _ = learn.get_preds(dl=dls.valid)
        mse = mean_squared_error(dls.valid.y.items, preds)
        return mse


def train_fastai_with_optuna(data, problem_type, n_trials=50):
    logger.info("Starting FastAI training with Optuna")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: fastai_objective(trial, data, problem_type), n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Train the final model with the best parameters
    dls = data.dataloaders(bs=best_params['bs'])
    config = tabular_config(ps=best_params['ps'], embed_p=best_params['embed_p'])
    learn = tabular_learner(
        dls, layers=best_params['layers'], config=config, metrics=accuracy if problem_type == 'classification' else rmse
    )
    learn.fit_one_cycle(5)

    # Evaluate the final model
    test_dl = learn.dls.test_dl(data.all_df)
    preds, _ = learn.get_preds(dl=test_dl)

    if problem_type == 'classification':
        accuracy = accuracy_score(data.y.items, preds.argmax(dim=1))
        f1 = f1_score(data.y.items, preds.argmax(dim=1), average='weighted')
        logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
        results = {'model': 'fastai_tabular', 'accuracy': accuracy, 'f1': f1, **best_params}
    else:
        mse = mean_squared_error(data.y.items, preds)
        r2 = r2_score(data.y.items, preds)
        logger.info(f"Final model performance - MSE: {mse}, R2 Score: {r2}")
        results = {'model': 'fastai_tabular', 'mse': mse, 'r2': r2, **best_params}

    return results, learn


def train_fastai(data):
    batch_size = 64
    dls = data.dataloaders(bs=batch_size)

    learn = tabular_learner(dls, layers=[200, 100], metrics=[accuracy])

    learn.fit_one_cycle(4, 1e-2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/model_{timestamp}.pkl"
    os.makedirs('models', exist_ok=True)
    learn.export(filename)
