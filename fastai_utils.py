from typing import Any, Dict, Tuple, Union

import dill
import numpy as np
import optuna
import pandas as pd
from fastai.tabular.all import *
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from pyxtend import struct
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


def load_fastai_data(
    path: str,
    target: str,
    split_method: str,
    logger,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    This doesn't split the data anymore
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

    return df


def load_and_prepare_fastai_data(
    path: str,
    target: str,
    split_method: str,
    problem_type: str,
    logger,
    test_size: float = 0.25,
    random_state: int = 42,
):
    df = load_fastai_data(
        path,
        target,
        split_method,
        logger,
        test_size,
        random_state,
    )

    return prepare_fastai_data(df, target, problem_type)


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

    return data, df[target]


def train_fastai_with_optuna(data, problem_type, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, problem_type), n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Train the final model with the best parameters
    dls = data.dataloaders(bs=best_params['bs'])
    learn = tabular_learner(
        dls,
        layers=best_params['layers'],
        emb_drop=best_params['emb_drop'],
        ps=best_params['ps'],
        metrics=accuracy if problem_type == 'classification' else rmse,
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
