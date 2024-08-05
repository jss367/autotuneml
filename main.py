import argparse
import csv
import logging
import os
from datetime import datetime
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model spaces
model_spaces = {
    'xgboost': {
        'classifier': XGBClassifier,
        'regressor': XGBRegressor,
        'hyperopt_space': {
            'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': 180,
            'seed': 0,
        },
        'optuna_space': {
            'max_depth': lambda trial: trial.suggest_int("max_depth", 3, 18),
            'gamma': lambda trial: trial.suggest_float('gamma', 1, 9),
            'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 40, 180),
            'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0, 1),
            'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1),
            'min_child_weight': lambda trial: trial.suggest_float('min_child_weight', 0, 10),
            'n_estimators': 180,
            'seed': 0,
        },
    },
    'random_forest': {
        'classifier': RandomForestClassifier,
        'regressor': RandomForestRegressor,
        'hyperopt_space': {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 700, 1)),
            "max_depth": scope.int(hp.quniform('max_depth', 1, 100, 1)),
            "min_samples_split": scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            "max_features": hp.choice('max_features', ['sqrt', 'log2']),
            "random_state": 42,
        },
        'optuna_space': {
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 10, 700),
            "criterion": lambda trial: trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": lambda trial: trial.suggest_int('max_depth', 1, 100),
            "min_samples_split": lambda trial: trial.suggest_int('min_samples_split', 2, 20),
            "min_samples_leaf": lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
            "max_features": lambda trial: trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            "random_state": 42,
        },
    },
    'linear': {
        'classifier': LogisticRegression,
        'regressor': LinearRegression,
        'hyperopt_space': {
            'C': hp.loguniform('C', -4, 4),
            'penalty': hp.choice('penalty', ['l1', 'l2']),
            'solver': hp.choice('solver', ['liblinear', 'saga']),
            'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 100)),
        },
        'optuna_space': {
            'C': lambda trial: trial.suggest_loguniform('C', 1e-4, 1e4),
            'penalty': lambda trial: trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': lambda trial: trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': lambda trial: trial.suggest_int('max_iter', 100, 1000, 100),
        },
    },
    'fastai_tabular': {
        'classifier': tabular_learner,
        'regressor': tabular_learner,
        'hyperopt_space': {
            'layers': hp.choice(
                'layers',
                [
                    [200, 100],
                    [500, 200],
                    [1000, 500, 200],
                ],
            ),
            'emb_drop': hp.uniform('emb_drop', 0, 0.5),
            'ps': hp.uniform('ps', 0, 0.5),
            'bs': scope.int(hp.quniform('bs', 32, 256, 32)),
        },
        'optuna_space': {
            'layers': lambda trial: trial.suggest_categorical(
                'layers',
                [
                    [200, 100],
                    [500, 200],
                    [1000, 500, 200],
                ],
            ),
            'emb_drop': lambda trial: trial.suggest_float('emb_drop', 0, 0.5),
            'ps': lambda trial: trial.suggest_float('ps', 0, 0.5),
            'bs': lambda trial: trial.suggest_int('bs', 32, 256, 32),
        },
    },
}


def load_and_split_data(
    path: str, target: str, split_method: str, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

    if split_method == 'date':
        df = df.sort_values('Date')
        split_index = int(len(df) * (1 - test_size))
        split_date = df.iloc[split_index]['Date']
        logger.info(f"Splitting data by date. Split date: {split_date}")
        train_df = df[df['Date'] < split_date]
        test_df = df[df['Date'] >= split_date]
    else:
        logger.info(f"Splitting data randomly with test_size={test_size}")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    logger.info(f"Data prepared. Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
    return train_df, test_df


def prepare_fastai_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, problem_type: str
) -> Tuple[TabularPandas, TabularPandas, pd.Series, pd.Series]:
    """
    We're assuming the data has already been split into test and train
    """

    continuous_vars, categorical_vars = cont_cat_split(train_df, dep_var=target)

    preprocessing = [Categorify, FillMissing, Normalize]

    if problem_type == 'regression':
        y_block = RegressionBlock()
    else:
        y_block = CategoryBlock()

    train_data = TabularPandas(
        train_df,
        procs=preprocessing,
        cat_names=categorical_vars,
        cont_names=continuous_vars,
        y_names=target,
        y_block=y_block,
        splits=None,
    )
    test_data = TabularPandas(
        test_df,
        procs=preprocessing,
        cat_names=categorical_vars,
        cont_names=continuous_vars,
        y_names=target,
        y_block=y_block,
        splits=None,
    )
    return train_data, test_data, train_df[target], test_df[target]


def prepare_other_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, problem_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = train_df.drop([target, 'Date'], axis=1)
    X_test = test_df.drop([target, 'Date'], axis=1)
    y_train = train_df[target]
    y_test = test_df[target]

    if problem_type == 'classification':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    path: str,
    target: str,
    split_method: str,
    problem_type: str,
    test_size: float = 0.25,
    random_state: int = 42,
    is_fastai: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], Tuple[TabularPandas, TabularPandas, pd.Series, pd.Series]
]:
    train_df, test_df = load_and_split_data(path, target, split_method, test_size, random_state)

    if is_fastai:
        return prepare_fastai_data(train_df, test_df, target, problem_type)
    else:
        return prepare_other_data(train_df, test_df, target, problem_type)


def train_model(params: Dict[str, Any], model_class, X_train, y_train, X_test, y_test, problem_type: str, target: str):
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:
        if model_class == tabular_learner:
            # FastAI specific training
            dls = X_train.dataloaders(bs=params['bs'])

            learn = tabular_learner(dls, layers=list(params['layers']))
            learn.fit_one_cycle(5)
            test_dl = learn.dls.test_dl(X_test)
            preds, _ = learn.get_preds(dl=test_dl)
            if problem_type == 'regression':
                loss = mean_squared_error(y_test, preds)
            else:
                loss = 1 - accuracy_score(y_test, preds.argmax(dim=1))
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
            result = train_model(params, model_class, X_train, y_train, X_test, y_test, problem_type, target)
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


def train_and_evaluate(
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
            return {'model': model_name, 'mse': mse, 'r2': r2, **best_hyperparams}
        else:
            accuracy = accuracy_score(y_test, preds.argmax(dim=1))
            f1 = f1_score(y_test, preds.argmax(dim=1), average='weighted')
            logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
            return {'model': model_name, 'accuracy': accuracy, 'f1': f1, **best_hyperparams}
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
            return {'model': model_name, 'mse': mse, 'r2': r2, **best_hyperparams}
        else:
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
            return {'model': model_name, 'accuracy': accuracy, 'f1': f1, **best_hyperparams}


def save_results(results: Dict[str, Any]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{results['model']}_results_{timestamp}.csv"
    try:
        logger.info(f"Saving results to {filename}")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        logger.info(f"Results saved successfully to {filename}")
    except IOError as e:
        logger.error(f"Error saving results to {filename}: {str(e)}")


def train_fastai(train_df_wrapper):

    batch_size = 8  # 128
    dls = train_df_wrapper.dataloaders(bs=batch_size)

    learn = tabular_learner(dls, layers=[20, 10])

    learn.fit(4, 1e-2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/model_{timestamp}.pkl"
    os.makedirs('results', exist_ok=True)
    learn.export(filename, pickle_module=dill)


def main(args):
    os.makedirs('results', exist_ok=True)

    for model_name in args.models:
        try:
            is_fastai = model_name == 'fastai_tabular'
            X_train, X_test, y_train, y_test = load_and_prepare_data(
                args.data_path,
                args.target,
                args.split_method,
                args.problem_type,
                args.test_size,
                args.random_state,
                is_fastai,
            )

            if is_fastai:
                # if is_fastai, don't use optimization
                train_fastai(X_train)

            else:
                best_hyperparams = run_hyperopt(
                    model_name, X_train, y_train, X_test, y_test, args.problem_type, args.num_trials
                )
                results = train_and_evaluate(
                    model_name, best_hyperparams, X_train, y_train, X_test, y_test, args.problem_type
                )
                save_results(results)
        except Exception as e:
            logger.error(f"Failed to optimize and train {model_name}: {str(e)}")

    logger.info("Process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the target variable column")
    parser.add_argument(
        "--models",
        nargs='+',
        default=['xgboost', 'random_forest', 'linear', 'fastai_tabular'],
        help="List of models to train and evaluate",
    )
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials for hyperparameter optimization")
    parser.add_argument(
        "--test_size", type=float, default=0.25, help="Proportion of the dataset to include in the test split"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument(
        "--split_method",
        type=str,
        choices=['random', 'date'],
        default='random',
        help="Method to split the data into train and test sets",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=['classification', 'regression'],
        required=True,
        help="Type of machine learning problem",
    )

    args = parser.parse_args()
    main(args)
