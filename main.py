import argparse
import csv
import logging
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import optuna
import pandas as pd
from fastai.tabular.all import *
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model spaces
model_spaces = {
    'xgboost': {
        'model': XGBClassifier,
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
        'model': RandomForestClassifier,
        'hyperopt_space': {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 700, 1)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
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
    'logistic_regression': {
        'model': LogisticRegression,
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
        'model': tabular_learner,
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


def load_and_prepare_data(
    path: str, target: str, split_method: str, test_size: float = 0.25, random_state: int = 42, is_fastai: bool = False
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], tuple[TabularPandas, TabularPandas, pd.Series, pd.Series]
]:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

    # Identify categorical and continuous columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cont_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cont_cols.remove(target)  # Remove target from continuous columns
    if 'Date' in cont_cols:
        cont_cols.remove('Date')  # Remove Date from continuous columns

    if split_method == 'date':
        # Sort the dataframe by date
        df = df.sort_values('Date')

        # Calculate the split point
        split_index = int(len(df) * (1 - test_size))
        split_date = df.iloc[split_index]['Date']

        logger.info(f"Splitting data by date. Split date: {split_date}")

        train_df = df[df['Date'] < split_date]
        test_df = df[df['Date'] >= split_date]
    else:
        logger.info(f"Splitting data randomly with test_size={test_size}")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    logger.info(f"Data prepared. Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")

    if split_method == 'date':
        logger.info(f"Train date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        logger.info(f"Test date range: {test_df['Date'].min()} to {test_df['Date'].max()}")

    if is_fastai:
        # Create FastAI TabularPandas objects
        procs = [Categorify, FillMissing, Normalize]
        train_data = TabularPandas(
            train_df, procs=procs, cat_names=cat_cols, cont_names=cont_cols, y_names=target, splits=None
        )
        test_data = TabularPandas(
            test_df, procs=procs, cat_names=cat_cols, cont_names=cont_cols, y_names=target, splits=None
        )
        return train_data, test_data, train_df[target], test_df[target]
    else:
        # Return pandas DataFrames for non-FastAI models
        X_train = train_df.drop([target, 'Date'], axis=1)
        X_test = test_df.drop([target, 'Date'], axis=1)
        y_train = train_df[target]
        y_test = test_df[target]
        return X_train, X_test, y_train, y_test


from fastai.tabular.all import *
from sklearn.metrics import accuracy_score, mean_squared_error


def train_model(params: Dict[str, Any], model_class, X_train, y_train, X_test, y_test):
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:
        if model_class == tabular_learner:
            # FastAI specific training
            procs = [Categorify, FillMissing, Normalize]
            cont_names = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_names = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            dls = TabularDataLoaders.from_df(
                X_train, y_name='r_posn', cat_names=cat_names, cont_names=cont_names, procs=procs, bs=params['bs']
            )
            model = tabular_learner(dls, layers=params['layers'], emb_drop=params['emb_drop'], ps=params['ps'])
            model.fit_one_cycle(5)  # You might want to make the number of epochs configurable
            preds, _ = model.get_preds(dl=dls.test_dl(X_test))
            loss = mean_squared_error(y_test, preds)
        else:
            # Sklearn and XGBoost training
            model = model_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Check if it's a classifier or regressor
            if hasattr(model, "predict_proba"):
                # It's a classifier
                loss = 1 - accuracy_score(y_test, preds)
            else:
                # It's a regressor
                loss = mean_squared_error(y_test, preds)

        logger.info(f"Achieved loss: {loss}")
        return {'loss': loss, 'status': STATUS_OK}

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return {'status': STATUS_OK, 'loss': np.inf}


def run_hyperopt(model_name: str, X_train, y_train, X_test, y_test, num_trials: int = 50):
    logger.info(f"Starting Hyperopt optimization for {model_name}")
    model_info = model_spaces[model_name]
    trials = Trials()

    def objective(params):
        result = train_model(params, model_info['model'], X_train, y_train, X_test, y_test)
        return result

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

        # Log the number of trials completed
        logger.info(f"Completed {len(trials.trials)} trials for {model_name}")

        # Log the best score achieved
        best_score = min([t['result']['loss'] for t in trials.trials])
        logger.info(f"Best score achieved for {model_name}: {best_score}")

    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        raise

    return best_hyperparams


def train_and_evaluate(model_name: str, best_hyperparams: Dict[str, Any], X_train, y_train, X_test, y_test):
    logger.info(f"Training final {model_name} model with best hyperparameters")
    model_class = model_spaces[model_name]['model']
    model = model_class(**best_hyperparams)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    logger.info(f"Final model performance - F1 Score: {f1}, Accuracy: {acc}")
    return {'model': model_name, 'f1': f1, 'acc': acc, **best_hyperparams}


def save_results(results: Dict[str, Any]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{results['model']}_results_{timestamp}.csv"
    logger.info(f"Saving results to {filename}")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)


def main(args):
    for model_name in args.models:
        try:
            is_fastai = model_name == 'fastai_tabular'
            data = load_and_prepare_data(
                args.data_path, args.target, args.split_method, args.test_size, args.random_state, is_fastai
            )
            if is_fastai:
                X_train, X_test, y_train, y_test = data
            else:
                X_train, X_test, y_train, y_test = data

            best_hyperparams = run_hyperopt(model_name, X_train, y_train, X_test, y_test, args.num_trials)
            results = train_and_evaluate(model_name, best_hyperparams, X_train, y_train, X_test, y_test)
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
        default=['xgboost', 'random_forest', 'logistic_regression', 'fastai_tabular'],
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

    args = parser.parse_args()
    main(args)
