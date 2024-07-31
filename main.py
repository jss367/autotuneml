import argparse
import csv
import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
        'space': {
            'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': 180,
            'seed': 0,
            'eval_metric': 'logloss',
        },
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'space': {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 700, 1)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": scope.int(hp.quniform('max_depth', 1, 100, 1)),
            "min_samples_split": scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            "max_features": hp.choice('max_features', ['sqrt', 'log2']),
            "random_state": 42,
        },
    },
    'logistic_regression': {
        'model': LogisticRegression,
        'space': {
            'C': hp.loguniform('C', -4, 4),
            'penalty': hp.choice('penalty', ['l1', 'l2']),
            'solver': hp.choice('solver', ['liblinear', 'saga']),
            'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 100)),
        },
    },
}


def load_and_prepare_data(
    path: str, target: str, split_method: str, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")

    if split_method == 'date':
        # Sort the dataframe by date
        df = df.sort_values('Date')

        # Calculate the split point
        split_index = int(len(df) * (1 - test_size))
        split_date = df.iloc[split_index]['Date']

        logger.info(f"Splitting data by date. Split date: {split_date}")

        train_df = df[df['Date'] < split_date]
        test_df = df[df['Date'] >= split_date]

        X_train = train_df.drop([target, 'Date'], axis=1)
        X_test = test_df.drop([target, 'Date'], axis=1)
        y_train = train_df[target]
        y_test = test_df[target]
    else:
        logger.info(f"Splitting data randomly with test_size={test_size}")
        X = df.drop([target, 'Date'], axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    logger.info(f"Data prepared. Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    if split_method == 'date':
        logger.info(f"Train date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        logger.info(f"Test date range: {test_df['Date'].min()} to {test_df['Date'].max()}")

    return X_train, X_test, y_train, y_test


def train_model(params: Dict[str, Any], model_class, X_train, y_train, X_test, y_test):
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:
        model = model_class(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        logger.info(f"Achieved accuracy: {accuracy}")
        return {'loss': -accuracy, 'status': STATUS_OK}
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return {'status': STATUS_OK, 'loss': np.inf}


def run_hyperopt(model_name: str, X_train, y_train, X_test, y_test, num_trials: int = 50):
    logger.info(f"Starting hyperparameter optimization for {model_name}")
    model_info = model_spaces[model_name]
    trials = Trials()

    fmin_function = lambda params: train_model(params, model_info['model'], X_train, y_train, X_test, y_test)

    try:
        best = fmin(
            fn=fmin_function,
            space=model_info['space'],
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            show_progressbar=False,
        )

        best_hyperparams = space_eval(model_info['space'], best)
        logger.info(f"Best hyperparameters for {model_name}: {best_hyperparams}")

        return best_hyperparams
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        raise


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
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        args.data_path, args.target, args.split_method, args.test_size, args.random_state
    )

    for model_name in args.models:
        try:
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
        default=['xgboost', 'random_forest', 'logistic_regression'],
        help="List of models to train and evaluate",
    )
    parser.add_argument("--num_trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
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
