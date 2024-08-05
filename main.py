import argparse
import csv
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Union

import dill
import joblib
import optuna
import pandas as pd
import yaml
from fastai.tabular.all import TabularPandas
from hyperopt.pyll.base import scope
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fastai_train import load_and_prepare_fastai_data, prepare_fastai_data, train_fastai_with_optuna
from log_config import logger
from skl_train import run_hyperopt, train_and_evaluate_best_params


class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

    def items(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}.items()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data):
        def convert(obj):
            if isinstance(obj, dict):
                return cls(**{k: convert(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        return convert(data)


def load_config(path):
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config.from_dict(config_dict)


def verify_dataset(df, target):
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")


def load_and_split_data(
    path: str, split_method: str, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    logger.info(f"Raw data shape: {df.shape}")

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


def extract_date_info(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Convert 'Date' column to datetime if it's not already
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    # Extract date features
    for df in [train_df, test_df]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

    # Drop the original 'Date' column
    X_train = train_df.drop([target, 'Date'], axis=1)
    X_test = test_df.drop([target, 'Date'], axis=1)
    y_train = train_df[target]
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def encode(problem_type: str, y_train, y_test):
    if problem_type == 'classification':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    return y_train, y_test


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
    train_df, test_df = load_and_split_data(path, split_method, test_size, random_state)

    verify_dataset(train_df, target)

    if is_fastai:
        return prepare_fastai_data(train_df, test_df, target, problem_type)
    else:
        X_train, X_test, y_train, y_test = extract_date_info(train_df, test_df, target)
        y_train, y_test = encode(problem_type, y_train, y_test)
        return X_train, X_test, y_train, y_test


def save_results(results: Dict[str, Any], timestamp: str):
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


def save_model(model, model_name: str, timestamp: str):
    os.makedirs('models', exist_ok=True)
    model_filename = f'models/best_{model_name.lower()}_{timestamp}.joblib'
    joblib.dump(model, model_filename)
    logger.info(f"Model saved successfully to {model_filename}")


def main(args):
    run_config = load_config(args.run_config_path)
    optim_config = load_config('configs/optimization_config.yaml')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in run_config.models:
        try:
            is_fastai = model_name == 'fastai_tabular'
            if is_fastai:
                data = load_and_prepare_fastai_data(
                    args.data_path,
                    args.target,
                    run_config.problem_type,
                )
                results, model = train_fastai_with_optuna(data, run_config, optim_config)
                save_results(results, timestamp)
                save_model(model, model_name, timestamp)
            else:
                X_train, X_test, y_train, y_test = load_and_prepare_data(
                    args.data_path,
                    args.target,
                    run_config.split_method,
                    run_config.problem_type,
                )
                best_hyperparams = run_hyperopt(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    run_config.problem_type,
                    run_config.num_trials,
                    optim_config,
                )
                results, model = train_and_evaluate_best_params(
                    model_name,
                    best_hyperparams,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    run_config.problem_type,
                    optim_config,
                )
                save_results(results, timestamp)
                save_model(model, model_name, timestamp)

            logger.info(f"Best {model_name} model has been saved in the 'models' directory.")
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.error(f"Failed to optimize and train {model_name}: {str(e)} in {file_name}, line {line_number}")

    logger.info("Process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the target variable column")
    parser.add_argument("--run_config_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
