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

from data import load_and_prepare_data
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
