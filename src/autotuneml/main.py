import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict

import joblib

from autotuneml.configs.config_class import Config, convert_to_config, load_config
from autotuneml.data import load_and_prepare_data
from autotuneml.fastai_train import load_and_prepare_fastai_data, train_fastai_with_optuna
from autotuneml.log_config import logger
from autotuneml.skl_train import run_hyperopt, train_and_evaluate_best_params


def save_results(results: Dict[str, Any], timestamp: str):
    os.makedirs('results', exist_ok=True)
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


def run_autotuneml(data_path, target, run_config_path=None, optim_config_path=None):
    if run_config_path:
        run_config = load_config(run_config_path)
    else:
        default_config_path = 'src/autotuneml/configs/pipeline_config.py'
        run_config = load_config(default_config_path)
    run_config = convert_to_config(run_config.PipelineConfig)

    if optim_config_path:
        optim_config = load_config(optim_config_path)
    else:
        default_optim_config_path = 'src/autotuneml/configs/optimization_config.py'
        optim_config = load_config(default_optim_config_path)
    optim_config = convert_to_config(optim_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    for model_name in run_config.models:
        try:
            is_fastai = model_name == 'fastai_tabular'
            if is_fastai:
                data = load_and_prepare_fastai_data(
                    data_path,
                    target,
                    run_config.problem_type,
                )
                model_results, model = train_fastai_with_optuna(data, run_config, optim_config)
            else:
                X_train, X_test, y_train, y_test = load_and_prepare_data(
                    data_path,
                    target,
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
                model_results, model = train_and_evaluate_best_params(
                    model_name,
                    best_hyperparams,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    run_config.problem_type,
                    optim_config,
                )

            save_results(model_results, timestamp)
            save_model(model, model_name, timestamp)
            results[model_name] = model_results

            logger.info(f"Best {model_name} model has been saved in the 'models' directory.")
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.error(f"Failed to optimize and train {model_name}: {str(e)} in {file_name}, line {line_number}")
            results[model_name] = {"error": str(e)}

    logger.info("Process completed")
    return results


def main(args):
    return run_autotuneml(args.data_path, args.target, args.run_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline with Hyperparameter Optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of the target variable column")
    parser.add_argument("--run_config_path", type=str, help="Path to the run configuration file")
    parser.add_argument("--optim_config_path", type=str, help="Path to the optimization configuration file")

    args = parser.parse_args()
    run_autotuneml(args.data_path, args.target, args.run_config_path, args.optim_config_path)
