import sys
from typing import Any, Dict

import numpy as np
from fastai.tabular.all import tabular_learner
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll.base import scope
from pyxtend import struct
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from log_config import logger


def train_model(params: Dict[str, Any], model_class, X_train, X_test, y_train, y_test, problem_type: str):
    """Sklearn and XGBoost training"""
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:

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


def convert_config_to_hyperopt_space(config_space):
    hyperopt_space = {}
    for param, value in config_space.items():
        if isinstance(value, list):
            if len(value) == 2:
                if param in ['max_depth', 'n_estimators']:
                    hyperopt_space[param] = scope.int(hp.quniform(param, value[0], value[1], 1))
                elif all(isinstance(v, int) for v in value):
                    hyperopt_space[param] = hp.quniform(param, value[0], value[1], 1)
                elif all(isinstance(v, float) for v in value):
                    hyperopt_space[param] = hp.uniform(param, value[0], value[1])
            elif len(value) > 2:
                hyperopt_space[param] = hp.choice(param, value)
        elif isinstance(value, (int, float)):
            hyperopt_space[param] = value
    return hyperopt_space


def get_model_class(model_name, problem_type):
    if model_name == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier if problem_type == 'classification' else XGBRegressor
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return RandomForestClassifier if problem_type == 'classification' else RandomForestRegressor
    elif model_name == 'linear':
        from sklearn.linear_model import LinearRegression, LogisticRegression

        return LogisticRegression if problem_type == 'classification' else LinearRegression
    # Add other model types as needed
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def run_hyperopt(
    model_name: str, X_train, y_train, X_test, y_test, problem_type: str, num_trials: int = 50, optim_config=None
):
    logger.info(f"Starting Hyperopt optimization for {model_name}")
    model_info = optim_config.model_spaces[model_name]
    trials = Trials()

    # Convert the config space to Hyperopt space
    hyperopt_space = convert_config_to_hyperopt_space(model_info.hyperopt_space)

    def objective(params):
        try:
            logger.info(f"Starting objective function with params: {params}")
            # model_info = optim_config.model_spaces[model_name]
            model_class = get_model_class(model_name, problem_type)
            result = train_model(params, model_class, X_train, X_test, y_train, y_test, problem_type)
            logger.info(f"Finished objective function. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return {'loss': np.inf, 'status': STATUS_FAIL}

    try:
        best = fmin(
            fn=objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            show_progressbar=False,
        )

        best_hyperparams = space_eval(hyperopt_space, best)
        logger.info(f"Best hyperparameters for {model_name}: {best_hyperparams}")

        logger.info(f"Completed {len(trials.trials)} trials for {model_name}")

        best_score = min([t['result']['loss'] for t in trials.trials if t['result']['status'] == STATUS_OK])
        logger.info(f"Best score achieved for {model_name}: {best_score}")

    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        logger.error(f"Error during hyperparameter optimization: {str(e)} in {file_name}, line {line_number}")
        return None

    return best_hyperparams


def train_and_evaluate_best_params(
    model_name: str, best_hyperparams: Dict[str, Any], X_train, y_train, X_test, y_test, problem_type: str, optim_config
):
    logger.info(f"Training final {model_name} model with best hyperparameters")
    model_info = optim_config.model_spaces[model_name]

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
        model_class = get_model_class(model_name, problem_type)
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
