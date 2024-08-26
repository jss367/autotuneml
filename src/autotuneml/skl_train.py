import sys
from typing import Any, Dict

import numpy as np
from fastai.tabular.all import tabular_learner
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from hyperopt.pyll.base import scope
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from autotuneml.configs.config_class import Config
from autotuneml.log_config import logger

MODEL_CONFIG_MAP = {
    "xgboost": "XGBoostConfig",
    "random_forest": "RandomForestConfig",
    "linear": "LinearConfig",
    "fastai_tabular": "FastaiTabularConfig",
}


def train_model(params: Dict[str, Any], model_class, X_train, X_test, y_train, y_test, problem_type: str):
    """Sklearn and XGBoost training"""
    logger.info(f"Training {model_class.__name__} with params: {params}")
    try:
        # Ensure integer parameters for RandomForestClassifier
        if model_class.__name__ == 'RandomForestClassifier':
            for param in ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
                if param in params:
                    params[param] = int(params[param])

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
        logger.debug(f"Converting parameter: {param} with value: {value}")
        if isinstance(value, list):
            if len(value) == 2:
                if all(isinstance(v, bool) for v in value):
                    hyperopt_space[param] = hp.choice(param, value)
                elif all(isinstance(v, (int, float)) for v in value):
                    low, high = value
                    if low > high:
                        logger.warning(f"Invalid range for {param}: [{low}, {high}]. Swapping values.")
                        low, high = high, low
                    if all(isinstance(v, int) for v in value):
                        hyperopt_space[param] = scope.int(hp.quniform(param, low, high, 1))
                    elif param in ['lr', 'weight_decay']:
                        hyperopt_space[param] = hp.loguniform(param, np.log(low), np.log(high))
                    else:
                        hyperopt_space[param] = hp.uniform(param, low, high)
                else:
                    hyperopt_space[param] = hp.choice(param, value)
            else:
                hyperopt_space[param] = hp.choice(param, value)
        elif isinstance(value, (int, float, bool)):
            hyperopt_space[param] = value
        else:
            hyperopt_space[param] = hp.choice(param, [value])

        logger.debug(f"Converted {param} to: {hyperopt_space[param]}")

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
    logger.info("Starting Hyperopt optimization for %s", model_name)
    # model_info = optim_config.model_spaces[model_name]
    trials = Trials()

    model_config_name = MODEL_CONFIG_MAP.get(model_name.lower())
    if model_config_name is None:
        raise ValueError(f"Configuration for model '{model_name}' not found in MODEL_CONFIG_MAP")

    try:
        model_config = getattr(optim_config, model_config_name)
        if not isinstance(model_config, Config):
            raise AttributeError
    except AttributeError:
        raise ValueError(f"Configuration '{model_config_name}' not found in optim_config")

    # Convert the config space to Hyperopt space
    hyperopt_space = convert_config_to_hyperopt_space(model_config.hyperopt_space)

    def objective(params):
        try:
            logger.info("Starting objective function with params: %s", params)
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
        # Create default hyperparameters based on the hyperopt_space
        best_hyperparams = {}
        for key, value in hyperopt_space.items():
            if isinstance(value, (int, float)):
                best_hyperparams[key] = value
            elif hasattr(value, 'pos_args'):  # This checks if it's a hyperopt distribution
                if 'quniform' in str(value):
                    best_hyperparams[key] = value.pos_args[1]  # Use the lower bound
                elif 'choice' in str(value):
                    best_hyperparams[key] = value.pos_args[0][0]  # Use the first choice
                else:
                    best_hyperparams[key] = value.pos_args[0]  # Use the first argument as default
            else:
                best_hyperparams[key] = value  # For any other case, just use the value as is
        return None

    return best_hyperparams


def train_and_evaluate_best_params(
    model_name: str, best_hyperparams: Dict[str, Any], X_train, y_train, X_test, y_test, problem_type: str, optim_config
):
    logger.info(f"Training final {model_name} model with best hyperparameters")
    # model_info = optim_config.model_spaces[model_name]

    # Filter out hyperparameters that are not applicable to the specific model
    model_class = get_model_class(model_name, problem_type)
    valid_params = model_class().get_params().keys()
    filtered_hyperparams = {k: v for k, v in best_hyperparams.items() if k in valid_params}

    logger.info(f"Filtered hyperparameters for {model_name}: {filtered_hyperparams}")

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
