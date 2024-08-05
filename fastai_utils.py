import os
from datetime import datetime

import optuna
import pandas as pd
import yaml
from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    EarlyStoppingCallback,
    FillMissing,
    Normalize,
    RegressionBlock,
    TabularPandas,
    accuracy,
    cont_cat_split,
    rmse,
    tabular_config,
    tabular_learner,
)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from log_config import logger


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config()


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


def fastai_objective(trial, data, problem_type, config):
    fastai_config = config['model_spaces']['fastai_tabular']['hyperopt_space']

    n_layers = trial.suggest_int('n_layers', *fastai_config['n_layers'])
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'layer_{i}', *fastai_config['layer_size']))

    ps = trial.suggest_float('ps', *fastai_config['ps'])
    bs = trial.suggest_categorical('bs', fastai_config['bs'])

    lr_range = fastai_config['lr']
    lr_low = float(lr_range[0])
    lr_high = float(lr_range[1])
    lr = trial.suggest_float('lr', lr_low, lr_high, log=True)

    embed_p = trial.suggest_float('embed_p', *fastai_config['embed_p'])

    dls = data.dataloaders(bs=bs)

    config = tabular_config(ps=ps, embed_p=embed_p)
    metrics = [accuracy] if problem_type == 'classification' else [rmse]
    learn = tabular_learner(dls, layers=layers, config=config, metrics=metrics)

    try:
        learn.fit_one_cycle(10, lr, cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)])
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        return float('inf')  # Return a large value to indicate failure

    # Evaluate the model
    preds, targets = learn.get_preds(dl=dls.valid)
    if problem_type == 'classification':
        acc = accuracy_score(targets.numpy(), preds.argmax(dim=1).numpy())
        print(f"Trial accuracy: {acc}")
        return -acc  # Optuna minimizes the objective, so we return negative accuracy
    else:
        mse = mean_squared_error(targets.numpy(), preds.numpy())
        print(f"Trial MSE: {mse}")
        return mse


def train_fastai_with_optuna(data, problem_type, config, n_trials=50):
    logger.info("Starting FastAI training with Optuna")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: fastai_objective(trial, data, problem_type, config), n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Train the final model with the best parameters
    dls = data.dataloaders(bs=best_params['bs'])
    layers = [best_params[f'layer_{i}'] for i in range(best_params['n_layers'])]
    config = tabular_config(ps=best_params['ps'], embed_p=best_params['embed_p'])
    metrics = [accuracy] if problem_type == 'classification' else [rmse]
    learn = tabular_learner(dls, layers=layers, config=config, metrics=metrics)
    learn.fit_one_cycle(
        10, best_params['lr'], cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)]
    )

    # Evaluate the final model
    preds, targets = learn.get_preds(dl=dls.valid)
    if problem_type == 'classification':
        accuracy = accuracy_score(targets.numpy(), preds.argmax(dim=1).numpy())
        f1 = f1_score(targets.numpy(), preds.argmax(dim=1).numpy(), average='weighted')
        logger.info(f"Final model performance - Accuracy: {accuracy}, F1 Score: {f1}")
        results = {'model': 'fastai_tabular', 'accuracy': accuracy, 'f1': f1, **best_params}
    else:
        mse = mean_squared_error(targets.numpy(), preds.numpy())
        r2 = r2_score(targets.numpy(), preds.numpy())
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
