from typing import Union

import pandas as pd
from fastai.tabular.all import TabularPandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from autotuneml.fastai_train import prepare_fastai_data
from autotuneml.log_config import logger


def load_data(path):
    logger.info(f"Loading data from {path}")

    # First, read the CSV without parsing dates
    df = pd.read_csv(path)

    # Check if 'Date' column exists
    if 'Date' in df.columns:
        # If it exists, parse it as date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        # If it doesn't exist, throw a warning
        logger.warning("No 'Date' column in CSV file.")

    logger.info(f"Raw data shape: {df.shape}")
    return df


def load_and_split_data(
    path: str, split_method: str, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = load_data(path)

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
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], tuple[TabularPandas, TabularPandas, pd.Series, pd.Series]
]:
    train_df, test_df = load_and_split_data(path, split_method, test_size, random_state)

    verify_dataset(train_df, target)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    if is_fastai:
        return prepare_fastai_data(combined_df, target, problem_type)

    if "Date" in train_df.columns:
        train_df = train_df.drop(columns="Date")
        test_df = test_df.drop(columns="Date")

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]

    y_train, y_test = encode(problem_type, y_train, y_test)
    return X_train, X_test, y_train, y_test


def verify_dataset(df, target):
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")
