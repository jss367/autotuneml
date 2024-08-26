from typing import Tuple, Union

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

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    if is_fastai:
        return prepare_fastai_data(combined_df, target, problem_type)

    if "Date" in train_df.columns:
        X_train, X_test, y_train, y_test = extract_date_info(train_df, test_df, target)
    else:
        X_train = train_df.drop(columns=[target])
        X_test = test_df.drop(columns=[target])
        y_train = train_df[target]
        y_test = test_df[target]

    y_train, y_test = encode(problem_type, y_train, y_test)
    return X_train, X_test, y_train, y_test


def verify_dataset(df, target):
    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' not found in the dataset.")
