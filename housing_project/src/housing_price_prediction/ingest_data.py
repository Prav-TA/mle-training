import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from src.housing_price_prediction.constants import HOUSING_URL, LOGS_DIR
from src.housing_price_prediction.utils import get_project_root, setup_logging


def fetch_data(output_path: str):
    """
    Download the housing dataset from the given URL and extract it to the specified output path.

    Parameters
    ----------
    output_path : str
        The path where the downloaded and extracted data will be saved.
    Returns
    -------
    None
    """

    os.makedirs(output_path, exist_ok=True)
    tgz_path = os.path.join(output_path, "housing.tgz")

    logging.info(f"Downloading data from {HOUSING_URL} to {tgz_path}")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    logging.info(f"Data downloaded to {tgz_path}")

    logging.info(f"Extracting data to {output_path}")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=output_path)
    logging.info(f"Data extracted to {output_path}")


def load_data(data_path: str):
    """
    Load housing data from the given path.

    Parameters
    ----------
    data_path : str
        The path to the housing dataset (CSV file).
    Returns
    -------
    DataFrame
        The loaded housing dataset as a pandas DataFrame.
    """

    logging.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def preprocess_df(df):
    """
    Preprocess the input DataFrame by dropping the ocean_proximity feature and imputing missing values using median imputation.

    Parameters:
    df (DataFrame): The input DataFrame containing housing data.

    Returns:
    DataFrame: The preprocessed DataFrame with ocean_proximity dropped and median imputation applied.
    """
    proximity_feature = df[["ocean_proximity"]]
    df = df.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    preprocessed_df = df_imputed.join(pd.get_dummies(proximity_feature, drop_first=True))
    return preprocessed_df


def feature_engineering(df):
    """
    Add derived features to the housing dataset.

    Parameters:
    df (DataFrame): The housing dataset containing the original features.

    Returns:
    DataFrame: The dataset with the additional features, which are:
        income_cat: median income categorized into 5 bins
        rooms_per_household: total rooms divided by households
        bedrooms_per_room: total bedrooms divided by total rooms
        population_per_household: population divided by households
    """
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def train_test_split(df):
    """
    Split the data into training and test sets.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame to be split.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrame, DataFrame, Series, Series
        The training and test data for features and target.
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    return train_set, test_set


def main():

    parser = argparse.ArgumentParser(description="Ingest housing data")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data",
        help="Path to save the downloaded data",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=LOGS_DIR / "ingest_data.log",
        help="Path to save the log file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging Level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()
    project_root = get_project_root()

    output_path = os.path.join(project_root, args.output_path)
    log_path = os.path.join(project_root, args.log_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_path=log_path, log_level=args.log_level)

    logging.info("Data ingestion module started.")
    raw_data_path = os.path.join(output_path, "raw")
    os.makedirs(raw_data_path, exist_ok=True)
    fetch_data(raw_data_path)
    logging.info("Data fetched successfully.")
    df = load_data(raw_data_path + "/housing.csv")
    df = preprocess_df(df)
    df = feature_engineering(df)

    processed_data_path = os.path.join(output_path, "processed")
    os.makedirs(processed_data_path, exist_ok=True)

    df.to_csv(processed_data_path + "/housing_preprocessed.csv", index=False)
    logging.info("Data preprocessed successfully.")
    logging.info(f"Preprocessed data saved to {processed_data_path}")

    # Split the data into training and test sets

    train_set, test_set = train_test_split(df)
    train_set.to_csv(processed_data_path + "/train_data.csv", index=False)
    test_set.to_csv(processed_data_path + "/test_data.csv", index=False)
    logging.info("Data split into training and test sets successfully.")
    logging.info(f"Train set saved to {processed_data_path}")
    logging.info(f"Test set saved to {processed_data_path}")


if __name__ == "__main__":
    main()
