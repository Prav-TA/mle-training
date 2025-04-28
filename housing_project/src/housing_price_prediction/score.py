import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.housing_price_prediction.constants import ARTIFACTS_DIR, LOGS_DIR
from src.housing_price_prediction.utils import get_project_root, setup_logging


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a given model on a test set.

    Parameters
    ----------
    model : object
        A scikit-learn model object.
    X_test : array-like, shape (n_samples, n_features)
        The test data.
    y_test : array-like, shape (n_samples,)
        The target values for the test data.

    Returns
    -------
    rmse : float
        The root mean squared error of the model on the test data.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/processed/test_data.csv",
        help="Path to test data CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=ARTIFACTS_DIR / "model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=LOGS_DIR / "score.log",
        help="Path to save log file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging Level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()
    project_root = get_project_root()
    log_path = os.path.join(project_root, args.log_path)
    test_path = os.path.join(project_root, args.test_path)
    model_path = os.path.join(project_root, args.model_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_path=log_path, log_level=args.log_level)
    logging.info(f"Testing a trained model...")

    df = pd.read_csv(test_path)
    X_test = df.drop("median_house_value", axis=1)
    y_test = df["median_house_value"]

    model = joblib.load(model_path)
    rmse = evaluate_model(model, X_test, y_test)

    logging.info(f"Model RMSE on test data: {rmse:.2f}")
    print(f"Model RMSE on test data: {rmse:.2f}")


if __name__ == "__main__":
    main()
