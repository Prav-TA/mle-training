import argparse
import logging
import os
import test

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from src.housing_price_prediction.constants import ARTIFACTS_DIR, LOGS_DIR
from src.housing_price_prediction.utils import get_project_root, setup_logging


def train_linear(X, y):
    """
    Train a linear regression model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The features to train the model on.
    y : array-like, shape (n_samples,)
        The target values to train the model against.

    Returns
    -------
    model : LinearRegression
        The trained model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_tree(X, y):
    """
    Train a decision tree regressor model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The features to train the model on.
    y : array-like, shape (n_samples,)
        The target values to train the model against.

    Returns
    -------
    model : DecisionTreeRegressor
        The trained decision tree model.
    """
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model


def train_forest_with_grid_search(X, y):
    """
    Train a random forest regressor model with hyperparameter tuning.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The features to train the model on.
    y : array-like, shape (n_samples,)
        The target values to train the model against.

    Returns
    -------
    model : RandomForestRegressor
        The trained model with best hyperparameters.
    """
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def main():
    parser = argparse.ArgumentParser(description="Train a model on housing data.")
    parser.add_argument(
        "--train_path",
        default="data/processed/train_data.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--test_path",
        default="data/processed/test_data.csv",
        help="Path to test data CSV",
    )

    parser.add_argument(
        "--model_path",
        default=ARTIFACTS_DIR / "model.joblib",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--model_type",
        choices=["linear", "tree", "forest"],
        default="forest",
        help="Model to train",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=LOGS_DIR / "train.log",
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
    log_path = os.path.join(project_root, args.log_path)
    train_path = os.path.join(project_root, args.train_path)
    model_path = os.path.join(project_root, args.model_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_path=log_path, log_level=args.log_level)
    logging.info(f"Training a {args.model_type} model...")

    train_df = pd.read_csv(train_path)

    X_train = train_df.drop("median_house_value", axis=1)
    y_train = train_df["median_house_value"].copy()

    if args.model_type == "linear":
        model = train_linear(X_train, y_train)
    elif args.model_type == "tree":
        model = train_tree(X_train, y_train)
    elif args.model_type == "forest":
        model = train_forest_with_grid_search(X_train, y_train)

    logging.info(f"Model training completed successfully")
    logging.info(f"R2 score on training data: {model.score(X_train, y_train)}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
