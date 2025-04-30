from unittest import mock

import pandas as pd
import pytest

from src.housing_price_prediction import train


@pytest.fixture
def temp_output_path(tmp_path):
    """Fixture to create a temporary output directory."""
    return tmp_path / "data"


# Define the dummy_train_data fixture
@pytest.fixture
def dummy_train_data():
    return pd.DataFrame(
        {
            "longitude": [1.0, 2.0],
            "latitude": [3.0, 4.0],
            "housing_median_age": [5, 6],
            "total_rooms": [7, 8],
            "total_bedrooms": [9, 10],
            "population": [11, 12],
            "households": [13, 14],
            "median_income": [15, 16],
            "median_house_value": [17, 18],
        }
    )


# Test function with the mock patches and fixture
@mock.patch("src.housing_price_prediction.train.pd.read_csv")
@mock.patch("src.housing_price_prediction.train.train_linear")
@mock.patch("src.housing_price_prediction.train.train_tree")
@mock.patch("src.housing_price_prediction.train.train_forest_with_grid_search")
@mock.patch("joblib.dump")  # Mock joblib.dump to prevent the pickle error
def test_train_model(
    mock_joblib_dump,
    mock_train_forest,
    mock_train_tree,
    mock_train_linear,
    mock_read_csv,
    dummy_train_data,
    temp_output_path,
    monkeypatch,
):
    # Mock the read_csv method
    mock_read_csv.return_value = dummy_train_data

    # --- Setup: fake argparse arguments ---
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_path",
            str(dummy_train_data),
            "--model_path",
            str(temp_output_path / "model.joblib"),
            "--model_type",
            "forest",
            "--log_level",
            "DEBUG",
        ],
    )

    # --- Run ---
    train.main()

    # --- Assert the correct model was called ---
    mock_train_forest.assert_called_once()
    mock_train_tree.assert_not_called()
    mock_train_linear.assert_not_called()

    # Ensure joblib.dump was called to save the model
    mock_joblib_dump.assert_called_once()
