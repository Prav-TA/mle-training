from unittest import mock

import numpy as np
import pandas as pd
import pytest
from src.housing_price_prediction import score


@pytest.fixture
def dummy_test_data():
    """Fixture that returns a dummy test DataFrame."""
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
            "median_house_value": [17, 18],  # This is the target column
        }
    )


@pytest.fixture
def dummy_model():
    """Fixture that returns a dummy model with a predict method."""
    model = mock.MagicMock()
    model.predict.return_value = np.array([17, 18])
    return model


def test_score_main(monkeypatch, dummy_test_data, dummy_model, tmp_path):
    # Patch pd.read_csv to return dummy_test_data
    monkeypatch.setattr(score.pd, "read_csv", lambda path: dummy_test_data)

    # Patch joblib.load to return dummy_model
    monkeypatch.setattr(score.joblib, "load", lambda path: dummy_model)

    # Patch sys.argv to simulate command line arguments
    monkeypatch.setattr(
        "sys.argv",
        [
            "score.py",
            "--test_path",
            str(tmp_path / "dummy_test.csv"),
            "--model_path",
            str(tmp_path / "dummy_model.joblib"),
            "--log_level",
            "DEBUG",
        ],
    )

    # Patch logging
    monkeypatch.setattr(score, "setup_logging", lambda log_path, log_level: None)

    # Run main
    score.main()
