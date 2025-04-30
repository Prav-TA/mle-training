import os
import shutil
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.housing_price_prediction import ingest_data


@pytest.fixture
def temp_output_path(tmp_path):
    """Fixture to create a temporary output directory."""
    return tmp_path / "data"


@mock.patch("src.housing_price_prediction.ingest_data.fetch_data")
@mock.patch("src.housing_price_prediction.ingest_data.load_data")
def test_ingest_data_main(
    mock_load_data, mock_fetch_data, temp_output_path, monkeypatch
):

    sample_data = pd.DataFrame(
        {
            "longitude": np.random.uniform(-124.0, -113.0, 100),
            "latitude": np.random.uniform(32.0, 42.0, 100),
            "housing_median_age": np.random.randint(1, 52, 100),
            "total_rooms": np.random.randint(1, 10000, 100),
            "total_bedrooms": np.random.randint(1, 5000, 100),
            "population": np.random.randint(1, 5000, 100),
            "households": np.random.randint(1, 5000, 100),
            "median_income": np.random.randint(1, 15, 100),
            "median_house_value": np.random.randint(15000, 500000, 100),
            "ocean_proximity": np.random.choice(
                ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND"], 100
            ),
        }
    )
    mock_load_data.return_value = sample_data

    # --- Setup: fake argparse arguments ---
    monkeypatch.setattr(
        "sys.argv",
        [
            "ingest_data.py",
            "--output_path",
            str(temp_output_path),
            "--log_level",
            "DEBUG",
        ],
    )

    # --- Run ---
    ingest_data.main()

    # --- Assertions ---

    # Check that fetch_data was called
    raw_data_path = os.path.join(str(temp_output_path), "raw")
    mock_fetch_data.assert_called_once_with(raw_data_path)

    # Check output processed folder exists
    processed_dir = os.path.join(str(temp_output_path), "processed")
    assert os.path.isdir(processed_dir)

    # Check output files exist
    assert os.path.isfile(os.path.join(processed_dir, "housing_preprocessed.csv"))
    assert os.path.isfile(os.path.join(processed_dir, "train_data.csv"))
    assert os.path.isfile(os.path.join(processed_dir, "test_data.csv"))

    df_preprocessed = pd.read_csv(
        os.path.join(processed_dir, "housing_preprocessed.csv")
    )
    assert (
        "rooms_per_household" in df_preprocessed.columns
        and "population_per_household" in df_preprocessed.columns
        and "income_cat" in df_preprocessed.columns
    )
