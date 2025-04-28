import pandas as pd
import pytest

from src.housing_price_prediction.ingest_data import load_data


def test_load_data(tmp_path):
    # Create a dummy CSV
    dummy_csv = tmp_path / "housing.csv"
    dummy_df = pd.DataFrame(
        {
            "longitude": {0: -122.23, 1: -122.22},
            "latitude": {0: 37.88, 1: 37.86},
            "housing_median_age": {0: 41.0, 1: 21.0},
            "total_rooms": {0: 880.0, 1: 7099.0},
            "total_bedrooms": {0: 129.0, 1: 1106.0},
            "population": {0: 322.0, 1: 2401.0},
            "households": {0: 126.0, 1: 1138.0},
            "median_income": {0: 8.3252, 1: 8.3014},
            "median_house_value": {0: 452600.0, 1: 358500.0},
            "ocean_proximity": {0: "NEAR BAY", 1: "NEAR BAY"},
        }
    )
    dummy_df.to_csv(dummy_csv, index=False)

    loaded_df = load_data(str(dummy_csv))

    # Validate the loaded DataFrame
    assert loaded_df.shape == dummy_df.shape
    assert all(loaded_df.columns == dummy_df.columns)
    pd.testing.assert_frame_equal(loaded_df, dummy_df)
