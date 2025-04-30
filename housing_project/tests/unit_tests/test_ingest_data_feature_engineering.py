import pandas as pd
import pytest

from src.housing_price_prediction.ingest_data import feature_engineering


def test_feature_engineering():
    sample_data = pd.DataFrame(
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
            "ocean_proximity_INLAND": {0: False, 1: False},
            "ocean_proximity_ISLAND": {0: False, 1: False},
            "ocean_proximity_NEAR BAY": {0: True, 1: True},
            "ocean_proximity_NEAR OCEAN": {0: False, 1: False},
        }
    )

    feature_engineered_df = feature_engineering(sample_data)
    assert (
        "income_cat" in feature_engineered_df.columns
        and "rooms_per_household" in feature_engineered_df.columns
        and "bedrooms_per_room" in feature_engineered_df.columns
        and "population_per_household" in feature_engineered_df.columns
    )
