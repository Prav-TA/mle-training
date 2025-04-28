import numpy as np
import pandas as pd
import pytest

from src.housing_price_prediction.ingest_data import preprocess_df


def test_preprocess_df():
    # Dummy DataFrame
    data = pd.DataFrame(
        {
            "longitude": {
                0: -122.23,
                1: -122.22,
                2: -122.24,
                3: -122.25,
                4: -122.25,
                5: -122.25,
                6: -122.25,
                7: -122.25,
                8: -122.26,
                9: -122.25,
            },
            "latitude": {
                0: 37.88,
                1: 37.86,
                2: 37.85,
                3: 37.85,
                4: 37.85,
                5: 37.85,
                6: 37.84,
                7: 37.84,
                8: 37.84,
                9: 37.84,
            },
            "housing_median_age": {
                0: 41.0,
                1: 21.0,
                2: 52.0,
                3: 52.0,
                4: 52.0,
                5: 52.0,
                6: 52.0,
                7: 52.0,
                8: 42.0,
                9: 52.0,
            },
            "total_rooms": {
                0: 880.0,
                1: 7099.0,
                2: 1467.0,
                3: 1274.0,
                4: 1627.0,
                5: 919.0,
                6: 2535.0,
                7: 3104.0,
                8: 2555.0,
                9: 3549.0,
            },
            "total_bedrooms": {
                0: 129.0,
                1: 1106.0,
                2: 190.0,
                3: 235.0,
                4: 280.0,
                5: 213.0,
                6: 489.0,
                7: 687.0,
                8: 665.0,
                9: 707.0,
            },
            "population": {
                0: 322.0,
                1: 2401.0,
                2: 496.0,
                3: 558.0,
                4: 565.0,
                5: 413.0,
                6: 1094.0,
                7: 1157.0,
                8: 1206.0,
                9: 1551.0,
            },
            "households": {
                0: 126.0,
                1: 1138.0,
                2: 177.0,
                3: 219.0,
                4: 259.0,
                5: 193.0,
                6: 514.0,
                7: 647.0,
                8: 595.0,
                9: 714.0,
            },
            "median_income": {
                0: 8.3252,
                1: 8.3014,
                2: 7.2574,
                3: 5.6431,
                4: 3.8462,
                5: 4.0368,
                6: 3.6591,
                7: 3.12,
                8: 2.0804,
                9: 3.6912,
            },
            "median_house_value": {
                0: 452600.0,
                1: 358500.0,
                2: 352100.0,
                3: 341300.0,
                4: 342200.0,
                5: 269700.0,
                6: 299200.0,
                7: 241400.0,
                8: 226700.0,
                9: 261100.0,
            },
            "ocean_proximity": {
                0: "NEAR BAY",
                1: "NEAR BAY",
                2: "NEAR BAY",
                3: "NEAR BAY",
                4: "NEAR BAY",
                5: "NEAR BAY",
                6: "NEAR BAY",
                7: "NEAR BAY",
                8: "NEAR BAY",
                9: "NEAR BAY",
            },
        }
    )

    processed = preprocess_df(data)

    # Check if 'ocean_proximity' is dropped and no NaN values are present
    assert (
        "ocean_proximity" not in processed.columns and processed.isna().sum().sum() == 0
    )
