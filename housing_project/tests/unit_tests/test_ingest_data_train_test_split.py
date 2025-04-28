import test

import numpy as np
import pandas as pd
import pytest

from src.housing_price_prediction.ingest_data import train_test_split


def test_train_test_split():
    """
    Test the train_test_split function to ensure it splits the data correctly.
    """
    sample_data = pd.DataFrame(
        {
            "median_income": np.random.randint(1, 15, 100),
            "median_house_value": np.random.randint(15000, 500000, 100),
            "ocean_proximity": np.random.choice(
                ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND"], 100
            ),
            "income_cat": np.random.randint(1, 5, 100),
            "housing_median_age": np.random.randint(1, 52, 100),
            "total_rooms": np.random.randint(1, 10000, 100),
            "total_bedrooms": np.random.randint(1, 5000, 100),
            "population": np.random.randint(1, 5000, 100),
            "households": np.random.randint(1, 5000, 100),
        }
    )

    train_set, test_set = train_test_split(sample_data)

    # Check if the split is correct
    assert train_set.shape[1] == test_set.shape[1]
