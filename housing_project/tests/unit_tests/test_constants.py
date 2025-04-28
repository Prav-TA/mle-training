from src.housing_price_prediction.constants import HOUSING_URL


def test_constants():
    """
    Tests that the constants in the constants module are correctly set.

    This includes HOUSING_PATH and HOUSING_URL.
    """
    assert (
        HOUSING_URL
        == "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    )
