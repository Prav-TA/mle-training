import os.path as op
import sys

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import columns, data_frames, range_indexes, series
from pandas.testing import assert_frame_equal, assert_series_equal
from test_archive.tests.test_data import datasets
from test_archive.tsttemplate.analysis import calc_vif, string_cleaning

HERE = op.dirname(op.abspath(__file__))
test_path = op.join(HERE, "..", "..", "..", "..")
sys.path.append(test_path)


# test conditions with numerical values excluding inf and nans
@given(
    value=data_frames(
        index=range_indexes(min_size=5, max_size=10),
        columns=columns(["A", "B"], dtype=float, unique=True),
        rows=st.tuples(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1000000000000,
                max_value=1000000000000,
            ),
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                min_value=-1000000000000,
                max_value=1000000000000,
            ),
        ),
    )
)
@pytest.mark.parametrize("test_input,expected", datasets.calc_vif())
def test_calc_vif(value, test_input, expected):
    df = calc_vif(value)
    assert df["VIF"].dtype == "float64"
    assert isinstance(df, pd.DataFrame)
    assert_frame_equal(calc_vif(test_input), expected)


@given(value=series(elements=st.text()))
@pytest.mark.parametrize("test_input,expected", datasets.string_cleaning())
def test_string_cleaning(value, test_input, expected):
    df = string_cleaning(value)
    assert df.dtype == "object"
    assert isinstance(df, (pd.Series))
    assert_series_equal(string_cleaning(test_input), expected)
