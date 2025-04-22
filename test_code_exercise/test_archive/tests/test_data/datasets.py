import pandas as pd


def calc_vif():
    input_df = pd.DataFrame(
        {
            "X1": [50, 30, 80, 100, 10, 43, 69, 55, 10],
            "X2": [54, 36, 89, 100, 454, 999, -65, -55, 0],
            "X3": [-50, -30, -80, -100, -10, -43, -69, -55, -99],
        }
    )
    input_df1 = pd.DataFrame(
        {
            "X1": [501, 301, 800, -10, 30, 4334, 69, 5509, 1071],
            "X2": [1543, 3634, 8209, 12100, 4678, 0, -615, -550, 0],
            "X3": [-50, -30, -80, -100, -10, -43, -69, -55, -99],
        }
    )

    output_df = pd.DataFrame(
        {
            "variables": {0: "X1", 1: "X2", 2: "X3"},
            "VIF": {0: 5.123503325771861, 1: 1.113287024667442, 2: 4.984758542752185},
        }
    )
    output_df1 = pd.DataFrame(
        {
            "variables": {0: "X1", 1: "X2", 2: "X3"},
            "VIF": {0: 1.5742808397655255, 1: 2.040318515183488, 2: 2.6830759375524957},
        }
    )
    return [(input_df, output_df), (input_df1, output_df1)]


def string_cleaning():
    input_series = pd.Series(
        [
            "abc_txt",
            "abc$txt",
            "Abc_txt",
            "ABD_TXT",
            "A[ABC_TXT]",
            "ABC(TXT)",
            "abc  ",
            " abc ",
        ]
    )
    input_series1 = pd.Series(["123", " 123 ", "1234 ", "123(456)"])

    output_series = pd.Series(
        ["abc_txt", "abc$txt", "Abc_txt", "ABD_TXT", "A", "ABC", "abc", "abc"]
    )
    output_series1 = pd.Series(["123", "123", "1234", "123"])
    return [(input_series, output_series), (input_series1, output_series1)]
