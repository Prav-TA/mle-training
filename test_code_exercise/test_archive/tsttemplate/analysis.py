import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def string_cleaning(
    string_series,
    special_chars_to_keep="._,$&",
    remove_chars_in_braces=True,
    strip=True,
    lower=False,
):
    """Clean strings to remove any unwanted characters.

    Removes special character, characters between square/
    round braces, multiple spaces, leading/tailing spaces

    Parameters
    ----------
    string_series         : pandas string Series
    special_chars_to_keep : string having special characters that have to
                            be kept
    remove_chars_in_braces: Logical if to keep strings in braces.
                            e.g: "Packages (8oz)" will be "Packages"
    strip                 : True(default), if False it will not remove
                            extra/leading/tailing spaces
    lower                 : False(default), if True it will convert all
                            characters to lowercase

    Returns
    -------
     pandas series

    Raises
    ------
    AttributeError if variable passed is not string.
    KeyError if variable name is wrong.
    """
    # FIXME: Handle Key Error Runtime Exception
    try:
        if lower:
            # Convert names to lowercase
            string_series = string_series.str.lower()
        if remove_chars_in_braces:
            # Remove characters between square and round braces
            string_series = string_series.str.replace(r"\(.*\)|\[.*\]", "", regex=True)
        else:
            # Add braces to special character list, so that they will not be
            # removed further
            special_chars_to_keep = special_chars_to_keep + "()[]"
        if special_chars_to_keep:
            # Keep only alphanumeric character and some special
            # characters(.,_-&)
            reg_str = "[^\\w" + "\\".join(list(special_chars_to_keep)) + " ]"
            string_series = string_series.str.replace(reg_str, "")
        if strip:
            # Remove multiple spaces
            string_series = string_series.str.replace(r"\s+", " ", regex=True)
            # Remove leading and trailing spaces
            string_series = string_series.str.strip()
        return string_series
    except AttributeError:
        print("Variable datatype is not string")
    except KeyError:
        print("Variable name mismatch")


def calc_vif(X):
    """Calculate the Variance Inflation Factor (VIF) for the given data.

    VIF is a measure of multicollinearity between X variables. In general, a value of 10 or above indicated high degree of collinearity.

    Parameters
    ----------
    X: pd.Dataframe
        Input data frame with the features which VIF is to be computed.

    Returns
    -------
    pd.DataFrame
        Computed VIFs with each row for each feature.
    """
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif
