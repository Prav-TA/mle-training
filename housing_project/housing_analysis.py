"""Download and process California housing data for analysis."""

import os
import tarfile
from urllib.parse import urlparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from six.moves import urllib

matplotlib.use("TkAgg")

# Define the path to the dataset
# and the URL to download it from
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join(".", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    """
    Safely extract tar files by checking for path traversal.

    Parameters
    ----------
    tar : tarfile.TarFile
        The tar file object to extract.
    path : str, optional
        Directory to extract files to.
    members : list, optional
        Optional list of members to extract.
    numeric_owner : bool, optional
        Preserve numeric owner if set.
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_dir = os.path.abspath(path)
        abs_target = os.path.abspath(member_path)
        if not abs_target.startswith(abs_dir):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(  # noqa: S202
        path=path,
        members=members,
        numeric_owner=numeric_owner,
    )


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download the housing data from the given URL into the given path.

    Parameters
    ----------
    housing_url : str
        The URL of the housing data.
    housing_path : str
        The directory where the housing data is to be downloaded.
    """
    passed_url = urlparse(housing_url)
    if passed_url.scheme not in ["http", "https"]:
        raise ValueError("Invalid scheme. Only http and https are supported.")

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)  # noqa: S310

    with tarfile.open(tgz_path) as housing_tgz:
        safe_extract(housing_tgz, housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load the housing data from a CSV file located at the given path.

    Parameters
    ----------
    housing_path : str
        The path where the housing data CSV file is located.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
housing = load_housing_data(housing_path=HOUSING_PATH)


housing.plot(kind="scatter", x="longitude", y="latitude")
plt.show()


corr_matrix = housing.drop("ocean_proximity", axis=1).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# fmt: off
housing["rooms_per_household"] = (
    housing["total_rooms"] / housing["households"]
)
housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["population_per_household"] = (
    housing["population"] / housing["households"]
)
# fmt: on

# Drop the ocean_proximity column
housing_num = housing.drop("ocean_proximity", axis=1)
