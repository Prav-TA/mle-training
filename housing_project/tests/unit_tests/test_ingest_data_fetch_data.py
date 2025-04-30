import os
import tarfile
from unittest import mock

import pytest

from src.housing_price_prediction.constants import HOUSING_URL
from src.housing_price_prediction.ingest_data import fetch_data


@mock.patch("src.housing_price_prediction.ingest_data.urllib.request.urlretrieve")
@mock.patch("src.housing_price_prediction.ingest_data.tarfile.open")
@mock.patch("src.housing_price_prediction.ingest_data.os.makedirs")
def test_fetch_data(mock_makedirs, mock_tarfile_open, mock_urlretrieve, tmp_path):
    output_path = tmp_path / "data"
    fetch_data(str(output_path))

    # Check output directory created
    mock_makedirs.assert_called_with(str(output_path), exist_ok=True)

    # Check URL download called
    tgz_path = os.path.join(output_path, "housing.tgz")
    mock_urlretrieve.assert_called_once_with(HOUSING_URL, tgz_path)

    # Check tarfile extraction
    mock_tarfile_open.assert_called_once_with(tgz_path)
    mock_tar = mock_tarfile_open.return_value.__enter__.return_value
    mock_tar.extractall.assert_called_once_with(path=str(output_path))
