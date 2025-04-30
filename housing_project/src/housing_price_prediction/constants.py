import os
from pathlib import Path

from src.housing_price_prediction.utils import get_project_root

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Data directories
ARTIFACTS_DIR = Path("artifacts")
LOGS_DIR = Path("logs")
