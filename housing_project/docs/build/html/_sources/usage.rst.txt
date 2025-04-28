Usage Guide - **Housing Price Prediction**
===========================================

-   This section provides a brief overview of how to use the code in this repository.

-   It is preferrable to create the conda environment using the env.yml file provided in the repository. This will ensure that all dependencies are installed correctly.


Steps to Run This Project
-------------------------
-   Open the project folder using : **`cd housing_project`**

-   Then, run the following command to *create the conda environment*:

        **`conda env create -f deploy/conda/env.yml`**

    This will create a conda environment named **`mle-dev`** with all the required dependencies.

-   To *activate the environment*, run the following command:

        **`conda activate mle-dev`**

-   To *install the src package* run the following command:

        **`pip install -e .`**


-   To run Ingest Data:

        **`python src/housing_price_prediction/ingest_data.py --output_path data`**

-   To Train the model:

        **`python src/housing_price_prediction/train.py --train_path data/processed/train_data.csv --model_path artifacts/model.joblib`**

-   To evaluate the trained model

        **`python src/housing_price_prediction/score.py --test_path data/processed/test_data.csv --model_path artifacts/model.joblib`**

-   To verify installations

        **`pytest tests/installation_tests/test_imports.py`**

-   To run all the unit and functional test cases

        **`pytest tests/`**


-   Additional helps:
Following commands help *cleaning unwanted cache files*

        **`find . -type d -name "__pycache__" -exec rm -rf {} \+`**

        **`find . -type d -name "*.egg-info" -exec rm -rf {} \+`**

        **`find . -type d -name "*.pytest_cache" -exec rm -rf {} \+`**