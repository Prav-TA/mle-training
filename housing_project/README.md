# Housing Price Prediction

This project is a production-ready pipeline for training and evaluating a machine learning model to predict housing prices in California.


## How to Run This Project

First, open the project folder using `cd housing_project`

### Step 1: Create the conda environment
`conda env create -f deploy/conda/env.yml`

This will create the environment named `mle-dev`

### Step 2: Activate the environment
`conda activate mle-dev`

### Step 3: Install the package in editable mode
`pip install -e .` to have `src` installed in the env from the code folder

## Usage:

### Ingest Data
`python src/housing_price_prediction/ingest_data.py --output_path data`

### Train the model
`python src/housing_price_prediction/train.py --train_path data/processed/train_data.csv --model_path artifacts/model.joblib`

### Evaluate the model
`python src/housing_price_prediction/score.py --test_path data/processed/test_data.csv --model_path artifacts/model.joblib`

### To verify installations
`pytest tests/installation_tests/test_imports.py`

### To run all the unit and functional test cases
`pytest tests/`


## Additional helps:
Following commands help cleaning unwanted cache files
`find . -type d -name "__pycache__" -exec rm -rf {} \+`
`find . -type d -name "*.egg-info" -exec rm -rf {} \+`
`find . -type d -name "*.pytest_cache" -exec rm -rf {} \+`