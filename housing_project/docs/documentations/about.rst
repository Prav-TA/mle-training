About the Project
=================

This project is a production-ready machine learning pipeline for predicting housing prices in California.

It includes the following main components:

- **Data ingestion**: Download and split the dataset into training and test sets.
- **Model training**: Train a regression model on the processed training data.
- **Model evaluation**: Score the model using test data and compute RMSE.
- **Logging**: Each script generates a log file for better traceability.
- **Testing**: Includes installation tests, unit tests, and functional tests.
- **Packaging**: Project is installable as a library (`pip install -e .`).
- **Environment Management**: Uses a Conda environment YAML file.
- **Documentation**: Auto-generated using Sphinx with NumPy-style docstrings.
