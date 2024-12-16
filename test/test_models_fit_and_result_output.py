import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import sys 
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.models_fit_and_result_output import models_fit_and_result_output

@pytest.fixture
def sample_data():
    """Fixture to create sample data and preprocessing pipeline."""
    # Load Iris dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), X.columns)
        ],
        remainder='passthrough'
    )

    return preprocessor, X_train, X_test, y_train, y_test

def test_models_fit_and_result_output_files(sample_data, tmp_path):
    """Test if classification reports are saved correctly for each model."""
    preprocessor, X_train, X_test, y_train, y_test = sample_data

    # Temporary directory for output files
    output_dir = tmp_path / "results"
    os.makedirs(output_dir, exist_ok=True)

    # Call the function
    models_fit_and_result_output(preprocessor, X_train, y_train, X_test, y_test, str(output_dir))

    # Define expected file names based on model names
    expected_files = [
        "classification_report_Logistic_Regression.txt",
        "classification_report_Decision_Tree.txt",
        "classification_report_Support_Vector_Machine.txt",
        "classification_report_K-Nearest_Neighbors.txt"
    ]

    # Check if all expected files are created
    for file_name in expected_files:
        output_file = output_dir / file_name
        assert output_file.exists(), f"Output file {file_name} was not created."

        # Check content in the output file
        content = output_file.read_text()
        assert "precision" in content, f"{file_name} does not contain precision scores."
        assert "recall" in content, f"{file_name} does not contain recall scores."
        assert "f1-score" in content, f"{file_name} does not contain f1 scores."

def test_models_fit_and_result_output_print(sample_data, capsys, tmp_path):
    """Test if hyperparameter tuning and evaluation messages are printed."""
    preprocessor, X_train, X_test, y_train, y_test = sample_data

    # Temporary directory for output files
    output_dir = tmp_path / "results"
    os.makedirs(output_dir, exist_ok=True)

    # Call the function
    models_fit_and_result_output(preprocessor, X_train, y_train, X_test, y_test, str(output_dir))

    # Capture printed output
    captured = capsys.readouterr()

    # Check if model tuning and evaluation messages are printed
    assert "Tuning hyperparameters for Logistic Regression" in captured.out
    assert "Evaluating Logistic Regression on test set..." in captured.out
    assert "Tuning hyperparameters for Decision Tree" in captured.out
    assert "Evaluating Decision Tree on test set..." in captured.out
    assert "Tuning hyperparameters for Support Vector Machine" in captured.out
    assert "Evaluating Support Vector Machine on test set..." in captured.out
    assert "Tuning hyperparameters for K-Nearest Neighbors" in captured.out
    assert "Evaluating K-Nearest Neighbors on test set..." in captured.out