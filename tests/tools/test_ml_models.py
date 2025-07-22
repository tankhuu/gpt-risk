import pytest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np

# Import the tools to be tested
from app.tools.ml_models import run_fraud_detection_model, run_credit_risk_model

# Sample data for tool inputs
SAMPLE_TRANSACTION = {"transaction_amount": 100.0}
SAMPLE_LOAN_APP = {"loan_amount": 20000}


@patch("joblib.load")
@patch("os.path.exists", return_value=True)
def test_run_fraud_detection_model_success(mock_exists, mock_joblib_load):
    """
    Unit test for the fraud detection tool on a successful run.
    Mocks the model loading to avoid file system dependency.
    """
    # Create a mock model object
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.1, 0.9]]
    )  # [prob_class_0, prob_class_1]
    mock_joblib_load.return_value = mock_model

    # Call the tool
    result_str = run_fraud_detection_model.invoke(SAMPLE_TRANSACTION)
    result = json.loads(result_str)

    # Assertions
    mock_exists.assert_called_once()
    mock_joblib_load.assert_called_once()
    mock_model.predict_proba.assert_called_once()
    assert "fraud_probability" in result
    assert result["fraud_probability"] == 0.9


@patch("os.path.exists", return_value=False)
def test_run_fraud_detection_model_not_found(mock_exists):
    """
    Unit test for the fraud detection tool when the model file is not found.
    """
    result_str = run_fraud_detection_model.invoke(SAMPLE_TRANSACTION)
    result = json.loads(result_str)

    mock_exists.assert_called_once()
    assert "error" in result
    assert "not found" in result["error"]


@patch("joblib.load")
@patch("os.path.exists", return_value=True)
def test_run_credit_risk_model_success(mock_exists, mock_joblib_load):
    """
    Unit test for the credit risk tool on a successful run.
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.88, 0.12]])
    mock_joblib_load.return_value = mock_model

    result_str = run_credit_risk_model.invoke(SAMPLE_LOAN_APP)
    result = json.loads(result_str)

    mock_exists.assert_called_once()
    mock_joblib_load.assert_called_once()
    mock_model.predict_proba.assert_called_once()
    assert "default_probability" in result
    assert result["default_probability"] == 0.12
