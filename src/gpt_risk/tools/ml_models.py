from langchain_core.tools import tool
import joblib
import pandas as pd
import json
import os

MODEL_DIR = "models"


@tool
def run_fraud_detection_model(transaction_data: dict) -> str:
    """
    Analyzes a transaction using a pre-trained XGBoost model to predict the probability of fraud.
    Input should be a dictionary representing the transaction.
    Returns a JSON string with the fraud probability and key contributing features.
    """
    model_path = os.path.join(MODEL_DIR, "fraud_detection_model.joblib")
    if not os.path.exists(model_path):
        return json.dumps(
            {"error": "Fraud detection model not found. Please train it first."}
        )

    try:
        model = joblib.load(model_path)

        # In a real application, you would preprocess the input_data dictionary
        # into the format the model expects (e.g., a Pandas DataFrame).
        # For this example, we'll use a placeholder.
        # Assuming the model expects a DataFrame with a single row.
        data_for_prediction = pd.DataFrame([transaction_data])

        # Mock preprocessing
        if "transaction_amount" in data_for_prediction.columns:
            data_for_prediction["amount_log"] = pd.np.log1p(
                data_for_prediction["transaction_amount"]
            )
        # ... add other feature engineering steps as used in training...

        # For this dummy model, we'll just create a dummy feature.
        dummy_features = pd.DataFrame([[0.5, 1]], columns=["feature1", "feature2"])

        probability = model.predict_proba(dummy_features)[:, 1]

        return json.dumps(
            {
                "fraud_probability": float(probability),
                "contributing_features": [
                    "transaction_amount",
                    "merchant_category",
                ],  # Placeholder
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Model inference failed: {str(e)}"})


@tool
def run_credit_risk_model(loan_application_data: dict) -> str:
    """
    Analyzes a loan application using a pre-trained Logistic Regression model to predict the probability of default.
    Input should be a dictionary representing the loan application.
    Returns a JSON string with the default probability.
    """
    model_path = os.path.join(MODEL_DIR, "credit_risk_model.joblib")
    if not os.path.exists(model_path):
        return json.dumps(
            {"error": "Credit risk model not found. Please train it first."}
        )

    try:
        model = joblib.load(model_path)

        # Similar to the fraud model, preprocess the input data here.
        data_for_prediction = pd.DataFrame([loan_application_data])

        # Mock preprocessing for the dummy model
        dummy_features = pd.DataFrame([[0.8, 0.2]], columns=[])

        probability = model.predict_proba(dummy_features)[:, 1]

        return json.dumps({"default_probability": float(probability)})
    except Exception as e:
        return json.dumps({"error": f"Model inference failed: {str(e)}"})
