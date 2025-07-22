import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

print("Training dummy fraud detection model...")

# 1. Create dummy data
# In a real scenario, you would load the IEEE Fraud Detection dataset here.
X = pd.DataFrame(np.random.rand(100, 2), columns=["feature1", "feature2"])
y = pd.Series(np.random.randint(0, 2, 100))

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 4. Evaluate (optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dummy Model Accuracy: {accuracy:.2f}")

# 5. Save the model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "fraud_detection_model.joblib")
joblib.dump(model, model_path)

print(f"Dummy fraud detection model saved to {model_path}")
