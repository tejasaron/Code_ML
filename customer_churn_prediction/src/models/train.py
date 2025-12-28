import os
import joblib
from src.data.preprocess import prepare_data  
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import sys
from pathlib import Path

# 1. FIX THE PATH FIRST
# If train.py is in src/models/, parents[2] takes you to the project root
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))  # Must be a string

 # modular architecture


DATA_PATH = "data/raw/Telco_Customer_Churn.csv"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

X, y, preprocessor = prepare_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

log_reg = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', log_reg)     
])

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC-AUC on test set: {roc_auc:.4f}")

joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'churn_model.pkl'))