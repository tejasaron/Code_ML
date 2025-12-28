import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model():

    # Load artifacts

    model = joblib.load('artifacts/model.joblib')
    X_test = joblib.load('artifacts/X_test.joblib')
    y_test = joblib.load('artifacts/y_test.joblib')

    # Make predictions 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred), 
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    }

    # Save metrics to a DataFrame
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('artifacts/evaluation_metrics.csv', index=False)

    print("Model Evaluation Metrics")
    print(metrics_df)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))




if __name__ == "__main__":
    evaluate_model()


