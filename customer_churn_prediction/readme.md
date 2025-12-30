Customer Churn Prediction with Explainability

This project implements an end-to-end customer churn prediction system using classical machine learning with a strong emphasis on interpretability and production deployment.

A logistic regression model is trained using a full preprocessing pipeline (scaling and categorical encoding) to predict the probability of customer churn in a telecom setting. Model performance is evaluated using ROC-AUC, precision, recall, and confusion matrix analysis, with a business-optimized decision threshold selected to reduce false negatives.

To ensure transparency, SHAP explainability is integrated to identify and visualize key churn drivers such as contract type, tenure, and monthly charges, enabling both global and individual-level insights.

The trained model is deployed as a FastAPI REST service with strict input validation and consistent inference behavior. The service is fully Dockerized with pinned dependencies to guarantee reproducibility and reliable model serving in production environments.

This repository demonstrates industry-standard practices across modeling, evaluation, explainable AI, and deployment.


How to Run

Train

python src/models/train.py


Evaluate

python src/evaluation.py


SHAP Explain

python src/explainability.py


Run API

uvicorn src.api.app:app --reload


Docker

docker build -t churn-api .
docker run -p 8000:8000 churn-api