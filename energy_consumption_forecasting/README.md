ENERGY CONSUMPTION FORECASTING FOR SMART GRIDS

OVERVIEW
This project implements an end-to-end machine learning system for short-term energy consumption forecasting in smart grids. It is designed following industry-standard machine learning engineering practices, covering the complete lifecycle from raw data processing to model deployment.

The system predicts future electricity demand (measured in MW) using historical consumption patterns and engineered temporal features. Predictions are exposed through a RESTful API suitable for real-time and batch inference use cases.

PROBLEM STATEMENT
Accurate energy demand forecasting is essential for modern power systems to ensure grid stability, optimize load balancing, support renewable energy integration, and enable cost-efficient operations such as dynamic pricing.

This project addresses the forecasting problem using supervised time-series regression models trained on utility-level energy consumption data, while preventing data leakage and ensuring production readiness.

KEY FEATURES

* Utility-wise time-series processing to avoid cross-entity data leakage
* Feature engineering using lag variables, rolling statistics, and time-based features
* Time-aware train and validation split aligned with real-world forecasting scenarios
* Baseline model (Linear Regression) for performance benchmarking
* Advanced non-linear model (Random Forest Regressor)
* Model persistence using joblib
* FastAPI-based inference service
* Dockerized deployment for portability and scalability
* Structured logging and error handling for production observability

SYSTEM ARCHITECTURE

Raw Energy Data (CSV)
|
v
Data Loading and Validation
|
v
Feature Engineering
(Time Features, Lag Features, Rolling Statistics)
|
v
Time-Based Train / Validation Split
|
v
Model Training

* Linear Regression (Baseline)
* Random Forest (Advanced)
  |
  v
  Model Persistence (Joblib)
  |
  v
  FastAPI Prediction Service
  |
  v
  Docker Container

TECHNOLOGY STACK

* Programming Language: Python 3.10
* Data Processing: Pandas, NumPy
* Machine Learning: Scikit-learn
* API Framework: FastAPI
* Model Serving: Uvicorn
* Containerization: Docker
* Version Control: Git

MODEL EVALUATION
Models are evaluated using industry-standard regression metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

A baseline linear regression model is established first. More complex models are only introduced after demonstrating measurable performance improvements over the baseline.

API USAGE

Running the API using Docker:

docker run -p 8000:8000 energy-forecasting-api

Prediction Endpoint:
POST /predict

Sample Request Payload:
{
"hour": 14,
"day_of_week": 2,
"month": 7,
"lag_1": 15000,
"lag_24": 14800,
"lag_168": 14200,
"rolling_mean_24": 14950,
"rolling_mean_168": 14500
}

Sample Response:
{
"predicted_load_mw": "***"
}

INDUSTRY ALIGNMENT
This project reflects real-world machine learning system design through:

* Reproducible data preparation pipelines
* Clear separation of training and inference logic
* Time-series aware validation strategy
* Containerized deployment suitable for cloud environments
* Logging and error handling to support monitoring and debugging

FUTURE ENHANCEMENTS

* Probabilistic forecasting with confidence intervals
* Integration of renewable energy inputs (solar and wind)
* Cloud deployment on AWS or GCP
* Automated retraining and model versioning pipelines

AUTHOR
Tejas Arondekar
