# Smart Grid Energy Forecasting & Fault Detection (Real-Time ML)

## Project Overview
This project builds a production-grade ML system for forecasting real-time energy load (kW)
using smart-grid telemetry such as voltage, current, active power, reactive power,
solar/wind generation, and environmental factors.

Dataset:
- 50,000+ rows
- 15-minute interval data
- Smart-grid parameters: Voltage, current, active/reactive power
- Renewable inputs: Solar & wind power generation
- Grid import power
- Environmental: Temperature, humidity, electricity price
- Fault indicators
- Target: Predicted Load (kW)

## ML Tasks
1. Short-term load forecasting (15-min, 1-hour, 24-hour)
2. Renewable integration analysis
3. Fault detection (classification)
4. Dynamic pricing impact study
5. Anomaly detection (voltage instability, overload)

## Deployment
- FastAPI inference API
- Docker container
- Real-time inference
- Monitoring with Prometheus + Grafana

## Evaluation Metrics
- MAE, RMSE, MAPE
- F1-score for faults
- Compared against persistence and seasonal baselines
