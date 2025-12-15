# Multi-Utility Energy Consumption Forecasting

## Project Overview
This project focuses on forecasting hourly electricity demand for multiple energy

## Problem Statement
Energy providers must accurately predict future electricity demand :

- Maintain grid stability
- Optimize generation planning
- Reduce operational risk
- Integrate renewable energy sources effectively

This project builds a machine learning-based forecasting pipeline that predicts short-term electricity load using historical consumption patterns.

## Dataset Description
- Hourly electricity consumption data (MW)
- Multiple energy utility companies
- Time span across multiple years
- Data merged into a unified time-series with a utility identifier

## Target Variable
- **MW**: Hourly electricity demand

## Key Objectives
- Forecast short-term energy demand
- Compare consumption patterns across utilities
- Build a production-ready ML pipeline
- Prepare the model for deployment

## Project Structure

energy_consumption_forecasting/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
├── src/
│ ├── data/
│ ├── features/
│ ├── models/
│ └── visualization/
├── models/
└── reports/

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Matplotlib, Seaborn
- Git & GitHub

## Status
In progress