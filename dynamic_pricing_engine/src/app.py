from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np


app = FastAPI(title = ' Dynamic Pricing Engine')

# Load Model + Feature List

model = joblib.load('D:/Code_ML/dynamic_pricing_engine/models/xgb_demand_model.pkl')
features = joblib.load('D:/Code_ML/dynamic_pricing_engine/models/feature_list.pkl')

# Load Data

df = pd.read_csv('D:/Code_ML/dynamic_pricing_engine/data/processed/processed_data.csv', parse_dates=['Date'])

#================
# Helper Functions
#================

def prepare_row(product_id, price):
    latest = df[df['Product_ID']==product_id].sort_values('Date').iloc[-1].copy()

    latest['Price']= price
    latest['Margin']= latest['Price'] - latest['Base_Cost']
    latest['Price_Cost_Ratio'] = latest['Price']/latest['Base_Cost']
    latest['Competitor_Delta'] = latest['Competitor_Price'] - latest['Price']

    X = pd.DataFrame([latest[features]])
    return X

def simulate_prices(product_id,price_range):

    latest = df[df['Product_ID']==product_id].sort_values('Date').iloc[-1].copy()
    rows = []

    for price in price_range:
        
        X= prepare_row(product_id,price)

        demand = model.predict(X)[0]
        demand = max(demand,0)

        revenue = price * demand
        profit = (price - latest['Base_Cost']) * demand

        rows.append([price, demand, revenue, profit])

    return pd.DataFrame(rows, columns=['Price','Predicted_Demand', 'Revenue','Profit'])


#=========================
# API Endpoints
#=========================

# Demand Prediction Endpoint

@app.get('/predict-demand')

def predict_demand(product_id: str, price: float):
    X = prepare_row(product_id, price)

    demand = float(model.predict(X)[0])

    return {'Product_ID': product_id, 'Price': price, 'Predicted_Demand': max(demand,0)}


#=========================
# Price Optimization Endpoints

@app.get('/optimal-price')

def optimal_price(product_id: str):

    latest = df[df['Product_ID'] == product_id].sort_values('Date').iloc[-1]

    price_range = np.linspace(latest['Base_Cost'], latest['Base_Cost'] * 2.5, 40)

    sims = simulate_prices(product_id, price_range)

    best = sims.loc[sims['Revenue'].idxmax()]

    return {
        'product_id': product_id,
        'optimal_price': float(best['Price']),# type: ignore
        'predicted_demand': float(best['Predicted_Demand']),# type: ignore
        'expected_revenue': float(best['Revenue']),# type: ignore
        'expected_profit': float(best['Profit']) # type: ignore
    }

#=========================
# Price Curve Simulation Endpoint 


@app.get('/simulate-curve')

def get_curve(product_id: str):
    latest = df[df['Product_ID'] == product_id].sort_values('Date').iloc[-1]

    price_range = np.linspace(latest['Base_Cost'], latest['Base_Cost'] * 2.5, 40)

    sims = simulate_prices(product_id, price_range)

    return sims.to_dict(orient='records')

