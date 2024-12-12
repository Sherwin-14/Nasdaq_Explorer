import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go 
import optuna

from app import *
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from xgboost import XGBRegressor 

@st.cache_resource
def create_lag_features(data, lag_steps=1): 
    for i in range(1, lag_steps + 1): 
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data.dropna() 

@st.cache_resource
def create_rolling_mean(data, window_size=3): 
    data['rolling_mean'] = data['Close'].rolling(window=window_size).mean() 
    return data.dropna() 

@st.cache_resource
def apply_fourier_transform(data): 
    fft = np.fft.fft(data['Close'].values) 
    data['fourier_transform'] = np.abs(fft) 
    return data

@st.cache_resource
def load_data(uploaded_file): 
    data = pd.read_csv(uploaded_file,parse_dates=['Date']) 
    return data

@st.cache_resource
def preprocess_data(data, lag_steps, window_size,model_name):
     if model_name == "XGBoost":
         data = create_lag_features(data, lag_steps) 
         data = create_rolling_mean(data, window_size) 
         data = data.dropna() 

     if 'Date' in data.columns: 
         data = data.drop(columns=['Date'])

     print(data.dtypes)

     X = data.drop(columns=['Close']) 
     y = data['Close'] 

     return X, y

def objective(trial, X_train, y_train, X_test, y_test, model_name):
    
     if model_name == "XGBoost": 
        param = { 'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3) } 
        model = XGBRegressor(**param)

     model.fit(X_train, y_train) 
     predictions = model.predict(X_test) 
     rmse = np.sqrt(mean_squared_error(y_test, predictions)) 

     return rmse

def prepare_future_features(data, lag_steps, window_size): 
    last_row = data.iloc[-1:].copy() 
    future_data = pd.DataFrame() 
    for i in range(1, lag_steps + 1): 
        future_data[f'lag_{i}'] = last_row['Close'].shift(i) 
        
    future_data['rolling_mean'] = last_row['Close'].rolling(window=window_size).mean() 
    future_data = future_data.dropna() 
    
    return future_data

@st.cache_resource
def train_and_forecast(X,y, model_name):

    print(X.isna().sum(),y)

    if model_name == "XGBoost": 
        X, y = preprocess_data(data, lag_steps, window_size, model_name) 
    else: 
        X, y = preprocess_data(data, lag_steps=1, window_size=1, model_name=model_name)

    train_size = int(len(X) * 0.8) 
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:] 
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    study = optuna.create_study(direction='minimize') 
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_name), n_trials=50) 
    
    st.write("Best parameters found: ", study.best_params) 
    st.write("Best RMSE: ", study.best_value)

    if model_name == "XGBoost": 
        model = XGBRegressor(**study.best_params)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) 

    future_dates = pd.date_range(data['Date'].iloc[-1], periods=7, freq='D') 
    future_features = prepare_future_features(data, lag_steps, window_size).values
    next_7_days = model.predict(future_features)

    return predictions , rmse , next_7_days

st.title("Forecasting with ML Models")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

if uplodaded_data is not None:
     data = load_data(uplodaded_data) 
     st.subheader("Choose the algorithm") 
     model_name = st.selectbox("Select the ML Model", ["Linear Regression", "XGBoost"]) # Only show feature engineering settings if XGBoost is selected 
     
     if model_name == "XGBoost": 
        st.subheader("Feature Engineering Settings") 
        lag_steps = st.number_input("Lag Steps", min_value=1, max_value=10, value=3) 
        window_size = st.number_input("Rolling Mean Window Size", min_value=1, max_value=10, value=3)

     if st.button("Start Forecasting"):
         
         if model_name == "XGBoost": 
            X,y = preprocess_data(data, lag_steps, window_size,model_name=model_name) 

         else: 
            data = preprocess_data(data, lag_steps=1, window_size=1, model_name = model_name) 
            
         predictions, rmse, next_7_days = train_and_forecast(X,y, model_name)
         st.write(predictions, next_7_days) 
         st.write(f"RMSE: {rmse:.2f}") 
         #forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions}) 
         #st.write(forecast_df)
         