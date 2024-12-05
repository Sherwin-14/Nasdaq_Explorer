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

def create_lag_features(data, lag_steps=1): 
    for i in range(1, lag_steps + 1): data[f'lag_{i}'] = data['Close'].shift(i)
    return data.dropna() 

def create_rolling_mean(data, window_size=3): 
    data['rolling_mean'] = data['Close'].rolling(window=window_size).mean() 
    return data.dropna() 

def apply_fourier_transform(data): 
    fft = np.fft.fft(data['Close'].values) 
    data['fourier_transform'] = np.abs(fft) 
    return data

@st.cache_resource
def load_data(uploaded_file): 
    data = pd.read_csv(uploaded_file,parse_dates=['Date']) 
    return data

def preprocess_data(data, lag_steps, window_size):
     data = create_lag_features(data, lag_steps) 
     data = create_rolling_mean(data, window_size)
     data = apply_fourier_transform(data) 

     if 'Date' in data.columns: 
         data = data.drop(columns=['Date'])
     
     return data.dropna()

def objective(trial, X_train, y_train, X_test, y_test, model_name):
     
     if model_name == "Linear Regression":
         model = LinearRegression() 

     elif model_name == "XGBoost": 
        param = { 'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3) } 
        model = XGBRegressor(**param)

     elif model_name == "Random Forest": 
        param = { 'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 20) } 
        model = RandomForestRegressor(**param)

     model.fit(X_train, y_train) 
     predictions = model.predict(X_test) 
     rmse = np.sqrt(mean_squared_error(y_test, predictions)) 
     return rmse

def train_and_forecast(data, model_name):

    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:] 

    X_train, y_train = train.drop(columns=['Close']), train['Close'] 
    X_test, y_test = test.drop(columns=['Close']), test['Close']

    study = optuna.create_study(direction='minimize') 
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_name), n_trials=50) 
    
    st.write("Best parameters found: ", study.best_params) 
    st.write("Best RMSE: ", study.best_value)

    if model_name == "Linear Regression": 
        model = LinearRegression() 

    elif model_name == "XGBoost": 
        model = XGBRegressor(**study.best_params)

    elif model_name == "Random Forest":
        model = RandomForestRegressor(**study.best_params) 

    elif model_name == "Gradient Boosting": 
        model = GradientBoostingRegressor(**study.best_params)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) 
    
    return test.index, y_test, predictions, rmse


st.title("Forecasting with ML Models")

st.subheader("Choose dataset for forecasting")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

if uplodaded_data is not None:
     data = load_data(uplodaded_data) 
     st.write(data.head()) 
     st.subheader("Choose the algorithm") 
     model_name = st.selectbox("Select the ML Model", ["Linear Regression", "XGBoost", "Random Forest", "Gradient Boosting"]) # Only show feature engineering settings if XGBoost is selected 
     
     if model_name == "XGBoost": 
        st.subheader("Feature Engineering Settings") 
        lag_steps = st.number_input("Lag Steps", min_value=1, max_value=10, value=3) 
        window_size = st.number_input("Rolling Mean Window Size", min_value=1, max_value=10, value=3)

     if st.button("Start Forecasting"):
         
         if model_name == "XGBoost": 
            data = preprocess_data(data, lag_steps, window_size) 

         else: 
            data = preprocess_data(data, lag_steps=1, window_size=1) 
            
         test_index, y_test, predictions, rmse = train_and_forecast(data, model_name) 
         st.write(f"RMSE: {rmse:.2f}") 
         forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions}) 
         st.write(forecast_df)
         