import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go 
import optuna

from app import *
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor 

@st.cache_resource
def create_sma_features(data, window_sizes=[5, 10, 15, 30]): 
    for window_size in window_sizes: 
        data[f'SMA_{window_size}'] = data['Close'].rolling(window=window_size).mean() 

    return data.dropna() 

@st.cache_resource
def create_ema_features(data, window_sizes=[9]): 
    for window_size in window_sizes: 
        data[f'EMA_{window_size}'] = data['Close'].ewm(span=window_size, adjust=False).mean() 

    return data.dropna() 

@st.cache_resource
def create_moving_average_features(data): 

    data = create_sma_features(data) 
    data = create_ema_features(data) 

    return data

@st.cache_resource
def load_data(uploaded_file): 
    data = pd.read_csv(uploaded_file,parse_dates=['Date']) 
    return data

@st.cache_resource
def relative_strength_idx(df, n=14):
     close = df['Close'] 
     delta = close.diff() 
     delta = delta[1:] 
     pricesUp = delta.copy() 
     pricesDown = delta.copy() 
     pricesUp[pricesUp < 0] = 0 
     pricesDown[pricesDown > 0] = 0 
     rollUp = pricesUp.rolling(n).mean() 
     rollDown = pricesDown.abs().rolling(n).mean() 
     rs = rollUp / rollDown 
     rsi = 100.0 - (100.0 / (1.0 + rs)) 
     
     return rsi

@st.cache_resource
def calculate_macd(data, span1=12, span2=26, signal_span=9):
    ema_12 = data['Close'].ewm(span=span1, min_periods=span1).mean()
    ema_26 = data['Close'].ewm(span=span2, min_periods=span2).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=signal_span, min_periods=signal_span).mean()
    return data

@st.cache_resource
def preprocess_data(data, window_sizes_sma=[5, 10, 15, 30], window_sizes_ema=[9], rsi_period=14, macd_spans=(12, 26, 9)):
     data = create_moving_average_features(data) # Calculate RSI and add it as a feature 
     data['RSI'] = relative_strength_idx(data, n=rsi_period).fillna(0)
     data = calculate_macd(data, span1=macd_spans[0], span2=macd_spans[1], signal_span=macd_spans[2])

     data = data.dropna()

     if 'Date' in data.columns: 
         data = data.drop(columns=['Date'])

     print(data.dtypes)

     X = data.drop(columns=['Close']) 
     y = data['Close'] 


     scaler = StandardScaler() 
     X_scaled = scaler.fit_transform(X) 
     X = pd.DataFrame(X_scaled, columns=X.columns)

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
        future_data[f'lag_{i}'] = [data['Close'].iloc[-i]] * 7  
        
    future_data['rolling_mean'] = [data['Close'].iloc[-window_size:].mean()] * 7
    
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
        model = XGBRegressor(**study.best_params,alpha=0.5)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) 

    future_dates = pd.date_range(data['Date'].iloc[-1], periods=7, freq='D') 
    future_features = prepare_future_features(data, lag_steps, window_size).values
    print(future_features)
    next_7_days = model.predict(future_features)

    return predictions, rmse, next_7_days

st.title("Forecasting with ML Models")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

if uplodaded_data is not None:
     data = load_data(uplodaded_data) 
     st.subheader("Choose the algorithm") 
     model_name = st.selectbox("Select the ML Model", ["XGBoost"]) # Only show feature engineering settings if XGBoost is selected 
     
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
         st.write(next_7_days) 
         st.write(f"RMSE: {rmse:.2f}") 
         #forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions}) 
         #st.write(forecast_df)
         