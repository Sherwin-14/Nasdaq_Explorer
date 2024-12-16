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
def preprocess_data(data, window_sizes_sma=[5, 10, 15, 30], window_sizes_ema=[9], rsi_period=14, macd_spans=(12, 26, 9), num_days = 7):

     data = create_moving_average_features(data)
     data['RSI'] = relative_strength_idx(data, n=rsi_period).fillna(0)
     data = calculate_macd(data, span1=macd_spans[0], span2=macd_spans[1], signal_span=macd_spans[2])

     #data = create_shifted_targets(data, num_days=num_days)

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


@st.cache_resource
def train_and_forecast(X,y, model_name):

    print(X.isna().sum(),y)

    X, y = preprocess_data(data) 

    train_size = int(len(X) * 0.85) 
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

    return predictions, rmse, model


def plot_feature_importance(model, feature_names):
    # Ensure that the feature importances are normalized and sorted correctly
    feature_importances = model.feature_importances_
    feature_importances = feature_importances / feature_importances.sum() * 100

    # Sort the importances
    sorted_importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    # Separate the sorted features and their importances for plotting
    features_sorted = [x[0] for x in sorted_importances]
    importances_sorted = [x[1] for x in sorted_importances]

    # Plot the sorted importances
    fig = go.Figure(data=[go.Bar(
        x=importances_sorted, 
        y=features_sorted, 
        orientation='h', 
        marker_color='royalblue', 
        marker_line_color='white', 
        marker_line_width=1
    )])

    fig.update_layout(
        xaxis_title='Importance (%)',
        yaxis_title='Feature',
        margin=dict(l=200, r=50, t=50, b=50),
        height=600,
        width=800,
        font=dict(size=12, family='Arial'),
        template='plotly_white',
        xaxis=dict(showgrid=False, tickfont=dict(color='black', size=12, family='Arial'), tickcolor='black'),
        yaxis=dict(showgrid=False, tickfont=dict(color='black', size=12, family='Arial'), tickcolor='black'),
        yaxis_autorange='reversed'
    )

    st.plotly_chart(fig, use_container_width=True)

st.title("Forecasting with ML Models")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

if uplodaded_data is not None:
     data = load_data(uplodaded_data) 
     st.subheader("Choose the algorithm") 
     model_name = st.selectbox("Select the ML Model", ["XGBoost"]) # Only show feature engineering settings if XGBoost is selected 

     if st.button("Start Forecasting"):
         X,y = preprocess_data(data)  
         predictions, rmse, model,= train_and_forecast(X,y, model_name)
         st.write(predictions) 
         st.write(f"RMSE: {rmse:.2f}") 
         st.subheader("Feature Importance")
         plot_feature_importance(model, X.columns)
         #st.write(next_7_days_pred)
         #forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions}) 
         #st.write(forecast_df)
         