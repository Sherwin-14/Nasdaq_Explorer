import pandas as pd 
import numpy as np 
import streamlit as st 
import plotly.graph_objects as go 
import pickle

from app import *
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, TimeSeriesSplit 
import pickle

st.title("Create SARIMA Model")

tab1, tab2 = st.tabs(["Best Model Parameters", "Optimal Model"])

with tab1:

    st.subheader("Upload a file for creation of arima model")
    uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"])

    if uplodaded_data is not None:

        df = pd.read_csv(uplodaded_data, parse_dates=['Date']) 
        st.subheader("Data after preprocessing and stationarity check") 
        st.dataframe(df.sample(5), use_container_width=True) 

        def run_sarima(df):
             
             if st.button('Get the best model parameters'): 
                df['Date'] = pd.to_datetime(df['Date']) 
                df = df.dropna() 
                train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False) 
                model = auto_arima(train_df['Close'], start_p=1, start_d=1, start_q=1, max_p=4, max_d=3, max_q=4, m=12, seasonal=True, error_action='warn', suppress_warnings=True, stepwise=True) 

                p, d, q = model.order 
                P, D, Q, s = model.seasonal_order

                if p == 0 and q == 0 and d == 0: 
                    st.markdown(""" #### Model order is (0,0,0). This may indicate that the data is a simple random walk. Try to collect more data or use a different model selection method. """) 
                    return (p, d, q, P, D, Q, s)
                
                st.markdown(f""" #### Result : The best SARIMA model for this dataset would have parameters of ({model.order[0]}, {model.order[1]}, {model.order[2]}) and seasonal order of ({model.seasonal_order[0]}, {model.seasonal_order[1]}, {model.seasonal_order[2]}, {model.seasonal_order[3]}) """) 
                st.session_state.pdq_values = (p, d, q, P, D, Q, s) 
                return (p, d, q, P, D, Q, s)
             
             else: 
                return (True, True, True, True, True, True, True)
             
        p, d, q, P, D, Q, s = run_sarima(df)

