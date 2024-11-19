import pandas as pd 
import numpy as np 
import streamlit as st 
import plotly.graph_objects as go 
import streamlit_antd_components as sac
import pickle

from app import *
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, TimeSeriesSplit 

st.title("Create SARIMA Model")

tab1, tab2 = st.tabs(["Best Model Parameters", "Optimal Model"])

with tab1:

    st.subheader("Upload a file for creation of sarima model")
    uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "3")

    st.session_state.pdqs_values  =  None

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
                    sac.result(label ='', description = 'This suggests that the data might be white noise or random walk. Please select any other dataset or add more data to current one!',status = 500 )
                    return (p, d, q, P, D, Q, s)
                
                if P == 0 and D == 0 and Q == 0: 
                    sac.result(label ='', description = 'This suggests that the data might not have seasonal patterns, please upload a different dataset or move to ARIMA section',status = 500 )
                    return (p, d, q, P, D, Q, s)
                
                sac.result(

                    label='Result',
                    description =f'The best SARIMA model for this dataset would have parameters of ({model.order[0]}, {model.order[1]}, {model.order[2]}) and seasonal order of ({model.seasonal_order[0]}, {model.seasonal_order[1]}, {model.seasonal_order[2]}, {model.seasonal_order[3]})',
                    status='success', icon = sac.BsIcon(name='house', size = 50, color=None)
                )
                st.session_state.pdqs_values = (p, d, q, P, D, Q, s) 
                return (p, d, q, P, D, Q, s)
             
             else: 
                return (True, True, True, True, True, True, True)
             
        p, d, q, P, D, Q, s = run_sarima(df)  

        if st.session_state.pdqs_values is None:
          st.session_state.pdqs_values =   p, d, q, P, D, Q , s

with tab2:
      
      if st.session_state.pdqs_values is not None:
            p, d, q, P, D, Q, S = st.session_state.pdqs_values
            st.subheader("Upload the same file for modelling")
            uploaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key = "4")

            if uploaded_data is not None:

                try:
                    df = pd.read_csv(uploaded_data, parse_dates=['Date'])
                    st.subheader("Data after preprocessing and stationarity check")
                    st.dataframe(df.sample(5), use_container_width=True)
                    st.warning("Before clicking the button below make sure that you have identified best set of parameters from previous tab.")
                except pd.errors.EmptyDataError:
                    st.error("The uploaded file is empty.")
                except pd.errors.ParserError:
                    st.error("Error parsing the uploaded file.")
                except Exception as e:
                    st.error(f"Error uploading file: {e}")

            else:
                 st.write("Please upload a CSV file")
      else:   
          st.write("Please upload the csv")
          print(st.session_state.pdqs_values)         



