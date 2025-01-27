import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from nixtla import NixtlaClient
from utilsforecast.preprocessing import fill_gaps


API_KEY = 'nixak-WDui79ni2fTamFnvLBKw1lhPBkGPZXdgoWrZBzfzMo2nTC7qpL4paIFYaxpHsM41WDk7vUhyMeFyYFJf'

nixtla_client = NixtlaClient(api_key = API_KEY)

@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    data['unique_id'] = 'id1'
    data.rename(columns={"Date": "ds", "Close": "y"}, inplace = True)
    data = data[['unique_id', 'ds', 'y']]
    data = fill_gaps(data, freq = 'D')
    data['y'] = data['y'].interpolate(method='linear', limit_direction='both')

    return data

@st.cache_resource
def forcasting(data):
    fcst_df = nixtla_client.forecast(data, h = 7, level=[68, 80], add_history= True, freq = 'D')
    return fcst_df


def plot(data, forecast):
    plt.figure(figsize=(18, 8))
    fig =  nixtla_client.plot(data, forecast, level=[68, 80])
    return st.pyplot(fig)


st.title("Forecasting with TimeGPT")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key="1")

if uplodaded_data is not None:
    data = load_data(uplodaded_data)

    if st.button("Start Forecasting"):
        forecast  = forcasting(data)
        st.subheader("Model predictions For Next 7 Days")
        st.dataframe(forecast[-7:], use_container_width=True)
        st.subheader("Model Performance")
        plot(data, forecast)

        

