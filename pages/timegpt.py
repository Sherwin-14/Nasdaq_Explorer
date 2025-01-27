import math
import pandas as pd
import streamlit as st

from nixtla import NixtlaClient

API_KEY = 'nixak-WDui79ni2fTamFnvLBKw1lhPBkGPZXdgoWrZBzfzMo2nTC7qpL4paIFYaxpHsM41WDk7vUhyMeFyYFJf'

nixtla_client = NixtlaClient(api_key = API_KEY)

@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    return data

@st.cache_resource
def forcasting(data):
    fcst_df = nixtla_client.forecast(data, h = 168, level=[68, 80], add_history=True)
    return fcst_df


def plot(data, forecast):
    return nixtla_client.plot(data, forecast, level=[80, 90])


st.title("Forecasting with TimeGPT")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key="1")

if uplodaded_data is not None:
    data = load_data(uplodaded_data)
    st.subheader("Choose the algorithm")

    if st.button("Start Forecasting"):
        forecast  = forcasting(data)

        

