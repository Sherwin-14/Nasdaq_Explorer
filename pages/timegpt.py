import math
import pandas as pd
import streamlit as st

from nixtla import NixtlaClient
from utilsforecast.preprocessing import fill_gaps
from sklearn.model_selection import train_test_split
from utilsforecast.losses import rmse
from utilsforecast.evaluation import evaluate


API_KEY = 'nixak-WDui79ni2fTamFnvLBKw1lhPBkGPZXdgoWrZBzfzMo2nTC7qpL4paIFYaxpHsM41WDk7vUhyMeFyYFJf'

nixtla_client = NixtlaClient(api_key = API_KEY)

@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    data['unique_id'] = 'id1'
    data.rename(columns={"Date": "ds", "Close": "y"}, inplace = True)
    data = data[['unique_id', 'ds', 'y']]
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df = fill_gaps(data, freq = 'D')
    test_df = fill_gaps(data, freq = 'D')
    train_df['y'] = train_df['y'].interpolate(method='linear', limit_direction='both')
    test_df['y'] = test_df['y'].interpolate(method='linear', limit_direction='both')

    return train_df, test_df

@st.cache_resource
def training(data, test):
    fcst_df = nixtla_client.forecast(data, h = len(test), level=[68, 80], add_history= True, freq = 'D', finetune_steps=10, finetune_loss='rmse', id_col='unique_id')
    print("Training Successful!")
    return fcst_df


@st.cache_resource
def error(test_df):
    evaluation = evaluate(test_df, metrics=[rmse], models=["TimeGPT"], target_col="y", id_col='unique_id')
    average_metrics = evaluation.groupby('metric')['TimeGPT'].mean()
    return average_metrics

@st.cache_resource
def forecasting(test_df):
    fcst_df_7 = nixtla_client.forecast(test_df, h = 7, level=[68, 80], freq = 'D')
    return fcst_df_7


def plot(test_df, forecast):
    fig =  nixtla_client.plot(test_df, forecast, level=[68, 80])
    return st.pyplot(fig)

st.title("Forecasting with TimeGPT")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key="1")

if uplodaded_data is not None:
    train_df, test_df = load_data(uplodaded_data)

    if st.button("Start Forecasting"):
        fcst_df  = training(train_df, test_df)
        fcst_df['ds'] = pd.to_datetime(fcst_df['ds'])
        test_df = pd.merge(test_df, fcst_df, 'left', ['unique_id', 'ds'])
        rmse = error(test_df)
        st.write("Best RMSE:",rmse)
        st.subheader("Model predictions For Next 7 Days")

        

