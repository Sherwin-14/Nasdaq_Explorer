import pickle
import pandas as pd

from app import *
from prophet import Prophet
from prophet.plot import plot_plotly
from pages.intro import *


st.title("Forecasting with FB Prophet")

st.warning("Please choose the appropriate stock in the Choose Stock To Analzye section and download the dataset before attempting forecasting.")

st.subheader("Raw Data")

uplodaded_data  = st.file_uploader("Choose a CSV file", type=["csv"],key = "10")

if uplodaded_data is not None:

    df = pd.read_csv(uplodaded_data)

    st.dataframe(df.head(),use_container_width = True)

    n_years = st.slider("Years of Prediction",1,4)
    period = n_years * 365

    train = df[['Date','Close']]
    train['Date'] = pd.to_datetime(train['Date']).dt.tz_localize(None)
    train = train.rename(columns = {"Date":"ds","Close":"y"})

    m = Prophet()
    try:
        m.fit(train) 
        future = m.make_future_dataframe(periods = period)
        forecast = m.predict(future) 
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        model_bytes = pickle.dumps(m)
    except Exception as e: 
        st.write(f"Error: {e}" )  

    st.subheader("Forecast Data")
    st.dataframe(forecast.tail().style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}]).background_gradient(cmap='Blues'), use_container_width=True)

    with st.container():
        st.subheader("Forecast Data With Predicitions For The Coming Years")
        fig1  = plot_plotly(m, forecast)
        st.plotly_chart(fig1)


    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)   

    st.download_button(
                label = "Download The FB Prophet Model",
                data=model_bytes,
                file_name="prophet_model.pkl",
                mime="application/octet-stream"
    )

else: 
    st.warning("Please upload a CSV file to proceed.")

