from app import *
from prophet import Prophet
from prophet.plot import plot_plotly
from pages.intro import *


st.title("Model your data with FB Prophet")

st.subheader("Raw Data")

st.dataframe(st.session_state.df.head(),use_container_width = True)

n_years = st.slider("Years of Prediction",1,4)
period = n_years * 365

st.session_state.df['Date'] = st.session_state.df['Date'].dt.tz_localize(None)

train = st.session_state.df[['Date','Close']]
train['Date'] = pd.to_datetime(train['Date']) # Ensure dates are in correct format
train = train.rename(columns = {"Date":"ds","Close":"y"})

m = Prophet()
try:
    m.fit(train) 
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future) 
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
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