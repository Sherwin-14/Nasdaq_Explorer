from app import *


st.title("Create An ARIMA Based Model")

st.subheader("Upload a file for creating arima model")
uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"])


if uplodaded_data is not None:
    df = pd.read_csv(uplodaded_data)
    st.subheader("Data after preprocessing and stationarity check")
    st.dataframe(df.sample(5),use_container_width = True)