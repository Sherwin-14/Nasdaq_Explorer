from app import *


st.title("Forecasting with DL Models")

uploaded_data = st.file_uploader("Choose a CSV File", type="csv", key="40")

@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    return data


if uploaded_data is not None:
    #data = load_data(uplodaded_data)
    st.subheader("Choose the algorithm")
    model_name = st.selectbox(
        "Select the ML Model", ["LSTM"]
    )  # Only show feature engineering settings if XGBoost is selected


 #if st.button("Start Forecasting"):
        #X, y = preprocess_data(data)
        #predictions, rmse, model = train_and_forecast(X, y, model_name, data)
        #st.subheader("Feature Importance")
        #plot_feature_importance(model, X.columns)
        # st.write(next_7_days_pred)
        # forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions})
        # st.write(forecast_df)
