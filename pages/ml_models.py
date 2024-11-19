from app import *


st.title("Forecasting with ML Models")

st.subheader("Choose dataset for forecasting")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

model = st.selectbox("Select the ML Model",["Linear Regression","XGBoost","Random Forest","Gradient Boosting"])