from app import *


st.title("Forecasting with ML Models")

st.subheader("Choose dataset for forecasting")

uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "1")

print(uplodaded_data)

st.subheader("Choose the algorithm")

model = st.selectbox("Select the ML Model",["Linear Regression","XGBoost","Random Forest","Gradient Boosting"])

st.button("Start Forecasting")

def create_lag_features(data, lag_steps=1): 
    for i in range(1, lag_steps + 1): data[f'lag_{i}'] = data['target'].shift(i)
    return data.dropna() 

def create_rolling_mean(data, window_size=3): 
    data['rolling_mean'] = data['target'].rolling(window=window_size).mean() 
    return data.dropna() 

def apply_fourier_transform(data): 
    fft = np.fft.fft(data['target'].values) 
    data['fourier_transform'] = np.abs(fft) 
    return data

@st.cache 
def load_data(uploaded_file): 
    data = pd.read_csv(uploaded_file) 
    return data

def preprocess_data(data, lag_steps, window_size):
     data = create_lag_features(data, lag_steps) 
     data = create_rolling_mean(data, window_size)
     data = apply_fourier_transform(data) 
     
     return data.dropna()

def objective(trial, X_train, y_train, X_test, y_test, model_name):
     
     if model_name == "Linear Regression":
         model = LinearRegression() 
     elif model_name == "XGBoost": 
        param = { 'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3) } 
        model = XGBRegressor(**param)
     elif model_name == "Random Forest": 
        param = { 'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 20) } 
        model = RandomForestRegressor(**param)