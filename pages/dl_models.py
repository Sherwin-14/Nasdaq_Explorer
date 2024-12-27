import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from app import *
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@st.cache_resource
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    return data

@st.cache_resource
def create_dataset(dataset, time_step=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset) - time_step - 1): 
        a = dataset[i:(i + time_step), 0] 
        dataX.append(a) 
        dataY.append(dataset[i + time_step, 0]) 

    return np.array(dataX), np.array(dataY)

def preprocess_data(df, time_step): # Scale the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1)) # Splitting dataset into train and test 
    training_size = int(len(df1) * 0.65) 
    test_size = len(df1) - training_size 
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1] # Create dataset for LSTM 
    X_train, y_train = create_dataset(train_data, time_step) 
    X_test, ytest = create_dataset(test_data, time_step) # Reshape input to be [samples, time steps, features] which is required for LSTM 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 
    
    return X_train, y_train, X_test, ytest


st.title("Forecasting with DL Models")

uploaded_data = st.file_uploader("Choose a CSV File", type="csv", key="40")


if uploaded_data is not None:
    #data = load_data(uplodaded_data)
    st.subheader("Choose the algorithm")
    model_name = st.selectbox(
        "Select the ML Model", ["LSTM"]
    )  # Only show feature engineering settings if XGBoost is selected


if st.button("Start Forecasting"):
    n = 100 
    df = load_data(uploaded_data)
    df1 = df.reset_index()['Close']
    X_train, y_train, X_test, ytest, scaler = preprocess_data(df1, n)

    st.write("LSTM model data preprocessing complete.") 
    st.write("X_train shape:", X_train.shape) 
    st.write("y_train shape:", y_train.shape) 
    st.write("X_test shape:", X_test.shape) 
    st.write("ytest shape:", ytest.shape)

    input_dim = 1 
    hidden_dim = 50 
    output_dim = 1 
    n_layers = 2 
    dropout = 0.2 
    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers, dropout) 
    
    st.write("LSTM model created.") 
    st.write(model)
        #predictions, rmse, model = train_and_forecast(X, y, model_name, data)
        #st.subheader("Feature Importance")
        #plot_feature_importance(model, X.columns)
        # st.write(next_7_days_pred)
        # forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions})
        # st.write(forecast_df)
