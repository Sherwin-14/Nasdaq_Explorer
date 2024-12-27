import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

from app import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    if np.isnan(df1).any(): 
        st.write("Scaled data contains NaN values.") 
        df1 = np.nan_to_num(df1)

    training_size = int(len(df1) * 0.80) 
    test_size = len(df1) - training_size 
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1] # Create dataset for LSTM 

    X_train, y_train = create_dataset(train_data, time_step) 
    X_test, ytest = create_dataset(test_data, time_step) # Reshape input to be [samples, time steps, features] which is required for LSTM 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 
    
    return X_train, y_train, X_test, ytest, scaler


def train_and_test_lstm(X_train, y_train, X_test, ytest, input_dim=1, hidden_dim=50, output_dim=1, n_layers=2, dropout=0.2, num_epochs=10, learning_rate=0.001):
    # Initialize the LSTM model
    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers, dropout)

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    ytest_torch = torch.tensor(ytest, dtype=torch.float32)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_torch)
        optimizer.zero_grad()

        # Calculate the loss
        loss = criterion(outputs, y_train_torch.unsqueeze(1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            st.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    st.write("Model training complete.")

    # Evaluate the model on the test data
    model.eval()
    test_outputs = model(X_test_torch)
    test_loss = criterion(test_outputs, ytest_torch.unsqueeze(1))
    st.write(f'Test Loss: {test_loss.item():.4f}')

    # Calculate RMSE
    rmse_lstm = math.sqrt(mean_squared_error(ytest, test_outputs.detach().numpy()))
    st.metric("RMSE of the LSTM Model", f"{rmse_lstm:.4f}")

    return model, test_outputs, rmse_lstm



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
        
        model, test_outputs, rmse_lstm = train_and_test_lstm(X_train, y_train, X_test, ytest)
        st.write(model)
        st.write(rmse_lstm) 
            #predictions, rmse, model = train_and_forecast(X, y, model_name, data)
            #st.subheader("Feature Importance")
            #plot_feature_importance(model, X.columns)
            # st.write(next_7_days_pred)
            # forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions})
        # st.write(forecast_df)
