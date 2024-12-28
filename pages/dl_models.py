import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import optuna
import plotly.graph_objs as go

from app import *
from torch.utils.data import Dataset, DataLoader
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
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device) 
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
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

@st.cache_resource
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

def objective(trial, X_train, y_train, X_test, ytest):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # Suggest hyperparameters
    input_dim = 1
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = 16 # Adjust as needed 
    accumulation_steps = 4
    num_epochs = 5

    # Initialize the LSTM model
    model = LSTMModel(input_dim, hidden_dim, output_dim=1, n_layers=n_layers, dropout=dropout).to(device)

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    ytest_torch = torch.tensor(ytest, dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(num_epochs): 
        model.train() 
        optimizer.zero_grad() 
        for i in range(0, len(X_train_torch), batch_size): 
            X_batch = X_train_torch[i:i + batch_size] 
            y_batch = y_train_torch[i:i + batch_size]
            outputs = model(X_batch) 
            loss = criterion(outputs, y_batch.unsqueeze(1)) 
            loss = loss / accumulation_steps # Normalize loss 
            loss.backward() 
            if (i // batch_size) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    # Evaluate the model on the test data
    model.eval()
    test_outputs = model(X_test_torch)
    test_loss = criterion(test_outputs, ytest_torch.unsqueeze(1))

    return test_loss.item()


@st.cache_resource
def train_and_test_lstm(X_train, y_train, X_test, ytest, input_dim, hidden_dim, output_dim, n_layers, dropout, num_epochs, learning_rate):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

    torch.cuda.empty_cache()

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
    ytest_torch = torch.tensor(ytest, dtype=torch.float32).to(device)

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
    rmse_lstm = math.sqrt(mean_squared_error(ytest, test_outputs.cpu().detach().numpy()))
    st.metric("RMSE of the LSTM Model", f"{rmse_lstm:.4f}")

    return model, test_outputs, rmse_lstm


def predict_next_days(model, scaler, last_n_data, n_steps=10,  days_to_predict=7):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    x_input = last_n_data.reshape(1, -1)
    temp_input = x_input.flatten().tolist()  

    lst_output = []
    i = 0
    while i < days_to_predict:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[-n_steps:])  
            x_input = x_input.reshape(1, n_steps, 1) 
            x_input_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)

            with torch.no_grad(): 
                 yhat = model(x_input_tensor)  # Make prediction
                 yhat = yhat.cpu().numpy()
                 temp_input.extend(yhat[0].tolist())
                 lst_output.extend(yhat.tolist())
                 i += 1

    # Inverse transform the predictions
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten().tolist()
    
    return lst_output

def plot_results(history, test, predictions):

    fig = go.Figure()

    # Add history data (in blue)
    fig.add_trace(go.Scatter(x=list(range(len(history))), y=history, mode='lines', name='History', line=dict(color='blue')))

    # Add test data (in black)
    fig.add_trace(go.Scatter(x=list(range(len(history), len(history) + len(test))), y=test, mode='lines', name='Test', line=dict(color='black')))

    # Add predictions for the next 7 days (in red)
    fig.add_trace(go.Scatter(x=list(range(len(history) + len(test), len(history) + len(test) + 7)), y=predictions, mode='lines', name='Predictions', line=dict(color='red')))

    # Update layout
    fig.update_layout(title='History, Test, and Predictions',
                      xaxis_title='Days',
                      yaxis_title='Values',
                      legend=dict(x=0, y=1))

    # Show the plot in Streamlit
    st.plotly_chart(fig)

    # Display next 7-day predictions in Streamlit metric format
    for i, prediction in enumerate(predictions):
        st.metric(label=f'Day {i + 1} Prediction', value=prediction)


st.title("Forecasting with DL Models")

uploaded_data = st.file_uploader("Choose a CSV File", type="csv", key="40")


if uploaded_data is not None:
    #data = load_data(uplodaded_data)
    st.subheader("Choose the algorithm")
    model_name = st.selectbox(
        "Select the DL Model", ["LSTM"]
    )  

    if st.button("Start Forecasting"):
        n = 100 
        df = load_data(uploaded_data)
        df1 = df.reset_index()['Close']
        X_train, y_train, X_test, ytest, scaler = preprocess_data(df1, n)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, ytest), n_trials=50) 
        st.write("Best RMSE:", study.best_value)
        st.write("Best Parameters:", study.best_params)
        best_params = study.best_params
        model, test_outputs, rmse_lstm = train_and_test_lstm(X_train, y_train, X_test, ytest, input_dim=1, hidden_dim=best_params['hidden_dim'], output_dim=1, n_layers=best_params['n_layers'], dropout=best_params['dropout'], num_epochs=5, learning_rate=best_params['learning_rate'])
        st.write(rmse_lstm)
        last_n_data = X_test[len(X_test) - n:]  
        predictions = predict_next_days(model, scaler, last_n_data, n_steps=n, days_to_predict=7)
        print(predictions)
        history = X_train
        test = X_test
        plot_results(history, test, predictions)
            #predictions, rmse, model = train_and_forecast(X, y, model_name, data)
            #st.subheader("Feature Importance")
            #plot_feature_importance(model, X.columns)
            # st.write(next_7_days_pred)
            # forecast_df = pd.DataFrame({'Date': test_index, 'Actual': y_test, 'Forecast': predictions})
        # st.write(forecast_df)
