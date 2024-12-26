import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


from app import *


st.title("Forecasting with DL Models")

uploaded_data = st.file_uploader("Choose a CSV File", type="csv", key="40")

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
