import pickle
import plotly.graph_objects as go
import streamlit_antd_components as sac

from app import *
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

st.title("Create an ARIMA Based Model")

tab1, tab2 = st.tabs(["Best Model Parameters", "Optimal Model"])

with tab1:

    st.subheader("Upload a file for creation of arima model")
    uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"])

    if uplodaded_data is not None:
        df = pd.read_csv(uplodaded_data,parse_dates = ['Date'])
        st.subheader("Data after preprocessing and stationarity check")
        st.dataframe(df.sample(5),use_container_width = True)
        
        def run_arima(df):

            if st.button('Get the best model parameters'):

                df['Date'] = pd.to_datetime(df['Date'])
                df = df.dropna()
                train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
                model = auto_arima(train_df['Close'], start_p=1, start_d=1, start_q=1, 
                                    max_p= 4, max_d= 3, max_q = 4, m=12, 
                                    seasonal=True, error_action='warn', 
                                    suppress_warnings=True, stepwise=True)

                # Print the summary of the best model
                print("Best Model Summary:")
                print(model.summary())

                # Get the optimal PDQ values
                p, d, q = model.order

                if p == 0 and q == 0 and d == 0:
                     sac.result(label ='', description = 'This suggests that the data might be white noise or random walk. Please select any other dataset or add more data to current one!',status = 500)
                     return (p,d,q)

                sac.result(

                    label='Sucess',
                    description =f'The best SARIMA model for this dataset would have parameters of ({model.order[0]}, {model.order[1]}, {model.order[2]}) and seasonal order of ({model.seasonal_order[0]}, {model.seasonal_order[1]}, {model.seasonal_order[2]}, {model.seasonal_order[3]})',
                    status='success', icon = sac.BsIcon(name='house', size = 50, color=None)
                )

                st.session_state.pdq_values = (p, d, q)

                return (p,d,q)   
            
            else:
                return (True,True,True)
                    
        p,d,q = run_arima(df)

    if 'pdq_values' not in st.session_state:
        st.session_state.pdq_values = None

    if st.session_state.pdq_values is not None:
        p, d, q = st.session_state.pdq_values    

with tab2:

    if st.session_state.pdq_values is not None:
        p, d, q = st.session_state.pdq_values   
        st.subheader("Upload the same file for modelling")
        uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key ="2")

        if uplodaded_data is not None:

            df = pd.read_csv(uplodaded_data,parse_dates = ['Date'])
            st.subheader("Data after preprocessing and stationarity check")
            st.dataframe(df.sample(5),use_container_width = True)

            st.warning("Before clicking the button below make sure that you have identified best set of parameters from previous tab.")

            if st.button("Start Processing") :

                df['Date'] = pd.to_datetime(df['Date'])
                df = df.dropna()
                df.set_index('Date', inplace=True)

                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                n_splits = 5 
                tscv = TimeSeriesSplit(n_splits=n_splits)

                for fold, (train_index, test_index) in enumerate(tscv.split(df), 1): 
                    train_df, test_df = df.iloc[train_index], df.iloc[test_index] 
                    model = ARIMA(train_df['Close'], order=(1, 0, 0))
                    model_fit = model.fit() # Generate the forecast data 
                    forecast = model_fit.forecast(steps=len(test_df))
                    forecast = forecast.dropna() 
                    test_df = test_df.dropna() 

                final_model = ARIMA(df['Close'], order=(p, d, q)) 
                final_model_fit = final_model.fit()    

                # Generate the forecast data
                forecast = final_model_fit.forecast(steps=30)

                # Create a DataFrame to display the forecast data and metrics
                forecast_df = pd.DataFrame({'Date': pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=30), 'Forecast': forecast})

                st.subheader("Forecast Data For Next 30 Days")
                # Display the forecast data and metrics
                st.dataframe(forecast_df.sample(10).sort_values(by = ['Date'],ascending = True).style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}]).background_gradient(cmap='Blues'),use_container_width = True)

                mean_value = forecast_df['Forecast'].mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecasted Data', line=dict(color='blue', dash='dash'))) # Set the title and labels 
                st.subheader("ARIMA Model Forecast for Next 30 Days and Mean Comparison")
                fig.update_layout(xaxis_title='Date', yaxis_title='', legend_title='Legend', showlegend=True) # Hide y-axis title
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=[mean_value]*len(forecast_df['Date']), mode='lines', name='Mean Value', line=dict(color='black', width=2, dash='solid')))
                fig.update_xaxes(showgrid=False) # Remove x-axis gridlines
                fig.update_yaxes(showgrid=False, showticklabels=False) # Remove y-axis gridlines and tick labels
                st.plotly_chart(fig)

                # Get the last 30 days of data for comparison
                test_values = df['Close'].iloc[-30:]

                # Create a figure to display the comparison graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_values.index, y=test_values, mode='lines', name='Actual Values', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=test_values.index, y=final_model_fit.forecast(steps=30), mode='lines', name='Predicted Values', line=dict(color='blue', dash='dash')))
                fig.update_layout(xaxis_title='Date', yaxis_title='Value', legend_title='Legend' )
                fig.update_xaxes(showgrid=True) 
                fig.update_yaxes(showgrid=True)  
                st.subheader("ARIMA Model Actual vs Predicted Values")
                st.plotly_chart(fig)

                # Calculate the metrics
                actual_values = df['Close'].iloc[-30:]
                predicted_values = final_model_fit.forecast(steps=30)

                actual_values = np.where(actual_values == 0, 1e-10, actual_values)
                actual_values = np.where(np.abs(actual_values) < 1e-10, 1e-10, actual_values)

                # Calculate the metrics
                mae = np.mean(np.abs(actual_values - predicted_values))
                mse = np.mean((actual_values - predicted_values)**2)
                rmse = np.sqrt(mse)
                rmspe = np.sqrt(np.mean((actual_values - predicted_values) / (np.abs(actual_values) + 1e-10))**2)

                # Create a DataFrame of the metrics
                metrics_df = pd.DataFrame({
                     'Metrics': ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Root Mean Squared Percentage Error'],
                     'Value': [mae, mse, rmse, rmspe]
                })

                st.subheader("ARIMA Model Performance Metrics")
                st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}).background_gradient(cmap='OrRd'),use_container_width = True)

                @st.cache_resource
                def get_model_file():
                    model_file = "arima_model.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model_fit, f)
                    return model_file

                model_file = get_model_file()
                st.download_button("Download The Model", model_file, file_name="arima_model.pkl")


                