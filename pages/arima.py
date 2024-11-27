import pickle
import plotly.graph_objects as go
import streamlit_antd_components as sac
import matplotlib.pyplot as plt


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
                                    max_p= 5, max_d= 5, max_q = 5, m=12, 
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

            df = pd.read_csv(uplodaded_data, parse_dates = ['Date'])
            st.subheader("Data after preprocessing and stationarity check")
            st.dataframe(df.sample(5),use_container_width = True)

            st.warning("Before clicking the button below make sure that you have identified best set of parameters from previous tab.")

            if st.button("Start Processing") :

                df['Date'] = pd.to_datetime(df['Date'])
                df = df.dropna()
                df.set_index('Date', inplace=True)

                print(df.isna().sum())

                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

                train_size = int(len(df) * 0.8)
                train_df = df[:train_size]
                test_df = df[train_size:]

                # Create and fit the ARIMA model on the training data
                final_model = ARIMA(train_df['Close'], order=(p,d, q)) 
                final_model_fit = final_model.fit() 

                st.write(final_model_fit.summary())

                start_index = test_df.index[0]
                end_index = test_df.index[-1]

                start_index = train_size
                end_index = train_size + len(test_df) - 1
                predicted_close = final_model_fit.predict(start=start_index, end=end_index) 

                predicted_close = predicted_close.reset_index(drop=True)

                # Forecast the next 30 days
                forecast_diff = final_model_fit.get_forecast(steps=30)
                forecast = forecast_diff.predicted_mean
                conf_int_95 = forecast_diff.conf_int(alpha=0.05)
                conf_int_90 = forecast_diff.conf_int(alpha=0.10)

                forecast_df = pd.DataFrame({ 'Date': pd.date_range(start = df.index[-1] + pd.DateOffset(days=1), periods=30), 'Forecast': forecast })
                test_df.reset_index(inplace=True) 
                comparison_df = pd.DataFrame({ 'Date': test_df['Date'], 'Actual': test_df['Close'], 'Predicted': predicted_close})

                print(comparison_df)
                print(forecast_df.dtypes)

                st.subheader("Diagnostics Plots for ARIMA")
                st.pyplot(final_model_fit.plot_diagnostics(figsize=(15,7)))

                residuals = comparison_df['Actual'] - comparison_df['Predicted']

                if not residuals.isna().any():

                    plt.figure(figsize=(15,5))
                    st.subheader("Distribution of Residuals ")
                    sns.kdeplot(residuals, shade=True)
                    plt.axvline(residuals.mean(), color='red', linestyle='--', label='Mean')
                    plt.axvline(residuals.median(), color='green', linestyle='--', label='Median')
                    plt.legend()
                    plt.title('KDE plot for Residuals')
                    plt.xlabel('Residuals')
                    plt.ylabel('Density')
                    st.pyplot(plt)
                else:
                     print("Residuals contain NaN values. Please check the data.")

                st.subheader("Forecast Data For Next 30 Days")
                # Display the forecast data and metrics
                st.dataframe(forecast_df.sample(10).sort_values(by = ['Date'],ascending = True).style.set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}]).background_gradient(cmap='Blues'),use_container_width = True)

                mean_value = forecast_df['Forecast'].mean()

                # Create a figure to display the comparison graph
                fig = go.Figure() 
                fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Actual'], mode='lines', name='Actual Values', line=dict(color='green', dash='dash'))) 
                fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Predicted'], mode='lines', name='Predicted Values', line=dict(color='blue', dash='dash')))
                fig.update_layout( xaxis_title='Date', yaxis_title='Value', legend_title='Legend', showlegend=True, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                st.subheader("ARIMA Model Actual vs Predicted Values on Test Data") 
                st.plotly_chart(fig)


                # Calculate the metrics
                actual_values = comparison_df['Actual'].values 
                predicted_values = comparison_df['Predicted'].values
                
                actual_values = np.where(actual_values == 0, 1e-10, actual_values)
                actual_values = np.where(np.abs(actual_values) < 1e-10, 1e-10, actual_values)

                predicted_values = np.where(predicted_values == 0, 1e-10, predicted_values)
                predicted_values = np.where(np.abs(predicted_values) < 1e-10, 1e-10, predicted_values)

                
                mae = np.mean(np.abs(actual_values - predicted_values)) 
                mse = np.mean((actual_values - predicted_values)**2) 
                rmse = np.sqrt(mse) 
                rmspe = np.sqrt(np.mean((actual_values - predicted_values) / (np.abs(actual_values) + 1e-10))**2) 
                metrics_df = pd.DataFrame({ 'Metrics': ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'Root Mean Squared Percentage Error'], 'Value': [mae, mse, rmse, rmspe] })

                st.subheader("ARIMA Model Performance Metrics")
                st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}).background_gradient(cmap='OrRd'),use_container_width = True)

                fig4 = plt.figure(figsize=(12,6))
                ax = fig4.add_subplot(111)

                conf_int_95.index = forecast.index
                conf_int_90.index = forecast.index


                ax.set_title('ARIMA model predictions')

                ax.plot(train_df.index, train_df['Close'], '--b', label='Training Data')
                ax.plot(test_df.index, test_df['Close'], '--', color='gray', label='Test Data')
                ax.plot(forecast.index, forecast, '--', color='red', label='Next 30 Day Forecasts')

                ax.fill_between(
                forecast.index,
                conf_int_95.iloc[:, 0],
                conf_int_95.iloc[:, 1],
                color="b",
                alpha=0.1,
                label="95% CI"

                )

                ax.fill_between(

                forecast.index,
                conf_int_90.iloc[:, 0],
                conf_int_90.iloc[:, 1],
                color="b",
                alpha=0.2,
                label="90% CI"

                )

                ax.set_xlim(pd.to_datetime(train_df.index[0]), pd.to_datetime(max(forecast.index)))

                ax.legend(loc="upper left")

                st.pyplot(fig4)

                @st.cache_resource
                def get_model_file():
                    model_file = "arima_model.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(final_model_fit, f)
                    return model_file
                
                model_file = get_model_file()
                st.download_button("Download The Model", model_file, file_name="arima_model.pkl")


                