import pickle
import plotly.graph_objects as go
import streamlit_antd_components as sac
import matplotlib.pyplot as plt
import math


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
                print(df)
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
        uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key ="3")

        if uplodaded_data is not None:

            df = pd.read_csv(uplodaded_data, parse_dates = ['Date'])
            st.subheader("Data after preprocessing and stationarity check")
            st.dataframe(df.sample(5),use_container_width = True)
            st.warning("Before clicking the button below make sure that you have identified best set of parameters from previous tab.")

            if st.button("Start Processing") :

                df['Date'] = pd.to_datetime(df['Date'])
                df = df.dropna()
                df.set_index('Date', inplace=True)

                print(df.tail(7))

                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

                train_size = int(len(df) * 0.8)
                train = df[:train_size]
                test = df[train_size:]

                def arima_modelling(train,test):
                    history = [x for x in train]
                    history = pd.Series(train.squeeze()).astype('float32')
                    history = history.dropna()
                    history = np.asarray(history, dtype = np.float32)
                    predictions = [x for x in train]
                    onlypreds = []

                    for t in range(len(test)+7):
                        model = ARIMA(history, order = (p,d ,q))
                        model = model.fit()
                        output = model.forecast()
                        output = pd.DataFrame(output)
                        yhat = output[0]
                        predictions.append(yhat[0])
                        onlypreds.append(yhat[0])
                        if t < len(test):
                            obs = test.iloc[t]
                            history = np.append(history, obs)
                        else:
                            obs = yhat[0]
                            history = np.append(history, obs)

                    return predictions, onlypreds
                
                preds, onlypreds  = arima_modelling(train, test)

                print(onlypreds)

                error_arima = math.sqrt(mean_squared_error(test, onlypreds[0:len(test)]))

                print(error_arima)

                x = np.append(train, onlypreds)
                pre = pd.DataFrame(x,columns = ["ARIMA"])
                df = df.reset_index()
                pre = pd.concat([pre,df['Close']], axis = 1)
                pre = pd.concat([pre,df['Date']],axis = 1)

                last_date = df['Date'].iloc[-1]

                print(last_date)

                new_dates = pd.date_range(last_date, periods = 7 , freq='D')
                pre.apply(lambda col: df['Date'].drop_duplicates().reset_index(drop=True))

                print(pre.tail(7))

                new_df = pd.DataFrame({'Date': new_dates, 'Close': pre['ARIMA'].tail(7)})

                # Create a new dataframe that contains both the history and predicted values
                combined_df = pd.concat([df, new_df], ignore_index=True)

                print(combined_df)

                colors = ['blue' if i < len(df) else 'red' for i in range(len(combined_df))]

                st.subheader("ARIMA Model Forecast")

                fig = go.Figure(data=[
                    go.Scatter(x=combined_df['Date'][:-7], y=combined_df['Close'][:-7], mode='lines', line=dict(color='blue'), name='History' ),
                    go.Scatter(x=combined_df['Date'][-7:], y=combined_df['Close'][-7:], mode='lines', line=dict(color='red'), name='Prediction'),
                    go.Scatter(x=combined_df['Date'][-7:], y=combined_df['Close'][-7:], mode='markers', marker=dict(size=8, color='red'), name='Predicted Points')
                ])

                fig.update_layout(
                    title='History and Predicted Values',
                    xaxis_title='Date',
                    yaxis_title='Stock Price',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    width = 1000,
                    height = 600,
                    xaxis=dict(showgrid=False),  # Remove gridlines from the x-axis
                    yaxis=dict(showgrid=False)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display the RMSE of the model \
                col1, col2 , col3 = st.columns(3)
                with col2:
                    st.subheader("Model Performance") 
                    st.metric(label="RMSE", value=f"{error_arima:.2f}")
                
                with col1:
                    st.subheader("Next 7-Day Predictions") 
                    for i in range(7): 
                        st.metric(label=f"Day {i+1}", value=f"{onlypreds[len(test) + i]:.7f}")

