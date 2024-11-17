import extra_streamlit_components as stx
import pickle
import plotly.graph_objects as go


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
                    st.markdown("""
                    #### Model order is (0,0,0). This may indicate that the data is a simple random walk.
                    Try to collect more data or use a different model selection method.
                    """)  

                    return (p,d,q)

                st.markdown(f""" #### Result : The best arima model for this dataset would have parameters of ({model.order[0]}, {model.order[1]}, {model.order[2]}) """)
               
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
        st.subheader("Upload a file for creation of arima model")
        uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"], key ="2")

        if uplodaded_data is not None:

            df = pd.read_csv(uplodaded_data,parse_dates = ['Date'])
            st.subheader("Data after preprocessing and stationarity check")
            st.dataframe(df.sample(5),use_container_width = True)

            st.warning("Before clicking the button below make sure that you have identified best set of paramerts from previous tab.")

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
                st.subheader("ARIMA Model Forecast and Mean Comparison")
                fig.update_layout(xaxis_title='Date', yaxis_title='Value', legend_title='Legend' )
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=[mean_value]*len(forecast_df['Date']), mode='lines', name='Mean Value', line=dict(color='black', width=2, dash='solid')))
                fig.update_xaxes(showgrid=True) 
                fig.update_yaxes(showgrid=True)  
                st.plotly_chart(fig)

                # Add a download button for the model
                @st.cache_resource
                def get_model_file():
                    model_file = "arima_model.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model_fit, f)
                    return model_file

                model_file = get_model_file()
                st.download_button("Download The Model", model_file, file_name="arima_model.pkl")
            
            

                