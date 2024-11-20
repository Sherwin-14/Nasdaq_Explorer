import statsmodels.api as sm

from app import *
from datetime import date
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Choose Stock Data To Analyze ")

tab1, tab2, tab3, tab4 = st.tabs(["Basic Data Exploration", "Time Series Decomposition", "Stationarity Tests" ,"ACF and PACF Plots" ])

with tab1:
    
    tickers_df = pd.read_csv('/home/sherwin/Projects/time_series_prediction/nasdaq_tickers.csv')

    stock_tickers = tickers_df['Symbol'].tolist()

    # Create a selectbox to select a stock ticker symbol
    selected_ticker = st.selectbox("Select a stock ticker symbol", stock_tickers)

    # Display the selected stock ticker symbol
    st.write("Selected stock ticker symbol :", selected_ticker)

    @st.cache_resource
    def fetch_and_display_data(selected_ticker):
        # Fetch data from Yahoo Finance
        data = yf.download(selected_ticker, start="2019-01-01", end= date.today().strftime("%Y-%m-%d"))

        # Create a dataframe with column names
        df = pd.DataFrame(data)
        df = df.reset_index()
        df.columns = ['Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        df.rename(columns = {'index': 'Date'}, inplace=True)

        print(df)

        # Convert index to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        st.session_state.df = df

        return df
    
    @st.cache_resource
    def validate_data(df):
        status_messages = [] 

        # Check for null values and fill them if necessary 
        if df.isnull().values.any(): 
            status_messages.append("Null values detected and filled using forward fill.")
            df['Close'] = df['Close'].ffill()
        else: 
            status_messages.append("No null values detected.")

        if df.duplicated().any():
            status_messages.append("Duplicate rows detected and removed.")
            df = df.drop_duplicates()
        else:
            status_messages.append("No duplicate rows detected.")

        expected_dtypes = {   
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Adj Close': 'float64',
        'Volume': 'float64'
        }

        incorrect_dtypes = []
        for col, dtype in expected_dtypes.items():
            if df[col].dtype != dtype:
                incorrect_dtypes.append(f"Column '{col}' has incorrect data type. Expected {dtype} but got {df[col].dtype}.")

        if incorrect_dtypes:
            status_messages.extend(incorrect_dtypes)
        else: 
            status_messages.append("All columns have the correct data types.")

        # Check date range
        expected_start_date = pd.to_datetime("2019-01-01") 
        expected_end_date = pd.to_datetime("2022-02-26") 
        if df.index.min() != expected_start_date or df.index.max() != expected_end_date: 
            status_messages.append(f"Date range does not match expected values.")
        else: 
            status_messages.append("Date range matches expected values.")
        
        # Check for missing dates 
        full_date_range = pd.date_range(start=expected_start_date, end=expected_end_date) 
        missing_dates = full_date_range.difference(df.index) 
        if not missing_dates.empty: 
            status_messages.append(f"Date range has missing entries.")
        else: 
            status_messages.append("No missing dates detected.")
        
        # Concatenate all status messages into a single string 
        validation_summary = "\n".join(status_messages) 
        
        # Display all status messages in a success prompt 
        with st.container(): 
            st.subheader("Data Validation Summary") 
            st.success(validation_summary)

        return df
    
    @st.cache_resource   
    def display_summary_and_graph(df):
        # Display summary statistics
        with st.container():
            st.subheader("Data Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        # Display graph with x-range slider
        with st.container():
            st.subheader("Comparing The Prices Up Untill Now")
            df = df.reset_index()
            df_long = df.melt(id_vars = 'Date', value_vars=['Open', 'High', 'Low', 'Close'], var_name = 'Variable', value_name='Value')
            fig = px.line(df_long, x = 'Date', y='Value', color='Variable')
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)             


    # Call the fetch_and_display_data function and get the dataframe
    df = fetch_and_display_data(selected_ticker)
    df = validate_data(df)
    display_summary_and_graph(df)

    if 'df' not in st.session_state:
        st.session_state.df = df

with tab2:
    
    st.subheader(f'Time Series Decomposition')
    values = ["Additive","Multiplicative"]
    ts_decompose_model = st.selectbox("Select a Decomposition Method which is appropriate for the data",values)

    if ts_decompose_model == "Additive":
        st.session_state.df['Close'] = st.session_state.df['Close'].dropna()
        decomposition = seasonal_decompose(st.session_state.df['Close'],  model ='additive', period=30)
    else:
        st.session_state.df['Close'] = st.session_state.df['Close'].dropna()
        decomposition = seasonal_decompose(st.session_state.df['Close'], model ='multiplicative', period=30)

    T, S, R = decomposition.trend, decomposition.seasonal, decomposition.resid

    # Create a DataFrame with the decomposed components 
    decomposed_df = pd.DataFrame({ 'Date': st.session_state.df['Date'], 'Trend': T, 'Seasonal': S, 'Residual': R }).dropna()

    with st.container():
        st.subheader('Trend')
        st.line_chart(T)

        st.subheader('Seasonality')
        st.line_chart(S)

        st.subheader('Residual')
        st.line_chart(R)  

    csv = decomposed_df.to_csv(index=False) 
    st.download_button("Download Decomposed Data as CSV", csv, "decomposed_data.csv")    

with tab3:

    st.subheader("Stationarity Tests")
    
    def check_stationarity(df, method='ADF'):
         if method == 'ADF':
            clean_df = df[['Date','Close']].dropna()
            original_dates = clean_df['Date']
            result = adfuller(clean_df['Close'],autolag='AIC')
            statistic, p_value, used_lag, n_obs, critical_values, icbest = result
            diff_count = 0
            print(p_value)
            while p_value > 0.05:
                st.warning(f'The series is not stationary after {diff_count} differences. Making it stationary...')
                clean_df['Close'] = clean_df['Close'].diff()
                print(clean_df.isna().sum())
                clean_df = clean_df.dropna()
                print(clean_df.isna().sum())
                result = adfuller(clean_df['Close'], autolag = 'AIC')
                statistic, p_value, used_lag, n_obs, critical_values, icbest = result
                diff_count += 1

            st.success(f'The series is now stationary after {diff_count} differences.')

            result_summary = pd.DataFrame({
                'Test': ['ADF Statistic'],
                'Value': [statistic],
                'p-value': [p_value],
                'Used Lag': [used_lag],
                'Number of Observations': [n_obs],
                'Critical Values': [critical_values],
                'IC Best': [icbest]
            })

            st.write(result_summary)

            print(clean_df.shape)

            stationary_df = pd.DataFrame({ 'Date': original_dates, 'Close': clean_df['Close'] }) 

            print(stationary_df)

            # Download the stationary data
            st.download_button(
                label="Download Stationary Data",
                data=stationary_df.to_csv(index=False),
                file_name="stationary_data.csv",
                mime="text/csv",
            )

    check_stationarity(st.session_state.df) 

    uplodaded_data = st.file_uploader("Choose a CSV file", type=["csv"],key = "2")

    if uplodaded_data is not None:
        df = pd.read_csv(uplodaded_data,parse_dates = ['Date'])
        st.subheader("Data after preprocessing and stationarity check")
        st.dataframe(df.sample(5),use_container_width = True)

        # Function to check stationarity of residuals
        def check_residual_stationarity(residual_series,dates):
            clean_series = residual_series.dropna() 
            result = adfuller(clean_series, autolag='AIC') 
            statistic, p_value, used_lag, n_obs, critical_values, icbest = result 
            diff_count = 0

            while p_value > 0.05: 
                st.warning(f'The residuals are not stationary after {diff_count} differences. Making them stationary...') 
                clean_series = clean_series.diff().dropna() 
                result = adfuller(clean_series, autolag='AIC') 
                statistic, p_value, used_lag, n_obs, critical_values, icbest = result 
                diff_count += 1 
            st.success(f'The residuals are now stationary after {diff_count} differences.')

            result_summary = pd.DataFrame({ 'Test': ['ADF Statistic'], 'Value': [statistic], 'p-value': [p_value], 'Used Lag': [used_lag], 'Number of Observations': [n_obs], 'Critical Values': [critical_values], 'IC Best': [icbest] }) 
            st.write(result_summary) 
            stationary_residuals = pd.Series(clean_series).dropna() 
            return stationary_residuals

        st.subheader("Stationarity Check for Residuals") 
        stationary_residuals = check_residual_stationarity(decomposed_df['Residual'], decomposed_df['Date'])
        if stationary_residuals is not None: 
            st.download_button("Download Stationary Residuals Data", data=stationary_residuals.to_csv(index=False), file_name="stationary_residuals_data.csv", mime="text/csv")


     
with tab4:

    st.subheader('Autoregression Plots')

    lags = st.slider('Choose the number of lags', min_value=1, max_value=50, value=20)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.dropna()
        temp_session_state = df
        st.write(df)

        # Generate ACF and PACF plots
        st.subheader('ACF and PACF Plots')
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(temp_session_state['Close'], ax=axes[0], lags=lags)
        plot_pacf(temp_session_state['Close'], ax=axes[1], lags=lags)
        st.pyplot(fig)

    


