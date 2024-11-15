import statsmodels.api as sm

from app import *
from datetime import date
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

st.title("Choose Stock Data To Analyze ")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Data Exploration", "Time Series Decomposition", "Exponential Smoothing","ACF and PACF Plots", "Stationarity Tests"])

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
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)

        return df

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
        'Volume': 'int64'
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
        
    def display_summary_and_graph(df):
        # Display summary statistics
        with st.container():
            st.subheader("Data Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

        # Display graph with x-range slider
        with st.container():
            st.subheader("Comparing The Prices Up Untill Now")
            fig = px.line(df, x=df.index, y=['Open', 'High', 'Close','Low'])
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

    with st.container():
        st.subheader('Trend')
        st.line_chart(T)

        st.subheader('Seasonality')
        st.line_chart(S)

        st.subheader('Residual')
        st.line_chart(R)  

with tab3:

    smoothing_type = st.multiselect(':orange[Chose Smoothing Type]',options=['Single', 'Double', 'Triple'], default=['Single'])
     
with tab4:

    st.subheader('Autoregression Plots')

    lags = st.slider('Choose the number of lags', min_value=1, max_value=50, value=20)

    show_arplots = st.toggle("Show ACF and PACF plots")

    if show_arplots:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(st.session_state.df['Close'], ax=axes[0], lags=lags)
        plot_pacf(st.session_state.df['Close'], ax=axes[1], lags=lags)
        st.pyplot(fig)


with tab5:

    st.subheader('Stationarity Tests') 


