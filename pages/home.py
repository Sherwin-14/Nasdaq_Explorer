from app import *

# Title and header
st.title("Nasdaq Explorer")
st.header("Analyze and Forecast Time Series Data")

# Introduction text
st.write("""
Nasdaq Explorer is a powerful tool for analyzing and forecasting time series data. With our app, you can easily explore your dataset, identify trends and patterns, and build accurate models to predict future values.
""")

# Key features section
st.subheader("Key Features:")
st.write("""
* **Multiple Modeling Options**: Choose from a range of models, including ARIMA, SARIMA, and Deep Learning models, to find the best fit for your data.
* **Prophet Modeling**: Use Facebook's Prophet library to build robust and accurate models for your time series data.
* **Data Exploration**: Visualize your data and gain insights into trends, seasonality, and anomalies.
* **Stock Data Analysis**: Insert your stock data and let our app help you identify the best possible model for forecasting future values.
* **Model Comparison**: Try out different models and compare their performance to find the best one for your needs.
""")

# How it works section
st.subheader("How it Works:")
st.write("""
1. Upload your time series data or insert your stock data.
2. Explore your data and identify trends and patterns.
3. Choose from a range of models to build and train.
4. Compare model performance and select the best one.
5. Use our app to forecast future values and make informed decisions.
""")
# Call to action button
st.button("Get Started")

# Optional: Add a link to a tutorial or documentation
st.write(
    "Need help getting started? Check out our [tutorial](https://example.com/tutorial) or [documentation](https://sherwin-14.github.io/Nasdaq_Explorer/)"
)
