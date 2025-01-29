import pandas as pd
import streamlit as st
import httpx

st.subheader("Sentiment Analysis For Stocks")

headers = {
    'x-rapidapi-key': "0a609bffe5msh4a4878f4995debfp1fc4c7jsn8d445fe23afe",
    'x-rapidapi-host': "stock-sentiment-api.p.rapidapi.com"
}

tickers_df = pd.read_csv("/home/sherwin/Projects/time_series_prediction/nasdaq_tickers_cleaned.csv")
stock_dict = dict(zip(tickers_df['Stock Name'], tickers_df['Symbol']))
stock = st.selectbox("Select a stock name", list(stock_dict.keys()))

url = f"https://stock-sentiment-api.p.rapidapi.com/avg_sentiment/?ticker={stock_dict[stock]}"

def get_stock_sentiment(stock):
    response = httpx.get(url, headers=headers)
    data = response.json()
    return data

@st.cache_resource
def extract_and_classify_sentiment(data):
    avg_sentiment = data['avg_sentiment']
    sentiment_classification = None
    
    if avg_sentiment > 0.5:
        sentiment_classification = "Positive"
    elif avg_sentiment < -0.5:
        sentiment_classification = "Negative"
    else:
        sentiment_classification = "Neutral"
    
    return sentiment_classification


data = get_stock_sentiment(stock)
sentiment_classification = extract_and_classify_sentiment(data)