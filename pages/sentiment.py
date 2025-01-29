import pandas as pd
import streamlit as st
import httpx

from intro import tickers_df

url = "https://stock-sentiment-api.p.rapidapi.com/avg_sentiment/?ticker={stock}"

headers = {
    'x-rapidapi-key': "0a609bffe5msh4a4878f4995debfp1fc4c7jsn8d445fe23afe",
    'x-rapidapi-host': "stock-sentiment-api.p.rapidapi.com"
}

stock_dict = dict(zip(tickers_df['Stock Name'], tickers_df['Symbol']))

stock = st.selectbox("Select a stock name", list(stock_dict.keys()))

def get_stock_sentiment(stock):
    response = httpx.get(url, headers=headers)
    return response.json()

def extract_and_classify_sentiment(data):
    avg_sentiment = data['avg_sentiment']
    
    if avg_sentiment > 0.5:
        sentiment_classification = "Positive"
    elif avg_sentiment < -0.5:
        sentiment_classification = "Negative"
    else:
        sentiment_classification = "Neutral"
    
    return avg_sentiment, sentiment_classification

stock_sentiment = get_stock_sentiment()
print(stock_sentiment)