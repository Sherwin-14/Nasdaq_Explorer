import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
import httpx

st.title("Sentiment Analysis For Stocks")

headers = {
    'x-rapidapi-key': "0a609bffe5msh4a4878f4995debfp1fc4c7jsn8d445fe23afe",
    'x-rapidapi-host': "stock-sentiment-api.p.rapidapi.com"
}

tickers_df = pd.read_csv("/home/sherwin/Projects/time_series_prediction/nasdaq_tickers_cleaned.csv")
stock_dict = dict(zip(tickers_df['Stock Name'], tickers_df['Symbol']))
stock = st.selectbox("Select a stock name", list(stock_dict.keys()), index=None)

if stock is not None:
    if st.button("Get Sentiment"):
        url = f"https://stock-sentiment-api.p.rapidapi.com/avg_sentiment/?ticker={stock_dict[stock]}"

        @st.cache_resource
        def get_stock_sentiment(url):
            response = httpx.get(url, headers=headers)
            data = response.json()
            print(data)
            return data
        
        def extract_and_classify_sentiment(data):
            avg_sentiment = data['avg_sentiment']
            sentiment_classification = None
            
            if avg_sentiment > 0.75:
                sentiment_classification = "Strongly Positive"
                return sac.result(
                    label= sentiment_classification,
                    description = "The overall sentiment of the stock is strongly positive, indicating a high level of enthusiasm and approval",
                    status="success",
                    icon=sac.BsIcon(name="house", size=50, color=None),
                    )
            elif avg_sentiment > 0.25:
                sentiment_classification = "Positive"
                return sac.result(
                    label= sentiment_classification,
                    description = "The overall sentiment of the text is positive, indicating a moderate level of enthusiasm and approval",
                    status="success",
                    icon=sac.BsIcon(name="house", size=50, color=None),
                    )
            elif avg_sentiment < -0.75:
                sentiment_classification = "Strongly Negative"
                return sac.result(
                    label= sentiment_classification,
                    description = "The overall sentiment of the text is strongly negative, indicating a high level of dissatisfaction and disapproval",
                    status="success",
                    icon=sac.BsIcon(name="house", size=50, color=None),
                    )
            elif avg_sentiment < -0.25:
                sentiment_classification = "Negative"
                return sac.result(
                    label= sentiment_classification,
                    description = "The overall sentiment of the text is negative, indicating a moderate level of dissatisfaction and disapproval.",
                    status="success",
                    icon=sac.BsIcon(name="house", size=50, color=None),
                    )
            else:
                sentiment_classification = "Neutral"
                return sac.result(
                    label= sentiment_classification,
                    description = "The overall sentiment of the text is neutral, indicating a lack of strong emotions or opinions.",
                    status="success",
                    icon=sac.BsIcon(name="house", size=50, color=None),
                    )

        data = get_stock_sentiment(url)
        sentiment_classification = extract_and_classify_sentiment(data)


