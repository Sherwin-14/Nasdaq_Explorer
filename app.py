import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf


from datetime import date


pages = {
    "Home": [
        st.Page("pages/home.py", title="Home", icon="🔥"),
        st.Page("pages/intro.py", title="Choose Stock to analyze", icon="🤔"),
    ],
    "Resources": [
        st.Page("pages/arima.py", title="ARIMA Modelling", icon="📊"),
        st.Page("pages/prophet.py", title="Facebook Prophet Modelling", icon="🛍️"),
        st.Page("pages/sarima.py", title="SARIMA Modelling", icon="💸"),
        st.Page("pages/ml_models.py", title="Machine Learning Models", icon="⏰"),
        st.Page("pages/dl_models.py", title="Deep Learning Models", icon="📋"),
        # st.Page("pages/suggested_items_analysis.py",title="Requested Items Analysis",icon="📋"),
        # st.Page("pages/rfm_model.py", title="Customer Segmentation Model", icon="👥"),
    ]
}

pg = st.navigation(pages)
pg.run()
