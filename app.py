import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from black_scholes import black_scholes_call, black_scholes_put, black_scholes_greeks
from monte_carlo import monte_carlo_call
from binomial_tree import binomial_tree_call
import requests
import matplotlib.pyplot as plt
import scipy.stats as si
import time

# Set Page Configuration
st.set_page_config(page_title="Black-Scholes Model", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .stApp { background-color: #0f0f0f; color: white; }
        .stSidebar { background-color: #1e1e1e; }
        .metric-card { background-color: #222222; padding: 15px; border-radius: 10px; text-align: center; }
        .stButton>button { width: 100%; border-radius: 10px; background-color: #333333; color: white; }
        .stSlider { background-color: #282828 !important; }
        .stAlert { background-color: #222222 !important; border-radius: 10px; }
        .section-title { text-align: center; font-size: 26px; color: white; }
        
        /* Fix for white text on white background */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stNumberInput input {
            color: black !important;
            background-color: white !important;
            border-radius: 5px !important;
        }

        /* Centering LinkedIn elements */
        .linkedin-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='section-title'>Black-Scholes Pricing Model</h1>", unsafe_allow_html=True)

# ---- LinkedIn Integration ----
st.subheader("👥 Connect with Me")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <a href="https://www.linkedin.com/in/astelnixon/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" height="40">
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("[🔗 Connect with me on LinkedIn](https://www.linkedin.com/in/astelnixon/)")

with col3:
    linkedin_share_url = "https://www.linkedin.com/shareArticle?mini=true&url=https://your-app-link.streamlit.app/"
    st.markdown(f"[📢 Share this App on LinkedIn](<{linkedin_share_url}>)", unsafe_allow_html=True)

# ---- Fetch Live Top Gainers ----
def get_live_top_gainers():
    url = "https://query1.finance.yahoo.com/v7/finance/spark?symbols=AAPL,MSFT,TSLA,NVDA,GOOGL,AMZN,META,BABA,AMD"
    try:
        response = requests.get(url).json()
        stocks = []
        for ticker, data in response['spark']['result'].items():
            if 'response' in data and data['response']:
                latest_price = data['response'][0]['meta']['regularMarketPrice']
                stocks.append((ticker, latest_price))
        return sorted(stocks, key=lambda x: x[1], reverse=True)[:4]  # Top 4 gainers
    except:
        return [("AAPL", 170), ("MSFT", 350), ("TSLA", 180), ("NVDA", 450)]  # Default values

# Display Live Top Stocks
st.subheader("🔥 Live Top 4 Stocks of the Week")
top_stocks = get_live_top_gainers()
col1, col2, col3, col4 = st.columns(4)

for i, (stock, price) in enumerate(top_stocks):
    with [col1, col2, col3, col4][i]:
        st.metric(label=f"📈 {stock}", value=f"${price:.2f}")

# ---- Sidebar Inputs ----
st.sidebar.header("🔢 Input Parameters")
ticker = st.sidebar.text_input("📉 Enter Stock Ticker", value="AAPL", key="ticker")

# Fetch Current Price
def fetch_stock_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return data["Close"].iloc[-1]
    except:
        return 150.0  # Default value

current_price = fetch_stock_price(ticker)
st.sidebar.success(f"✅ {ticker.upper()} Price: ${current_price:.2f}")

S = st.sidebar.slider("Underlying Price (S)", min_value=50.0, max_value=500.0, value=float(current_price))
K = st.sidebar.slider("Strike Price (K)", min_value=50.0, max_value=500.0, value=150.0)
T = st.sidebar.slider("Time to Maturity (Years)", min_value=0.1, max_value=5.0, value=1.0)
r_percent = st.sidebar.slider("Risk-free Rate (%)", min_value=0.1, max_value=10.0, value=5.0)
sigma_percent = st.sidebar.slider("Volatility (%)", min_value=5.0, max_value=100.0, value=20.0)

r = r_percent / 100  # Convert to decimal
sigma = sigma_percent / 100  # Convert to decimal

# ---- Interactive Heatmaps for Call and Put Prices ----
st.subheader("📊 Call & Put Price Heatmaps")

min_spot = st.sidebar.number_input("Min Spot Price", value=80.0, key="min_spot")
max_spot = st.sidebar.number_input("Max Spot Price", value=120.0, key="max_spot")
min_volatility = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.00, 0.10)
max_volatility = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.00, 0.30)

spot_prices = np.linspace(min_spot, max_spot, 10)
volatilities = np.linspace(min_volatility, max_volatility, 10)
call_matrix = np.zeros((10, 10))
put_matrix = np.zeros((10, 10))

for i, sigma_val in enumerate(volatilities):
    for j, S_val in enumerate(spot_prices):
        call_matrix[i, j] = black_scholes_call(S_val, K, T, r, sigma_val)
        put_matrix[i, j] = black_scholes_put(S_val, K, T, r, sigma_val)

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(data=go.Heatmap(
        z=call_matrix,
        x=spot_prices,
        y=volatilities,
        colorscale="Viridis",
        text=[[f"{call_matrix[i, j]:.2f}" for j in range(10)] for i in range(10)],
        hoverinfo="text"
    ))
    fig.update_layout(title="Call Price Heatmap", xaxis_title="Spot Price", yaxis_title="Volatility")
    st.plotly_chart(fig)

with col2:
    fig = go.Figure(data=go.Heatmap(
        z=put_matrix,
        x=spot_prices,
        y=volatilities,
        colorscale="Viridis",
        text=[[f"{put_matrix[i, j]:.2f}" for j in range(10)] for i in range(10)],
        hoverinfo="text"
    ))
    fig.update_layout(title="Put Price Heatmap", xaxis_title="Spot Price", yaxis_title="Volatility")
    st.plotly_chart(fig)