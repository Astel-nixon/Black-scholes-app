import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from black_scholes import black_scholes_call, black_scholes_put, black_scholes_greeks
from monte_carlo import monte_carlo_call, monte_carlo_simulation
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
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='section-title'>Option Pricing Models</h1>", unsafe_allow_html=True)

# ---- LinkedIn Connect Button ----
st.sidebar.markdown("## 🤝 Connect with me on LinkedIn")
st.sidebar.markdown("""
[![Connect on LinkedIn](https://img.shields.io/badge/Connect%20on%20LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/astelnixon/)
""", unsafe_allow_html=True)

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

# ---- Compute Option Prices Using Different Models ----
st.subheader("📈 Option Pricing Results")

binomial_price = binomial_tree_call(S, K, T, r, sigma)
monte_carlo_price = monte_carlo_call(S, K, T, r, sigma)
black_scholes_call_price = black_scholes_call(S, K, T, r, sigma)
black_scholes_put_price = black_scholes_put(S, K, T, r, sigma)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="📊 Black-Scholes Call Price", value=f"${black_scholes_call_price:.2f}")
    st.metric(label="📊 Black-Scholes Put Price", value=f"${black_scholes_put_price:.2f}")

with col2:
    st.metric(label="📊 Binomial Tree Call Price", value=f"${binomial_price:.2f}")

with col3:
    st.metric(label="📊 Monte Carlo Call Price", value=f"${monte_carlo_price:.2f}")

# ---- Interactive Heatmaps for Call and Put Prices (Only One Instance) ----
st.subheader("📊 Call & Put Price Heatmaps")

min_spot = st.sidebar.number_input("Min Spot Price", value=80.0, key="min_spot")
max_spot = st.sidebar.number_input("Max Spot Price", value=120.0, key="max_spot")
min_volatility = st.sidebar.slider("Min Volatility for Heatmap", 0.01, 1.00, 0.10, key="min_volatility")
max_volatility = st.sidebar.slider("Max Volatility for Heatmap", 0.01, 1.00, 0.30, key="max_volatility")

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

# ---- Monte Carlo Simulation ----
st.subheader("📊 Monte Carlo Stock Price Simulation")

num_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=50000, value=10000)
num_steps = st.sidebar.slider("Time Steps", min_value=10, max_value=365, value=252)

simulated_paths = monte_carlo_simulation(S, T, r, sigma, num_simulations=num_simulations, num_steps=num_steps)

fig, ax = plt.subplots()
for i in range(10):  # Plot only 10 sample paths for better visibility
    ax.plot(simulated_paths[:, i], alpha=0.5)

ax.set_title("Monte Carlo Simulated Stock Price Paths")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Stock Price")
st.pyplot(fig)

# ---- Profit/Loss Simulation ----
st.subheader("📈 Profit/Loss Simulation")
stock_range = np.linspace(S * 0.5, S * 1.5, 100)
call_payoff = np.maximum(stock_range - K, 0)
put_payoff = np.maximum(K - stock_range, 0)
fig, ax = plt.subplots()
ax.plot(stock_range, call_payoff, label="Call Option", color="green")
ax.plot(stock_range, put_payoff, label="Put Option", color="red")
ax.axhline(0, color='black', linestyle='--')
ax.set_title("Option Profit/Loss at Expiration")
ax.set_xlabel("Stock Price at Expiration")
ax.set_ylabel("Profit/Loss")
ax.legend()
st.pyplot(fig)