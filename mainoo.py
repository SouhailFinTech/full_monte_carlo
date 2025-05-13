# main.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")
st.title("📈 Monte Carlo Stock Price Simulator")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", value="AAPL,TSLA,BTC-USD")
time_horizon = st.sidebar.slider("Time Horizon (Days)", min_value=1, max_value=252*5, value=250)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=10000, value=1000)
portfolio_mode = st.sidebar.checkbox("Portfolio Mode")
run_simulation = st.sidebar.button("Run Simulation")

# Convert comma-separated string to list
ticker_list = [t.strip() for t in tickers.split(",")]

# Download historical data
@st.cache_data
def get_data(ticker_list):
    data = pd.DataFrame()
    for ticker in ticker_list:
        df = yf.download(ticker, start="2010-01-01", progress=False)
        data[ticker] = df['Adj Close']
    return data.ffill().dropna()

data = get_data(ticker_list)

if run_simulation:
    results = {}

    if portfolio_mode:
        weights_str = st.sidebar.text_input("Enter Portfolio Weights (same order)", value="0.4,0.4,0.2")
        weights = np.array([float(w) for w in weights_str.split(",")])
        if len(weights) != len(ticker_list):
            st.error("Weights length must match number of tickers!")
        else:
            st.subheader("📊 Portfolio Simulation")
            combined_log_returns = np.sum(np.log(data.pct_change()+1) * weights, axis=1)
            u = combined_log_returns.mean()
            var = combined_log_returns.var()
            drift = u - 0.5 * var
            stdev = combined_log_returns.std()
            S0 = (data.iloc[-1] * weights).sum()
            daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(time_horizon, num_simulations)))
            price_list = np.zeros_like(daily_returns)
            price_list[0] = S0
            for t in range(1, time_horizon):
                price_list[t] = price_list[t-1] * daily_returns[t]
            results["Portfolio"] = price_list
    else:
        for ticker in ticker_list:
            close_prices = data[ticker]
            log_returns = np.log(close_prices / close_prices.shift(1))
            u = log_returns.mean()
            var = log_returns.var()
            drift = u - 0.5 * var
            stdev = log_returns.std()
            S0 = close_prices.iloc[-1]
            daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(time_horizon, num_simulations)))
            price_list = np.zeros_like(daily_returns)
            price_list[0] = S0
            for t in range(1, time_horizon):
                price_list[t] = price_list[t-1] * daily_returns[t]
            results[ticker] = price_list

    # Plotting with Plotly
    fig = go.Figure()
    for key, price_list in results.items():
        mean_path = np.mean(price_list, axis=1)
        upper_95 = np.percentile(price_list, 97.5, axis=1)
        lower_95 = np.percentile(price_list, 2.5, axis=1)

        fig.add_trace(go.Scatter(x=np.arange(time_horizon), y=mean_path,
                                 name=f'{key} Mean Path', line=dict(width=2)))
        fig.add_trace(go.Scatter(x=np.arange(time_horizon), y=upper_95,
                                 fill=None, mode='lines', line_color='rgba(0,100,80,0.2)',
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=np.arange(time_horizon), y=lower_95,
                                 fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)',
                                 name=f'{key} 95% CI'))

    fig.update_layout(title="Monte Carlo Simulation - Future Price Paths",
                      xaxis_title="Days", yaxis_title="Price")
    st.plotly_chart(fig)

    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    for i, (key, price_list) in enumerate(results.items()):
        final_prices = price_list[-1]
        p90 = np.percentile(final_prices, 10)
        p50 = np.median(final_prices)
        expected_return = final_prices.mean()
        volatility = final_prices.std()

        with col1:
            st.write(f"### {key}")
        with col2:
            st.metric("Expected Price", f"${expected_return:.2f}")
        with col3:
            st.metric("P50 (Median)", f"${p50:.2f}")
        with col4:
            st.metric("Volatility", f"${volatility:.2f}")

    # Export Options
    st.sidebar.header("Export Options")

    # Export Chart as PNG
    if st.sidebar.button("Export Chart as PNG"):
        fig.write_image("simulation_chart.png")
        with open("simulation_chart.png", "rb") as file:
            btn = st.sidebar.download_button(
                label="Download PNG",
                data=file,
                file_name="simulation_chart.png",
                mime="image/png"
            )

    # Export CSV
    export_csv = st.sidebar.button("Export Simulation Data as CSV")
    if export_csv:
        df_export = pd.DataFrame({key: val[:, 0] for key, val in results.items()})
        csv = df_export.to_csv(index=False)
        st.sidebar.download_button(label="Download CSV", data=csv, file_name="simulation_results.csv", mime="text/csv")