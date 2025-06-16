# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("📈 Sales Forecasting with Triple Exponential Smoothing")
st.markdown("Upload your sales data (CSV or Excel) with `Date` and `Sales` columns.")

# Upload file
uploaded_file = st.file_uploader("Upload your sales file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
    else:
        df = pd.read_excel(uploaded_file, parse_dates=['Date'], dayfirst=True)

    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    st.subheader("📋 Data Preview")
    st.write(df.head())

    # Monthly and Weekly Aggregation
    df_monthly = df.resample('M').sum()
    df_weekly = df.resample('W').sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Monthly Sales Trend")
        st.line_chart(df_monthly)

    with col2:
        st.subheader("📅 Weekly Sales Trend")
        st.line_chart(df_weekly)

    # Forecasting - Monthly
    periods_monthly = 12
    train_m = df_monthly[:-periods_monthly]
    test_m = df_monthly[-periods_monthly:]

    model_m = ExponentialSmoothing(train_m, trend='add', seasonal='add', seasonal_periods=12).fit()
    pred_m = model_m.forecast(periods_monthly)

    mape_m = np.mean(np.abs((test_m['Sales'] - pred_m) / test_m['Sales'])) * 100

    # Forecasting - Weekly
    periods_weekly = 18
    train_w = df_weekly[:-periods_weekly]
    test_w = df_weekly[-periods_weekly:]

    model_w = ExponentialSmoothing(train_w, trend='add', seasonal='add', seasonal_periods=52).fit()
    pred_w = model_w.forecast(periods_weekly)

    mape_w = np.mean(np.abs((test_w['Sales'] - pred_w) / test_w['Sales'])) * 100

    st.subheader("📊 Monthly Forecast")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(train_m, label='Train')
    ax1.plot(test_m, label='Test')
    ax1.plot(pred_m, label='Forecast', linestyle='--')
    ax1.legend()
    st.pyplot(fig1)
    st.metric("MAPE (Monthly)", f"{mape_m:.2f}%")

    st.subheader("📊 Weekly Forecast")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(train_w, label='Train')
    ax2.plot(test_w, label='Test')
    ax2.plot(pred_w, label='Forecast', linestyle='--')
    ax2.legend()
    st.pyplot(fig2)
    st.metric("MAPE (Weekly)", f"{mape_w:.2f}%")

    # Optional: Full Forecast for Future
    st.subheader("📅 Full Monthly Forecast (Next 12 months)")
    final_model_m = ExponentialSmoothing(df_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast_future = final_model_m.forecast(12)
    st.line_chart(pd.concat([df_monthly, forecast_future.rename("Forecast")], axis=1))

else:
    st.info("Upload a file above to start forecasting.")
