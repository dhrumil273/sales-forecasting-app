# app2.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    # Seasonal Decomposition
    st.subheader("🔍 Seasonal Decomposition")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Data (Additive)**")
        decomposition_m = seasonal_decompose(df_monthly, model='additive', period=12)
        fig_m = decomposition_m.plot()
        fig_m.set_size_inches(10, 6)
        st.pyplot(fig_m)

    with col2:
        st.markdown("**Weekly Data (Additive)**")
        decomposition_w = seasonal_decompose(df_weekly, model='additive', period=52)
        fig_w = decomposition_w.plot()
        fig_w.set_size_inches(10, 6)
        st.pyplot(fig_w)

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

    # Full Forecast
    st.subheader("📅 Full Monthly Forecast (Next 12 months)")
    final_model_m = ExponentialSmoothing(df_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast_future_m = final_model_m.forecast(12)
    st.line_chart(pd.concat([df_monthly, forecast_future_m.rename("Forecast")], axis=1))

    st.subheader("📅 Full Weekly Forecast (Next 18 weeks)")
    final_model_w = ExponentialSmoothing(df_weekly, trend='add', seasonal='add', seasonal_periods=52).fit()
    forecast_future_w = final_model_w.forecast(18)
    st.line_chart(pd.concat([df_weekly, forecast_future_w.rename("Forecast")], axis=1))

else:
    st.info("Upload a file above to start forecasting.")
