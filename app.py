def preprocess_data(df):
    df.index = pd.to_datetime(df.index)
    df_monthly = df.resample('ME').sum().astype(int)
    df_weekly = df.resample('W').sum().astype(int)
    return df_monthly, df_weekly

def forecast_sales(df, period, seasonal_periods):
    train = df[:-period]
    test = df[-period:]
    
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(len(test)).astype(int)
    
    # Metrics
    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    return train, test, forecast, {'MAPE': mape, 'RMSE': rmse, 'MSE': mse, 'MAE': mae}
def plot_forecast(train, test, forecast, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label='Train', color='blue')
    test.plot(ax=ax, label='Test', color='orange')
    forecast.plot(ax=ax, label='Forecast', color='green')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    return fig
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📈 Sales Forecasting App using Triple Exponential Smoothing")

uploaded_file = st.file_uploader("Upload your sales Excel or CSV file", type=['csv', 'xlsx'])

def preprocess_data(df):
    df.index = pd.to_datetime(df.index)
    df_monthly = df.resample('ME').sum().astype(int)
    df_weekly = df.resample('W').sum().astype(int)
    return df_monthly, df_weekly

def forecast_sales(df, period, seasonal_periods):
    train = df[:-period]
    test = df[-period:]
    
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(len(test)).astype(int)
    
    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    return train, test, forecast, {'MAPE': mape, 'RMSE': rmse, 'MSE': mse, 'MAE': mae}

def plot_forecast(train, test, forecast, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label='Train', color='blue')
    test.plot(ax=ax, label='Test', color='orange')
    forecast.plot(ax=ax, label='Forecast', color='green')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    return fig

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date', dayfirst=True)
    else:
        df = pd.read_excel(uploaded_file, parse_dates=['Date'], index_col='Date')

    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    df_monthly, df_weekly = preprocess_data(df)

    st.subheader("📅 Monthly Forecast")
    train_m, test_m, forecast_m, metrics_m = forecast_sales(df_monthly['Sales'], period=12, seasonal_periods=12)
    st.pyplot(plot_forecast(train_m, test_m, forecast_m, "Monthly Sales Forecast"))
    st.write("📊 Monthly Forecast Metrics:")
    st.json(metrics_m)

    st.subheader("📆 Weekly Forecast")
    train_w, test_w, forecast_w, metrics_w = forecast_sales(df_weekly['Sales'], period=18, seasonal_periods=52)
    st.pyplot(plot_forecast(train_w, test_w, forecast_w, "Weekly Sales Forecast"))
    st.write("📊 Weekly Forecast Metrics:")
    st.json(metrics_w)