import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    return df


# -----------------------------
# Aggregate Sales
# -----------------------------
def aggregate_sales(df):
    sales_data = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    return sales_data


# -----------------------------
# Linear Regression
# -----------------------------
def train_linear_regression(sales_data):
    sales_data = sales_data.copy()
    sales_data['Date_Ordinal'] = sales_data['Date'].map(pd.Timestamp.toordinal)

    X = sales_data[['Date_Ordinal']]
    y = sales_data['Weekly_Sales']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    return model, predictions, mae


# -----------------------------
# ARIMA Forecast
# -----------------------------
def arima_forecast(sales_data, steps=12):
    model = ARIMA(sales_data['Weekly_Sales'], order=(5, 1, 0))
    result = model.fit()
    forecast = result.forecast(steps=steps)
    return forecast


# -----------------------------
# Plot Sales Trend
# -----------------------------
def plot_sales_trend(sales_data):
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data['Date'], sales_data['Weekly_Sales'])
    plt.title("Total Weekly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot Linear Regression
# -----------------------------
def plot_linear_regression(sales_data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data['Date'], sales_data['Weekly_Sales'], label="Actual Sales")
    plt.plot(sales_data['Date'], predictions, label="Linear Regression Prediction")
    plt.title("Linear Regression - Sales Prediction")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot ARIMA Forecast
# -----------------------------
def plot_arima_forecast(sales_data, forecast):
    future_dates = pd.date_range(
        start=sales_data['Date'].iloc[-1],
        periods=len(forecast) + 1,
        freq='W'
    )[1:]

    plt.figure(figsize=(12, 6))
    plt.plot(sales_data['Date'], sales_data['Weekly_Sales'], label="Historical Sales")
    plt.plot(future_dates, forecast, label="ARIMA Forecast")
    plt.title("ARIMA Forecast - Next Weeks Sales")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Running Sales Data Analysis & Forecasting...\n")

    df = load_data("data/walmart_sales.csv")
    sales_data = aggregate_sales(df)

    # 1️⃣ Sales Trend
    plot_sales_trend(sales_data)

    # 2️⃣ Linear Regression
    model, predictions, mae = train_linear_regression(sales_data)
    print("Linear Regression MAE:", mae)
    plot_linear_regression(sales_data, predictions)

    # 3️⃣ ARIMA Forecast
    forecast = arima_forecast(sales_data, steps=12)
    print("\nNext 12 Week Forecast:")
    print(forecast)
    plot_arima_forecast(sales_data, forecast)
