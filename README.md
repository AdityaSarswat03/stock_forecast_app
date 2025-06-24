# 📈 Stock Forecast App

A machine learning-based stock price forecasting web app built using Python, Streamlit, and XGBoost. This app allows users to interactively select a stock ticker and date range, then view and download a forecast for the next 7 to 30 days.

---

## 🔍 Features

- 🧠 **ML-Powered Forecasting**: Predicts future stock prices using an XGBoost regression model.
- 🔎 **Custom Input**: Choose your own stock ticker and historical date range.
- 📅 **Forecast Range**: Predicts the next 7–30 days of stock closing prices.
- 📊 **Visualization**: Shows historical vs forecasted data using interactive charts.
- 📥 **Download Option**: Export forecast results as a CSV file.
- 🧪 **Performance Metrics**: Displays R² Score, Mean Squared Error, and % change from current price.

---

## 🚀 Live Demo

👉 [Click here to try the live app](https://stockforecastapp-5j2att5a6kfjlykue9qxon.streamlit.app/)



---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| [Python](https://www.python.org/) | Core programming language |
| [Streamlit](https://streamlit.io/) | Web app framework |
| [yFinance](https://pypi.org/project/yfinance/) | Fetching live stock data |
| [Pandas](https://pandas.pydata.org/) | Data handling and feature engineering |
| [Scikit-learn](https://scikit-learn.org/) | Data preprocessing and metrics |
| [XGBoost](https://xgboost.readthedocs.io/) | Machine learning model |
| [Matplotlib](https://matplotlib.org/) | Chart plotting |

---
## 🛠️ How It Works

1. User inputs a stock ticker (e.g. AAPL, TSLA) and selects date range
2. App downloads historical data using `yfinance`
3. Features like moving averages, lag values, and price changes are generated
4. Data is scaled and passed to an `XGBoostRegressor` model
5. Forecast is generated recursively day-by-day
6. Results are displayed and can be downloaded


## 📁 Project Structure

stock-forecast-app/
├── app.py # Main Streamlit app script
├── requirements.txt # Python dependencies
└── README.md # This documentation file