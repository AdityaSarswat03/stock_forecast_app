import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Simple Stock Forecast", layout="wide")
st.title("📈 Simple Stock Forecasting")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 14)

@st.cache_data
def load_data(ticker, start, end):
    """Load stock data"""
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

def create_features(df):
    """Create simple features for prediction"""
    df = df.copy()
    
    # Simple moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    
    # Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_1'] = df['Price_Change'].shift(1)
    
    # Simple lag features
    df['Close_lag_1'] = df['Close'].shift(1)
    df['Close_lag_2'] = df['Close'].shift(2)
    
    # Time features
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    
    return df

# Load and prepare data
try:
    df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error("No data found. Try another ticker or date range.")
        st.stop()
    
    # Show raw data
    st.subheader("Stock Data")
    st.dataframe(df.tail())
    
    # Create features
    df_features = create_features(df)
    
    # Select feature columns
    feature_cols = ['MA_5', 'MA_20', 'Price_Change_1', 'Close_lag_1', 'Close_lag_2', 
                   'day', 'month', 'weekday']
    
    # Remove rows with NaN and prepare training data
    df_clean = df_features.dropna()
    
    if len(df_clean) < 20:
        st.error("Not enough data. Try a longer date range.")
        st.stop()
    
    X = df_clean[feature_cols]
    y = df_clean['Close']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test predictions
    test_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)
    
    # Simple forecast
    last_row = X.iloc[-1].copy()
    last_price = df_clean['Close'].iloc[-1]
    predictions = []
    
    for i in range(forecast_days):
        # Update time features
        future_date = df_clean['Date'].iloc[-1] + timedelta(days=i+1)
        last_row['day'] = future_date.day
        last_row['month'] = future_date.month
        last_row['weekday'] = future_date.weekday()
        
        # Predict
        pred_scaled = scaler.transform([last_row])
        pred = model.predict(pred_scaled)[0]
        predictions.append(pred)
        
        # Update for next prediction (simple approach)
        last_row['Close_lag_2'] = last_row['Close_lag_1']
        last_row['Close_lag_1'] = pred
    
    # Create forecast dataframe
    future_dates = [df_clean['Date'].iloc[-1] + timedelta(days=i) 
                   for i in range(1, forecast_days+1)]
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': predictions
    })
    
    # Plot results
    st.subheader("📈 Forecast Results")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 60 days)
    recent_data = df_clean.tail(60)
    ax.plot(recent_data['Date'], recent_data['Close'], 
           label='Historical', color='blue', linewidth=2)
    
    # Plot forecast
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], 
           label='Forecast', color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} Stock Price Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{float(r2):.3f}")
    with col2:
        st.metric("MSE", f"{float(mse):.2f}")
    with col3:
        current_price = float(df_clean['Close'].iloc[-1])
        forecast_price = float(predictions[-1])
        change_pct = ((forecast_price - current_price) / current_price) * 100
        st.metric("Forecast Change", f"{change_pct:.1f}%")
    
    # Show forecast table
    st.subheader("🔮 Forecast Details")
    forecast_display = forecast_df.copy()
    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
    forecast_display['Forecast'] = forecast_display['Forecast'].astype(float).round(2)
    st.dataframe(forecast_display)
    
    # Download option
    csv = forecast_df.to_csv(index=False)
    st.download_button("📥 Download Forecast", csv, f"{ticker}_forecast.csv")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Try a different stock ticker or date range.")