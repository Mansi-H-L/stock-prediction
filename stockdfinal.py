import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from textblob import TextBlob
import tweepy
from dotenv import load_dotenv
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from newsapi import NewsApiClient
from twilio.rest import Client
from sklearn.model_selection import train_test_split  # Add this import
import xgboost as xgb  # Add this import
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler  # Add this import
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from plotly.subplots import make_subplots
import plotly.graph_objects as go



st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
ALERT_PHONE_NUMBER = os.getenv('ALERT_PHONE_NUMBER')  # Phone number to receive alerts

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# News API v2 authentication
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
def get_realtime_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d', interval='1m')
    return data

# Sidebar for inputs
st.sidebar.title("Stock Market Analysis Dashboard")
stocks = ["AAPL", "TSLA", "GOOGL", "META"]
ticker = st.sidebar.selectbox("Select Stock Symbol", stocks)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Download historical data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)
data = data.ffill()

# Display real-time stock data
realtime_data = get_realtime_data(ticker)
st.markdown(f"### Real-time Stock Data for {ticker}")
st.dataframe(realtime_data)

# Function to fetch latest news from News API
def get_latest_news(symbol, api_key):
    all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page_size=5)
    return all_articles

# Function to analyze news sentiment
def analyze_news_sentiment(articles):
    sentiments = []
    for article in articles:
        text = article['title'] + " " + article['description']
        sentiment = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment)
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment

# Function to send SMS alert using Twilio
def send_alert(message):
    # Display message in Streamlit
    st.write(message)

    # Send SMS alert using Twilio
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        st.write("Alert sent via SMS.")
    except Exception as e:
        st.error(f"Failed to send SMS alert: {e}")

# Display real-time stock data


# Fetch and display news articles
st.markdown(f"### Latest News for {ticker}")
news_articles = get_latest_news(ticker, NEWS_API_KEY)
if news_articles['articles']:
    avg_sentiment = analyze_news_sentiment(news_articles['articles'])
    
    st.write(f"**Average Sentiment Polarity for {ticker}:** {avg_sentiment:.2f}")
    
    if avg_sentiment > 0:
        st.success("Positive sentiment")
    elif avg_sentiment < 0:
        st.error("Negative sentiment")
    else:
        st.warning("Neutral sentiment")
    
    for article in news_articles['articles']:
        st.markdown(f"**{article['title']}**")
        st.markdown(f"*{article['source']['name']}* - {article['publishedAt']}")
        st.markdown(article['description'])
        st.markdown(f"[Read more]({article['url']})")
        st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.write("No recent news available.")

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if not data.empty:
    # Ensure float type and flatten any Series-like columns
    volume = data['Volume'].astype(float).squeeze()
    low = data['Low'].astype(float).squeeze()
    high = data['High'].astype(float).squeeze()
    close = data['Close'].astype(float).squeeze()
    open_price = data['Open'].astype(float).squeeze()

    # Get scalar values safely
    highest_volume = float(volume.max())
    lowest_volume = float(volume.min())
    all_time_low_price = float(low.min())
    all_time_high_price = float(high.max())
    volume_last_7_days = float(volume[-7:].sum())
    volume_last_30_days = float(volume[-30:].sum())

    # Layout setup
    col1, col2 = st.columns([2, 3])

    # Plotly chart with volume and prices
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=data.index, y=volume, name="Volume", marker_color='blue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=close, name="Closing Price", mode='lines+markers', marker_color='red'), secondary_y=True)
    fig.add_trace(go.Scatter(x=data.index, y=open_price, name="Opening Price", mode='lines+markers', marker_color='green'), secondary_y=True)
    fig.update_layout(title=f"Closing VS Opening and Total Volume for {ticker}", xaxis_title="Date", width=950)

    # Display chart
    with col2:
        st.plotly_chart(fig, use_container_width=True)

    # Display metrics
    with col1:
        st.metric("Highest Volume Traded", f"{highest_volume / 1e6:.1f}M")
        st.metric("Lowest Volume Traded", f"{lowest_volume / 1e6:.1f}M")
        st.metric("All Time Low Price", f"${all_time_low_price:.2f}")
        st.metric("All Time High Price", f"${all_time_high_price:.2f}")
        st.metric("Volume Last 7 Days", f"{volume_last_7_days / 1e6:.2f}M")
        st.metric("Volume Last 30 Days", f"{volume_last_30_days / 1e6:.2f}M")

    # Summary Table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Stocks Summary")

    summary = {
        "Description": ["Volume", "High Price", "Low Price", "Closing Price"],
        "Last 7 Days": [
            round(volume[-7:].mean(), 2),
            round(high[-7:].mean(), 2),
            round(low[-7:].mean(), 2),
            round(close[-7:].mean(), 2)
        ],
        "Last 30 Days": [
            round(volume[-30:].mean(), 2),
            round(high[-30:].mean(), 2),
            round(low[-30:].mean(), 2),
            round(close[-30:].mean(), 2)
        ],
        "Selected Days": [
            round(volume.mean(), 2),
            round(high.mean(), 2),
            round(low.mean(), 2),
            round(close.mean(), 2)
        ]
    }

    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df.style.set_properties(**{'text-align': 'right'}))

if not data.empty:
    # Add a dropdown for model selection in the sidebar
    model_option = st.sidebar.selectbox("Select Model", ["LSTM", "Prophet", "XGBoost"])

    if model_option == "LSTM":
        # Stock Price Prediction using LSTM
        st.markdown("<hr>", unsafe_allow_html=True)
        st.write("### Stock Price Prediction using LSTM")

        # Load and preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']])

        # Prepare the data for LSTM
        prediction_days = 60
        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Predict the stock prices
        test_data = scaled_data[-prediction_days:]
        x_test = [test_data]
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_price = model.predict(x_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        st.write(f"Predicted price for the next day: {predicted_price[-1][0]}")

    elif model_option == "Prophet":
        # Stock Price Prediction using Prophet
        st.markdown("<hr>", unsafe_allow_html=True)
        st.write("### Stock Price Prediction using Prophet")

        # Prepare the data for Prophet
        prophet_data = data.reset_index()
        prophet_data['ds'] = prophet_data['Date']
        prophet_data['y'] = prophet_data['Close']
        prophet_data = prophet_data[['ds', 'y']]

        # Initialize and fit the model
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)

        # Make future predictions
        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)

        # Plot the results
        fig_prophet = prophet_model.plot(forecast)
        st.write(fig_prophet)

    elif model_option == "XGBoost":
        # Stock Price Prediction using XGBoost
        st.markdown("<hr>", unsafe_allow_html=True)
        st.write("### Stock Price Prediction using XGBoost")

        # Create features
        data['Date'] = pd.to_datetime(data.index)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek

        # Create target variable
        data['Target'] = data['Close'].shift(-1)
        data.dropna(inplace=True)

        # Prepare features and target
        X = data[['Year', 'Month', 'Day', 'DayOfWeek', 'Close']]
        y = data['Target']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train the model
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        xgb_model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")

        # Plot the results
        fig_xgb = go.Figure()
        fig_xgb.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Actual'))
        fig_xgb.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted'))
        fig_xgb.update_layout(title=f"{ticker} Stock Price Prediction using XGBoost", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_xgb)

    else:
        st.write("No data available for the selected stock and date range.")


    # Anomaly Detection
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Anomaly Detection in Volume and Close Price")

    def detect_anomalies(data, threshold=2):
        anomalies = []
        std_dev = np.std(data)
        mean = np.mean(data)
        for i, value in enumerate(data):
            if abs(value - mean) > threshold * std_dev:
                anomalies.append(i)
        return anomalies

    volume_anomalies = detect_anomalies(volume)
    price_anomalies = detect_anomalies(close)

    fig_anomalies = make_subplots(specs=[[{"secondary_y": True}]])
    fig_anomalies.add_trace(go.Scatter(x=data.index, y=volume, name="Volume", mode='lines'), secondary_y=False)
    fig_anomalies.add_trace(go.Scatter(x=data.index, y=close, name="Close Price", mode='lines'), secondary_y=True)
    fig_anomalies.add_trace(go.Scatter(x=data.index[volume_anomalies], y=volume[volume_anomalies], mode='markers', name="Volume Anomalies", marker=dict(color='red', size=10)), secondary_y=False)
    fig_anomalies.add_trace(go.Scatter(x=data.index[price_anomalies], y=close[price_anomalies], mode='markers', name="Price Anomalies", marker=dict(color='purple', size=10)), secondary_y=True)
    fig_anomalies.update_layout(title="Anomaly Detection in Volume and Close Price", xaxis_title="Date", width=950)

    st.plotly_chart(fig_anomalies)

    # Candlestick Charts for detailed view of price movements
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Candlestick Charts")

    fig_candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'])])
    fig_candlestick.update_layout(title=f"Candlestick Chart for {ticker}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_candlestick)

    # Sector Performance Comparison (if available)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Sector Performance Comparison")

    # Dummy data for demonstration (replace with actual sector performance data)
    sector_performance = {
        "AAPL": {"Sector": "Technology", "Performance": 0.15},
        "TSLA": {"Sector": "Automotive", "Performance": -0.05},
        "GOOGL": {"Sector": "Technology", "Performance": 0.12},
        "META": {"Sector": "Finance", "Performance": 0.08}
    }

    if ticker in sector_performance:
        sector = sector_performance[ticker]['Sector']
        performance = sector_performance[ticker]['Performance']

        st.write(f"**Sector:** {sector}")
        st.write(f"**Performance:** {performance}")

        # Visualize sector performance comparison
        sector_data = pd.DataFrame(sector_performance).transpose()
        sector_data['Performance'] = pd.to_numeric(sector_data['Performance'])
        fig_sector_performance = go.Figure(data=go.Bar(x=sector_data.index, y=sector_data['Performance'], name='Performance'))
        fig_sector_performance.update_layout(title=f"Sector Performance Comparison for {ticker}", xaxis_title="Stock", yaxis_title="Performance")
        st.plotly_chart(fig_sector_performance)

    else:
        st.warning("Sector performance data not available for the selected stock.")

    # Alerts
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("### Alerts")

    st.sidebar.write("Set Alerts")
    alert_price = st.sidebar.number_input("Alert if price crosses:", value=100.0)
    alert_volume = st.sidebar.number_input("Alert if volume exceeds:", value=1000000.0)

    if close[-1] > alert_price:
        send_alert(f"Alert: {ticker} price has crossed ${alert_price}")

    if volume[-1] > alert_volume:
        send_alert(f"Alert: {ticker} volume has exceeded {alert_volume}")

else:
    st.error("No data available for the selected date range. Please adjust your date range and try again.")
