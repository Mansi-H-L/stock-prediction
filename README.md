# Stock Prediction Website (Streamlit + ML)

A web-based application to visualize and forecast stock prices using real-time data, technical indicators, and machine learning models like , LSTM, Prophet, and XGBoost. Built using Python, Streamlit, Plotly, and several financial APIs.

---

##  Features

-  Historical stock data visualization with interactive charts
-  Forecasting using ARIMA, LSTM, Prophet, and XGBoost models
-  Technical indicators: Volume, High/Low Prices
-  Real-time news integration using NewsAPI
-  Twitter sentiment analysis
-  SMS alerts via Twilio API

---

##  Tech Stack

- **Frontend:** Streamlit, Plotly
- **Backend:** Python, Pandas, NumPy, yfinance
- **Machine Learning:** statsmodels Prophet, keras (LSTM), XGBoost
- **APIs:** 
  - Yahoo Finance (`yfinance`)
  - NewsAPI
  - Twitter API
  - Twilio (for SMS alerts)

---


### 1. Clone the Repository

```bash
git clone https://github.com/Mansi-H-L/stock-prediction.git
cd stock-prediction

2. Install Dependencies
Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

Install required Python packages

pip install -r requirements.txt

3. Set Up Environment Variables

Create a .env file or copy from .env.example, and add your API keys:
