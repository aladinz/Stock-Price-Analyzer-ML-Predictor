# 📈 Stock Price Analyzer & ML Predictor

A powerful web application built with **Streamlit** that analyzes historical stock prices and predicts future movements using machine learning and technical analysis.

## ✨ Features

### 📊 Overview Tab
- Real-time stock data visualization
- Historical price trends with interactive charts
- Key metrics: Latest price, 52-week high/low
- Data summary and record count

### 🤖 ML Model Tab
- **Linear Regression** model for next-day price prediction
- Model performance metrics:
  - Training & Testing R² scores
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Feature importance analysis
- Actual vs. Predicted price visualization

### 📈 Analysis Tab
- **Moving Averages** (customizable periods)
- **Next-day price prediction** with confidence scores
- Interactive price trend charts
- Real-time model insights

### 💹 Swing Trading Tab
- **Technical Indicators:**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Average True Range (ATR)
  - Support & Resistance levels
- **Trading Signals & Recommendations:**
  - BUY / SELL / HOLD recommendations
  - Signal scoring system
  - Trade setup with stop loss & take profit levels
  - Win rate analysis

### 📋 Data Tab
- Full dataset download as CSV
- Clean data export with all features

## 🚀 Deployment

### Deploy on Streamlit Cloud (Free & Easy)
1. Go to https://streamlit.io/cloud
2. Click **"New app"**
3. Select repository: `aladinz/Stock-Price-Analyzer-ML-Predictor`
4. Select main branch and `app.py`
5. Click **Deploy**

Your app will be live instantly! 🌍

## 💻 Local Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
# Clone the repository
git clone https://github.com/aladinz/Stock-Price-Analyzer-ML-Predictor.git
cd Stock-Price-Analyzer-ML-Predictor

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📦 Dependencies

- **streamlit** - Web app framework
- **yfinance** - Stock data fetching
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **plotly** - Interactive visualizations
- **numpy** - Numerical computing

## 📖 How to Use

1. **Enter Stock Ticker** - e.g., AAPL, GOOGL, MSFT, TSLA
2. **Set Date Range** - Choose historical period for analysis
3. **Adjust Parameters** (Optional):
   - Test set size (10-50%)
   - Short/Long moving average periods
4. **View Results**:
   - Historical trends and patterns
   - ML predictions for next day
   - Technical indicators and trading signals
   - Risk/Reward analysis

## ⚡ Quick Tips

✅ Use **1-year data range** for best results  
✅ Established stocks (AAPL, MSFT, GOOGL) have more reliable predictions  
✅ Check **Model Confidence (R²)** - higher is better  
✅ Use **Trading Signals** as supplementary analysis  
✅ Always practice **risk management** with stop losses

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It should NOT be used as financial advice. Stock market predictions carry risk, and past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## 🔧 Model Details

### Features Used
- **Close Price** - Current closing price
- **Short MA** - Short-term moving average
- **Long MA** - Long-term moving average

### Target Variable
- **Next Day Closing Price** - For one-step-ahead predictions

### Model Validation
- 80/20 train-test split
- Cross-validation with multiple metrics
- Error analysis with MAPE & directional accuracy

## 📊 Technical Indicators Explained

| Indicator | Purpose |
|-----------|---------|
| **RSI** | Identifies overbought (>70) / oversold (<30) conditions |
| **MACD** | Identifies trend changes and momentum |
| **Bollinger Bands** | Shows volatility and price extremes |
| **ATR** | Measures market volatility for risk management |
| **Support/Resistance** | Identifies key price levels |

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## 📄 License

Open source - Use freely for educational purposes

## 👨‍💻 Author

Made with ❤️ by **Aladdin**

---

**Happy Analyzing! 📈**
