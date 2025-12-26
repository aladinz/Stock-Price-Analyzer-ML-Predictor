import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ===== TECHNICAL INDICATORS FUNCTIONS =====
def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def find_support_resistance(data, lookback=20):
    """Find support and resistance levels"""
    local_max = data.rolling(window=lookback).max()
    local_min = data.rolling(window=lookback).min()
    resistance = local_max.iloc[-1]
    support = local_min.iloc[-1]
    return support, resistance

# Set page configuration
st.set_page_config(
    page_title="Stock Price Analyzer & ML Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-style {
        color: #1f77b4;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader-style {
        color: #2c3e50;
        font-size: 20px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-style'>📈 Stock Price Analyzer & ML Predictor</div>", unsafe_allow_html=True)
st.markdown("Analyze historical stock prices and predict next day's closing price using machine learning", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter the stock ticker symbol (e.g., AAPL, GOOGL, MSFT)")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365)
    )
with col2:
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now()
    )

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20)
    ma_short = st.slider("Short Moving Average (days)", min_value=2, max_value=20, value=3)
    ma_long = st.slider("Long Moving Average (days)", min_value=3, max_value=100, value=5)

# Validate inputs
if start_date >= end_date:
    st.error("❌ Error: Start date must be before end date!")
else:
    try:
        # Fetch historical data
        with st.spinner(f"📊 Fetching data for {ticker.upper()}..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Validate data was fetched successfully
        if data is None or len(data) == 0:
            st.error(f"❌ Error: No data found for ticker '{ticker.upper()}'. Please check the ticker symbol.")
            st.info("💡 Try using a valid stock ticker like AAPL, GOOGL, MSFT, or TSLA.")
            st.stop()
        
        # Handle MultiIndex columns if present (yfinance sometimes returns this)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Check if Close column exists
        if 'Close' not in data.columns:
            st.error("❌ Error: Could not retrieve price data. Please check the ticker symbol.")
            st.stop()
        
        df = data[['Close']].copy()
        
        # Check for NaN values in raw data
        if df['Close'].isna().sum() == len(df):
            st.error("❌ Error: All price data is missing. Please try a different ticker or date range.")
            st.stop()
        
        # Remove leading NaN values
        df = df.dropna()
        
        if len(df) == 0:
            st.error("❌ Error: No valid price data available. Please try a different ticker or date range.")
            st.stop()
        
        if len(df) < 20:
            st.warning(f"⚠️ Warning: Very limited data ({len(df)} points). Consider extending the date range for better predictions.")
        
        # Feature engineering - do this before tabs so data is ready for all tabs
        with st.spinner("⚙️ Creating features..."):
            # Dynamically adjust window sizes based on available data
            # For small datasets, use very small windows to preserve data
            if len(df) < 30:
                actual_ma_short = 2
                actual_ma_long = 3
            else:
                actual_ma_short = max(2, min(ma_short, max(3, len(df) // 10)))
                actual_ma_long = max(3, min(ma_long, max(5, len(df) // 8)))
            
            # Ensure long MA is larger than short MA
            if actual_ma_long <= actual_ma_short:
                actual_ma_long = actual_ma_short + 1
            
            df['MA_Short'] = df['Close'].rolling(window=actual_ma_short).mean()
            df['MA_Long'] = df['Close'].rolling(window=actual_ma_long).mean()
            df['Target'] = df['Close'].shift(-1)
            
            df_clean = df.dropna()
            
            # For small datasets, be very lenient with minimum samples
            if len(df_clean) < 3:
                st.error(f"❌ Error: Not enough data after processing. Have {len(df)} raw points but only {len(df_clean)} after feature engineering.")
                st.info(f"💡 Try one of these:\n- Extend your date range (currently {(end_date - start_date).days} days)\n- Check ticker symbol is correct\n- Use a more established stock ticker")
                st.stop()
            
            X = df_clean[['Close', 'MA_Short', 'MA_Long']].values
            y = df_clean['Target'].values
            
            # For very small datasets, use a smaller test size
            actual_test_size = min(test_size/100, 0.4) if len(df_clean) < 10 else test_size/100
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=actual_test_size, random_state=42
            )
            
            model = LinearRegression()
            model.fit(X_train, y_train)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🤖 ML Model", "📈 Analysis", "💹 Swing Trading", "📋 Data"])
        
        with tab1:
            st.markdown("<div class='subheader-style'>Historical Data Summary</div>", unsafe_allow_html=True)
            
            # Display first 5 rows
            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                st.dataframe(df.head(5), use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df), delta=None)
            with col2:
                latest = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2])
                delta = latest - prev
                st.metric("Latest Close", f"${latest:.2f}", delta=f"${delta:.2f}")
            with col3:
                highest = float(df['Close'].max())
                st.metric("52-Week High", f"${highest:.2f}")
            with col4:
                lowest = float(df['Close'].min())
                st.metric("52-Week Low", f"${lowest:.2f}")
            
            # Historical prices chart
            st.markdown("<div class='subheader-style'>Closing Price Trend</div>", unsafe_allow_html=True)
            
            # Create a simple dataframe for plotting
            chart_df = df[['Close']].reset_index()
            
            fig_close = px.line(chart_df, x='Date', y='Close', 
                              title='',
                              labels={'Close': 'Price ($)', 'Date': 'Date'},
                              template='plotly_dark')
            fig_close.update_traces(line=dict(color='#00D9FF', width=3))
            fig_close.update_layout(
                hovermode='x unified',
                height=450,
                showlegend=True,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)'),
                margin=dict(l=60, r=30, t=30, b=60),
                font=dict(family="Arial, sans-serif", size=12, color='#CCCCCC')
            )
            st.plotly_chart(fig_close, use_container_width=True)
        
        with tab2:
            st.markdown("<div class='subheader-style'>Machine Learning Model</div>", unsafe_allow_html=True)
            
            # Model Performance
            st.markdown("<div class='subheader-style'>Model Performance</div>", unsafe_allow_html=True)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training R²", f"{train_score:.4f}", help="Goodness of fit on training data")
            with col2:
                st.metric("Testing R²", f"{test_score:.4f}", help="Goodness of fit on test data")
            with col3:
                st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error")
            with col4:
                st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error")
            
            # Model Coefficients
            st.markdown("<div class='subheader-style'>Model Coefficients</div>", unsafe_allow_html=True)
            coef_data = pd.DataFrame({
                'Feature': [f'{ma_short}-Day MA', f'{ma_long}-Day MA', 'Close Price'],
                'Coefficient': [model.coef_[1], model.coef_[2], model.coef_[0]]
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coef_data, use_container_width=True)
            st.write(f"**Intercept:** ${model.intercept_:.4f}")
            
            # Actual vs Predicted
            st.markdown("<div class='subheader-style'>Actual vs Predicted Prices (Test Set)</div>", unsafe_allow_html=True)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                y=y_test,
                mode='markers',
                name='Actual Price',
                marker=dict(size=10, color='#00D9FF', opacity=0.8),
                hovertemplate='Actual: $%{y:.2f}<extra></extra>'
            ))
            fig_pred.add_trace(go.Scatter(
                y=y_pred_test,
                mode='markers',
                name='Predicted Price',
                marker=dict(size=10, color='#FF6B6B', opacity=0.8),
                hovertemplate='Predicted: $%{y:.2f}<extra></extra>'
            ))
            fig_pred.update_layout(
                title="",
                xaxis_title="Sample Index",
                yaxis_title="Price ($)",
                hovermode='closest',
                height=400,
                template='plotly_dark',
                showlegend=True,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                margin=dict(l=60, r=30, t=30, b=60),
                font=dict(family="Arial, sans-serif", size=12, color='#CCCCCC')
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with tab3:
            st.markdown("<div class='subheader-style'>Moving Averages & Prediction</div>", unsafe_allow_html=True)
            
            # Predict next day
            last_close = float(df['Close'].iloc[-1])
            last_ma_short = float(df['MA_Short'].iloc[-1])
            last_ma_long = float(df['MA_Long'].iloc[-1])
            next_day_features = np.array([[last_close, last_ma_short, last_ma_long]])
            predicted_next_price = float(model.predict(next_day_features)[0])
            
            # Prediction Cards
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Predicted Next Day Price", f"${predicted_next_price:.2f}")
                price_change = predicted_next_price - last_close
                direction = "📈 Up" if price_change > 0 else "📉 Down"
                st.write(f"{direction}: ${abs(price_change):.2f} ({abs(price_change/last_close)*100:.2f}%)")
            with col2:
                st.metric("🎲 Model Confidence (R²)", f"{test_score:.4f}")
                confidence_pct = test_score * 100
                st.write(f"Explains {confidence_pct:.1f}% of price variance")
            
            # Chart with moving averages and prediction
            chart_df_ma = df_clean[['Close', 'MA_Short', 'MA_Long']].reset_index()
            
            fig_ma = px.line(chart_df_ma, x='Date', y=['Close', 'MA_Short', 'MA_Long'],
                            title='',
                            labels={'value': 'Price ($)', 'Date': 'Date', 'variable': 'Series'},
                            template='plotly_dark')
            
            # Update line styles and colors
            fig_ma.for_each_trace(lambda t: t.update(line=dict(width=2)) if t.name == 'Close' else t.update(line=dict(width=2, dash='dash')))
            fig_ma.for_each_trace(lambda t: t.update(line_color='#00D9FF') if t.name == 'Close' else (
                t.update(line_color='#FFD700') if t.name == 'MA_Short' else t.update(line_color='#FF6B6B')))
            
            # Add prediction marker
            next_trading_day = df.index[-1] + timedelta(days=1)
            fig_ma.add_scatter(
                x=[next_trading_day],
                y=[predicted_next_price],
                mode='markers',
                name='Next Day Prediction',
                marker=dict(size=18, color='#00FF00', symbol='star'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: $%{y:.2f}<extra></extra>'
            )
            
            fig_ma.update_layout(
                hovermode='x unified',
                height=450,
                showlegend=True,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)'),
                margin=dict(l=60, r=30, t=30, b=60),
                font=dict(family="Arial, sans-serif", size=12, color='#CCCCCC')
            )
            st.plotly_chart(fig_ma, use_container_width=True)
        
        with tab4:
            st.markdown("<div class='subheader-style'>💹 SWING TRADING ANALYSIS</div>", unsafe_allow_html=True)
            
            # Calculate technical indicators with dynamic periods based on data size
            rsi_period = max(2, min(14, len(df) // 3))
            macd_fast = max(2, min(12, len(df) // 5))
            macd_slow = max(3, min(26, len(df) // 4))
            bb_period = max(2, min(20, len(df) // 3))
            atr_period = max(2, min(14, len(df) // 3))
            
            rsi = calculate_rsi(df['Close'], period=rsi_period)
            macd_line, signal_line, histogram = calculate_macd(df['Close'], fast=macd_fast, slow=macd_slow)
            upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(df['Close'], period=bb_period)
            atr = calculate_atr(df['High'] if 'High' in df.columns else df['Close'], 
                               df['Low'] if 'Low' in df.columns else df['Close'], 
                               df['Close'], period=atr_period)
            lookback_period = max(2, min(20, len(df) // 3))
            support, resistance = find_support_resistance(df['Close'], lookback=lookback_period)
            
            current_price = float(df['Close'].iloc[-1])
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            current_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0
            current_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0
            current_upper_bb = float(upper_bb.iloc[-1]) if not pd.isna(upper_bb.iloc[-1]) else current_price
            current_lower_bb = float(lower_bb.iloc[-1]) if not pd.isna(lower_bb.iloc[-1]) else current_price
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
            current_middle_bb = float(middle_bb.iloc[-1]) if not pd.isna(middle_bb.iloc[-1]) else current_price
            
            # Calculate Bollinger Band position
            bb_position = ((current_price - current_lower_bb) / (current_upper_bb - current_lower_bb)) * 100 if current_upper_bb != current_lower_bb else 50
            
            # ===== SECTION 4: Trading Summary & Recommendation (MOVED TO TOP) =====
            st.markdown("### 📋 Trading Summary & Recommendation", unsafe_allow_html=True)
            
            signals_bullish = 0
            signals_bearish = 0
            signal_details = []
            
            # Count bullish signals
            if current_rsi < 50:
                signals_bullish += 1
                signal_details.append("✅ RSI below 50 (momentum up)")
            else:
                signals_bearish += 1
                signal_details.append("❌ RSI above 50 (momentum down)")
            
            if float(current_macd) > float(current_signal):
                signals_bullish += 1
                signal_details.append("✅ MACD above signal line (bullish)")
            else:
                signals_bearish += 1
                signal_details.append("❌ MACD below signal line (bearish)")
            
            if float(current_price) > float(current_middle_bb):
                signals_bullish += 1
                signal_details.append("✅ Price above middle BB (uptrend)")
            else:
                signals_bearish += 1
                signal_details.append("❌ Price below middle BB (downtrend)")
            
            if float(predicted_next_price) > float(current_price):
                signals_bullish += 1
                signal_details.append(f"✅ Model predicts UP (${predicted_next_price:.2f})")
            else:
                signals_bearish += 1
                signal_details.append(f"❌ Model predicts DOWN (${predicted_next_price:.2f})")
            
            # Display recommendation
            recommendation_score = signals_bullish / (signals_bullish + signals_bearish) * 100
            
            if recommendation_score >= 75:
                rec_color = "🟢"
                rec_text = "STRONG BUY"
            elif recommendation_score >= 60:
                rec_color = "🟢"
                rec_text = "BUY"
            elif recommendation_score <= 25:
                rec_color = "🔴"
                rec_text = "STRONG SELL"
            elif recommendation_score <= 40:
                rec_color = "🔴"
                rec_text = "SELL"
            else:
                rec_color = "🟡"
                rec_text = "HOLD"
            
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(f"## {rec_color} {rec_text}")
                st.markdown(f"**Signal Score: {recommendation_score:.0f}%**  \n{signals_bullish} Bullish / {signals_bearish} Bearish")
            with col2:
                st.markdown("#### 📈 Trade Setup")
                suggested_stop_loss = current_price - (current_atr * 2)
                suggested_tp = current_price + (current_atr * 3)
                st.markdown(f"🎯 Entry: ${current_price:.2f}  \n🛑 SL: ${suggested_stop_loss:.2f}  \n💰 TP: ${suggested_tp:.2f}")
            
            st.markdown("---")
            
            # Modern Signal Details Chart
            st.markdown("### 📊 Signal Details Visualization", unsafe_allow_html=True)
            
            # Create circular signal strength chart
            fig_gauge = go.Figure()
            
            # Add background circle
            theta = np.linspace(0, 2*np.pi, 100)
            fig_gauge.add_trace(go.Scatterpolar(
                r=[1]*100,
                theta=np.degrees(theta),
                fill='toself',
                name='Background',
                marker_color='rgba(50,50,50,0.3)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add signal strength arc
            signal_angle = (recommendation_score / 100) * 360
            theta_signal = np.linspace(0, np.radians(signal_angle), 50)
            fig_gauge.add_trace(go.Scatterpolar(
                r=[0.8]*50,
                theta=np.degrees(theta_signal),
                fill='toself',
                name='Signal Strength',
                marker_color='#00D9FF',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add center text annotation
            fig_gauge.add_annotation(
                x=0.5, y=0.5,
                xref='paper', yref='paper',
                text=f"<b>{recommendation_score:.0f}%</b><br><sub>Signal Strength</sub>",
                showarrow=False,
                font=dict(size=32, color='#00D9FF', family="Arial Black"),
                xanchor='center', yanchor='middle'
            )
            
            fig_gauge.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1]),
                    angularaxis=dict(visible=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=350,
                template='plotly_dark',
                font=dict(family="Arial, sans-serif", size=14, color='#CCCCCC'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=30, r=30, t=30, b=30),
                showlegend=False
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Signal breakdown bar chart
            signals_data = {
                'Signal Type': ['Bullish', 'Bearish'],
                'Count': [signals_bullish, signals_bearish],
                'Color': ['#00FF00', '#FF0000']
            }
            
            fig_signals = go.Figure()
            fig_signals.add_trace(go.Bar(
                x=signals_data['Signal Type'],
                y=signals_data['Count'],
                marker_color=signals_data['Color'],
                text=signals_data['Count'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                showlegend=False
            ))
            fig_signals.update_layout(
                title="Trading Signals Breakdown",
                xaxis_title="Signal Type",
                yaxis_title="Count",
                height=300,
                template='plotly_dark',
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100,100,100,0.3)'),
                margin=dict(l=60, r=30, t=60, b=60),
                font=dict(family="Arial, sans-serif", size=12, color='#CCCCCC')
            )
            st.plotly_chart(fig_signals, use_container_width=True)
            
            # Detailed signal indicators table
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📌 Individual Signal Status")
                signal_status_data = []
                
                signal_status_data.append({
                    'Indicator': 'RSI Momentum',
                    'Value': f"{current_rsi:.2f}",
                    'Status': '🟢 Bullish' if current_rsi < 50 else '🔴 Bearish',
                    'Interpretation': 'Oversold (<30)' if current_rsi < 30 else ('Overbought (>70)' if current_rsi > 70 else 'Neutral')
                })
                
                signal_status_data.append({
                    'Indicator': 'MACD Cross',
                    'Value': f"{current_macd:.4f}",
                    'Status': '🟢 Bullish' if current_macd > current_signal else '🔴 Bearish',
                    'Interpretation': f"Signal: {current_signal:.4f}"
                })
                
                bb_pos = "Upper Zone" if bb_position > 80 else ("Lower Zone" if bb_position < 20 else "Mid Zone")
                signal_status_data.append({
                    'Indicator': 'Bollinger Band',
                    'Value': f"{bb_position:.1f}%",
                    'Status': '🟡 Neutral' if 20 <= bb_position <= 80 else ('🔴 Bearish' if bb_position > 80 else '🟢 Bullish'),
                    'Interpretation': bb_pos
                })
                
                signal_status_data.append({
                    'Indicator': 'ML Prediction',
                    'Value': f"${predicted_next_price:.2f}",
                    'Status': '🟢 Bullish' if predicted_next_price > current_price else '🔴 Bearish',
                    'Interpretation': f"vs Current: ${current_price:.2f}"
                })
                
                signal_df = pd.DataFrame(signal_status_data)
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🎯 Key Levels")
                st.metric("Resistance", f"${resistance:.2f}")
                st.metric("Current", f"${current_price:.2f}")
                st.metric("Support", f"${support:.2f}")
                st.metric("ATR", f"${current_atr:.2f}")
            
            st.markdown("---")
            st.markdown("#### Signal Details")
            for detail in signal_details:
                st.write(detail)
            
            st.markdown("---")
            
            # ===== SECTION 1: Technical Signals & Indicators =====
            st.markdown("### 📊 Technical Signals & Indicators", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            # RSI Signal
            with col1:
                if current_rsi > 70:
                    signal_color = "🔴"
                    signal_text = "Overbought"
                elif current_rsi < 30:
                    signal_color = "🟢"
                    signal_text = "Oversold"
                else:
                    signal_color = "🟡"
                    signal_text = "Neutral"
                st.metric(f"RSI ({rsi_period}) {signal_color}", f"{current_rsi:.2f}", signal_text)
            
            # MACD Signal
            with col2:
                if current_macd > current_signal:
                    signal_color = "🟢"
                    signal_text = "Bullish"
                else:
                    signal_color = "🔴"
                    signal_text = "Bearish"
                st.metric(f"MACD {signal_color}", f"{current_macd:.4f}", signal_text)
            
            # Resistance
            with col3:
                dist_to_resistance = ((resistance - current_price) / current_price) * 100
                st.metric("📍 Resistance", f"${resistance:.2f}", f"+{dist_to_resistance:.1f}%")
            
            # Support
            with col4:
                dist_to_support = ((current_price - support) / current_price) * 100
                st.metric("📍 Support", f"${support:.2f}", f"-{dist_to_support:.1f}%")
            
            st.markdown("---")
            
            # ===== SECTION 2: Volatility & Price Position (NOW 4-COLUMN ALIGNED) =====
            st.markdown("### 📈 Volatility & Price Position", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                st.metric(f"Bollinger Band Position ({bb_period})", f"{bb_position:.1f}%", 
                         "Near Upper" if bb_position > 80 else ("Near Lower" if bb_position < 20 else "Mid-range"))
            with col2:
                st.metric(f"ATR ({atr_period})", f"${current_atr:.2f}", "Volatility Measure")
            with col3:
                suggested_stop_loss = current_price - (current_atr * 2)
                suggested_tp = current_price + (current_atr * 3)
                st.metric("Risk/Reward", f"{(suggested_tp - current_price) / (current_price - suggested_stop_loss):.2f}:1", 
                         f"SL: ${suggested_stop_loss:.2f} | TP: ${suggested_tp:.2f}")
            with col4:
                st.metric("", "")  # Empty column for alignment
            
            st.markdown("---")
            
            # ===== SECTION 3: Model Prediction Quality =====
            st.markdown("### 🤖 Model Prediction Quality", unsafe_allow_html=True)
            
            # Calculate accuracy metrics
            y_pred_test = model.predict(X_test)
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            correct_direction = np.sum(np.sign(y_pred_test - X_test[:, 0]) == np.sign(y_test - X_test[:, 0])) / len(y_test) * 100
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                st.metric("Win Rate (Direction)", f"{correct_direction:.1f}%", 
                         "% of correct up/down predictions")
            with col2:
                st.metric("MAPE", f"{mape:.2f}%", "Mean Absolute Percentage Error")
            with col3:
                prediction_confidence = (test_score * 100) if test_score > 0 else 0
                st.metric("Model Confidence", f"{prediction_confidence:.1f}%", "R² on test set")
            with col4:
                st.metric("", "")  # Empty column for alignment
        
        with tab5:
            st.markdown("<div class='subheader-style'>Full Dataset</div>", unsafe_allow_html=True)
            
            st.dataframe(
                df_clean[['Close', 'MA_Short', 'MA_Long', 'Target']].rename(columns={
                    'MA_Short': f'{ma_short}-Day MA',
                    'MA_Long': f'{ma_long}-Day MA',
                    'Target': 'Next Day Price'
                }),
                use_container_width=True
            )
            
            # Download button
            csv = df_clean.to_csv(index=True)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{ticker.upper()}_predictions.csv",
                mime="text/csv"
            )
        
        # Footer
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #888; font-size: 14px; margin-top: 40px;'>Made with ❤️ by Aladdin</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check your internet connection or try a different ticker symbol.")
