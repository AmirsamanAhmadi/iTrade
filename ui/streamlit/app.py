import json
from datetime import datetime
import os, sys
from pathlib import Path

# Load environment variables from .env file at startup
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[2] / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Ensure project root is on sys.path so `services` and `backend` imports work when
# Streamlit runs the app from a different working directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from services.news_signal import NewsSignalService
from backend.backtesting.engine import BacktestEngine, BacktestConfig, SlippageModel
from backend.risk.risk_manager import RiskEngine
from services.tv_service import TradingViewService

st.set_page_config(page_title="Market Data & Trading Dashboard", layout="wide")
st.title("📈 Market Data & Trading Dashboard")

# ------------------------
# Configuration Handling
# ------------------------
UI_STATE_PATH = "db/ui_state.json"

DEFAULT_CONFIG = {
    "market_symbols": [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        "BTC-USD", "ETH-USD", "GC=F", "EURUSD=X", "GBPUSD=X",
        "SPY", "QQQ", "IWM", "DIA"
    ],
    "default_symbol": "AAPL",
    "market_interval": "1h",
    "market_days_fetch": 14,
    "user_symbols": [],
    "matching_pairs": {},
    "news_limit": 50
}

def load_config():
    if os.path.exists(UI_STATE_PATH):
        try:
            with open(UI_STATE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                # merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **saved}
        except Exception as e:
            st.error(f"Error loading config: {e}")
    return DEFAULT_CONFIG  # Fixed orphaned try-except

def _save_config():
    try:
        os.makedirs(os.path.dirname(UI_STATE_PATH), exist_ok=True)
        with open(UI_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.sidebar.error(f"Failed to save config: {e}")

if 'cfg' not in st.session_state:
    st.session_state.cfg = load_config()

# Initialize Services
if 'tv_service' not in st.session_state:
    st.session_state.tv_service = TradingViewService()

# ------------------------
# Sidebar: Symbol Management
# ------------------------
st.sidebar.header("📊 Symbol Management")

# Add custom symbol
with st.sidebar.expander("Add Custom Symbol", expanded=False):
    new_symbol = st.text_input("Enter Symbol (e.g., AAPL, BTC-USD)")
    if st.button("Add Symbol") and new_symbol:
        current_symbols = st.session_state.cfg.get("user_symbols", [])
        if new_symbol not in current_symbols:
            current_symbols.append(new_symbol.upper())
            st.session_state.cfg["user_symbols"] = current_symbols
            _save_config()
            st.success(f"Added {new_symbol.upper()}")
            st.rerun()
        else:
            st.error("Symbol already exists")

# Display current symbols
st.sidebar.subheader("Your Symbols")
user_symbols = st.session_state.cfg.get("user_symbols", [])
if user_symbols:
    for i, symbol in enumerate(user_symbols):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"• {symbol}")
        with col2:
            if st.button("🗑️", key=f"del_{i}"):
                user_symbols.pop(i)
                st.session_state.cfg["user_symbols"] = user_symbols
                _save_config()
                st.rerun()
else:
    st.sidebar.write("No custom symbols added")

# ------------------------
# Backtesting Helper Functions
# ------------------------
def generate_trading_signals(df, strategy_type):
    """Generate trading signals based on selected strategy"""
    signals = []
    
    if strategy_type == "MA Crossover":
        df['MA_short'] = df['Close'].rolling(window=10).mean()
        df['MA_long'] = df['Close'].rolling(window=30).mean()
        
        for i in range(1, len(df)):
            if df['MA_short'].iloc[i-1] <= df['MA_long'].iloc[i-1] and df['MA_short'].iloc[i] > df['MA_long'].iloc[i]:
                signals.append({'time': df.index[i], 'signal': 'BUY', 'price': df['Close'].iloc[i]})
            elif df['MA_short'].iloc[i-1] >= df['MA_long'].iloc[i-1] and df['MA_short'].iloc[i] < df['MA_long'].iloc[i]:
                signals.append({'time': df.index[i], 'signal': 'SELL', 'price': df['Close'].iloc[i]})
    
    elif strategy_type == "Mean Reversion":
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['std'] = df['Close'].rolling(window=20).std()
        df['upper'] = df['MA20'] + (df['std'] * 2)
        df['lower'] = df['MA20'] - (df['std'] * 2)
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] <= df['lower'].iloc[i]:
                signals.append({'time': df.index[i], 'signal': 'BUY', 'price': df['Close'].iloc[i]})
            elif df['Close'].iloc[i] >= df['upper'].iloc[i]:
                signals.append({'time': df.index[i], 'signal': 'SELL', 'price': df['Close'].iloc[i]})
    
    elif strategy_type == "RSI Strategy":
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        for i in range(1, len(df)):
            if df['RSI'].iloc[i-1] >= 30 and df['RSI'].iloc[i] < 30:
                signals.append({'time': df.index[i], 'signal': 'BUY', 'price': df['Close'].iloc[i]})
            elif df['RSI'].iloc[i-1] <= 70 and df['RSI'].iloc[i] > 70:
                signals.append({'time': df.index[i], 'signal': 'SELL', 'price': df['Close'].iloc[i]})
    
    else:  # Breakout
        df['high20'] = df['High'].rolling(window=20).max()
        df['low20'] = df['Low'].rolling(window=20).min()
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['high20'].iloc[i-1]:
                signals.append({'time': df.index[i], 'signal': 'BUY', 'price': df['Close'].iloc[i]})
            elif df['Close'].iloc[i] < df['low20'].iloc[i-1]:
                signals.append({'time': df.index[i], 'signal': 'SELL', 'price': df['Close'].iloc[i]})
    
    return signals

def calculate_backtest_returns(df, signals, initial_balance, risk_per_trade):
    """Calculate returns from trading signals"""
    balance = initial_balance
    position = None
    trades = []
    equity_curve = []
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        # Record equity
        if position is None:
            equity_curve.append({'time': timestamp, 'balance': balance})
        else:
            current_value = balance + (row['Close'] - position['entry_price']) * position['shares']
            equity_curve.append({'time': timestamp, 'balance': current_value})
        
        # Check for signals
        for signal in signals:
            if abs((pd.to_datetime(signal['time']) - timestamp).total_seconds()) < 3600:  # Within 1 hour
                if position is None and signal['signal'] == 'BUY':
                    # Enter long position
                    position_size = balance * risk_per_trade
                    shares = position_size / signal['price']
                    position = {
                        'type': 'LONG',
                        'entry_price': signal['price'],
                        'shares': shares,
                        'entry_time': timestamp
                    }
                    
                elif position is not None and position['type'] == 'LONG' and signal['signal'] == 'SELL':
                    # Exit position
                    exit_value = signal['price'] * position['shares']
                    profit_loss = exit_value - (position['entry_price'] * position['shares'])
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': signal['price'],
                        'shares': position['shares'],
                        'profit_loss': profit_loss,
                        'return_pct': (profit_loss / (position['entry_price'] * position['shares'])) * 100
                    })
                    
                    balance = balance + profit_loss
                    position = None
    
    # Calculate metrics
    winning_trades = [t for t in trades if t['profit_loss'] > 0]
    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_balance': balance,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_return': ((balance - initial_balance) / initial_balance) * 100
    }

def _load_recent_news(news_db_path, days=3, limit=200):
    """Load recent news items from JSONL files"""
    items = []
    try:
        files = sorted(Path(news_db_path).glob('news_*.jsonl'), reverse=True)
        cutoff_date = (datetime.utcnow().date() - pd.Timedelta(days=days-1))
        for p in files:
            try:
                # parse file date from stem
                stem = p.stem
                date_str = stem.split('_')[-1]
                file_date = datetime.fromisoformat(date_str).date()
                if file_date < cutoff_date:
                    continue
            except Exception:
                pass
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        items.append(rec)
                        if len(items) >= limit:
                            return items
            except Exception:
                continue
    except Exception:
        return []
    return items

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd

def generate_trading_signal(df, current_price):
    """Generate trading signal based on technical indicators"""
    try:
        latest_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        latest_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        latest_ma20 = df['MA20'].iloc[-1] if 'MA20' in df.columns else current_price
        latest_ma50 = df['MA50'].iloc[-1] if 'MA50' in df.columns else current_price
        
        # Generate signal logic
        signals = []
        
        # RSI signals
        if latest_rsi < 30:
            signals.append({'action': 'BUY', 'reason': 'RSI Oversold', 'confidence': 75})
        elif latest_rsi > 70:
            signals.append({'action': 'SELL', 'reason': 'RSI Overbought', 'confidence': 75})
        elif 45 < latest_rsi < 55:
            signals.append({'action': 'HOLD', 'reason': 'RSI Neutral', 'confidence': 50})
        
        # MA crossover signals
        if latest_ma20 > latest_ma50 and df['MA20'].iloc[-2] <= df['MA50'].iloc[-2]:
            signals.append({'action': 'BUY', 'reason': 'MA Cross Up', 'confidence': 80})
        elif latest_ma20 < latest_ma50 and df['MA20'].iloc[-2] >= df['MA50'].iloc[-2]:
            signals.append({'action': 'SELL', 'reason': 'MA Cross Down', 'confidence': 80})
        
        # MACD signals
        if latest_macd > 0:
            signals.append({'action': 'BUY', 'reason': 'MACD Positive', 'confidence': 60})
        elif latest_macd < 0:
            signals.append({'action': 'SELL', 'reason': 'MACD Negative', 'confidence': 60})
        
        # Combine signals and determine best action
        if not signals:
            return {'action': 'HOLD', 'reason': 'No clear signal', 'confidence': 0}
        
        # Weight the signals and get the strongest
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        if buy_signals and not sell_signals:
            avg_confidence = sum(s['confidence'] for s in buy_signals) / len(buy_signals)
            return {'action': 'BUY', 'reason': ', '.join(s['reason'] for s in buy_signals), 'confidence': avg_confidence}
        elif sell_signals and not buy_signals:
            avg_confidence = sum(s['confidence'] for s in sell_signals) / len(sell_signals)
            return {'action': 'SELL', 'reason': ', '.join(s['reason'] for s in sell_signals), 'confidence': avg_confidence}
        else:
            return {'action': 'HOLD', 'reason': 'Mixed signals - wait for clarity', 'confidence': 50}
            
    except Exception as e:
        return {'action': 'HOLD', 'reason': f'Error: {str(e)}', 'confidence': 0}

def get_recommended_positions(symbols, current_df, current_price):
    """Get recommended positions across multiple symbols"""
    recommendations = []
    
    for symbol in symbols[:10]:  # Limit to 10 for performance
        try:
            # Quick analysis for each symbol
            if mds:
                end = datetime.utcnow().date().isoformat()
                start = (datetime.utcnow().date() - pd.Timedelta(days=3)).isoformat()
                symbol_df = mds.fetch_ohlc(symbol, "1h", start=start, end=end)
                
                if symbol_df is not None and not symbol_df.empty:
                    symbol_price = symbol_df['Close'].iloc[-1]
                    symbol_rsi = calculate_rsi(symbol_df['Close'])
                    latest_rsi = symbol_rsi.iloc[-1] if not symbol_rsi.empty else 50
                    
                    # Simple recommendation logic
                    if latest_rsi < 35:
                        signal_text = "BUY"
                        entry_price = symbol_price * 0.998  # Slightly below market
                        stop_loss = symbol_price * 0.98  # 2% SL
                    elif latest_rsi > 65:
                        signal_text = "SELL"
                        entry_price = symbol_price * 1.002  # Slightly above market
                        stop_loss = symbol_price * 1.02  # 2% SL
                    else:
                        signal_text = "HOLD"
                        entry_price = symbol_price
                        stop_loss = symbol_price * 0.98
                    
                    recommendations.append({
                        'symbol': symbol,
                        'price': symbol_price,
                        'signal': signal_text,
                        'entry': entry_price,
                        'stop_loss': stop_loss
                    })
        except Exception:
            continue
    
    # Sort by signal strength (prioritize BUY signals first)
    buy_signals = [r for r in recommendations if r['signal'] == 'BUY']
    other_signals = [r for r in recommendations if r['signal'] != 'BUY']
    
    return buy_signals + other_signals

# ------------------------
# Main View
# ------------------------
st.write("### 🌟 Market Overview")
st.info("👈 Use the sidebar to add custom symbols for tracking!")

# ------------------------
# Market Overview Stats
# ------------------------
st.write("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Symbols", len(st.session_state.cfg.get("market_symbols", [])) + len(st.session_state.cfg.get("user_symbols", [])))
with col2:
    st.metric("Custom Symbols", len(st.session_state.cfg.get("user_symbols", [])))
with col3:
    st.metric("Default Interval", st.session_state.cfg.get("market_interval", "1h"))
with col4:
    st.metric("Data Range", f"{st.session_state.cfg.get('market_days_fetch', 14)} days")

# ------------------------
# Symbol Matching System
# ------------------------
def find_matching_pairs(selected_symbol, all_symbols):
    """Find potential matching pairs based on sector/category"""
    tech_stocks = ["AAPL", "GOOGL", "MSFT", "META", "NVDA", "NFLX", "TSLA"]
    crypto = ["BTC-USD", "ETH-USD"]
    commodities = ["GC=F"]
    forex = ["EURUSD=X", "GBPUSD=X"]
    etfs = ["SPY", "QQQ", "IWM", "DIA"]
    
    selected = selected_symbol.upper()
    if selected in tech_stocks:
        return [s for s in tech_stocks if s != selected and s in all_symbols]
    elif selected in crypto:
        return [s for s in crypto if s != selected and s in all_symbols]
    elif selected in commodities:
        return [s for s in commodities if s != selected and s in all_symbols]
    elif selected in forex:
        return [s for s in forex if s != selected and s in all_symbols]
    elif selected in etfs:
        return [s for s in etfs if s != selected and s in all_symbols]
    return []

# ------------------------
# Live Trading Dashboard
# ------------------------

# Shared data store for all sections
if 'trading_data' not in st.session_state:
    st.session_state.trading_data = {}
st.write("---")
st.write("### 🎯 Trading Dashboard & Position Suggestions")

mds = None
try:
    from services.market_data import MarketDataService
    mds = MarketDataService()
except Exception as e:
    st.warning(f"Market data service not available: {e}")

if mds:
    # Combine default and user symbols
    all_symbols = st.session_state.cfg.get("market_symbols", []) + st.session_state.cfg.get("user_symbols", [])
    all_symbols = list(set(all_symbols))  # Remove duplicates
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.subheader("🔍 Trading Setup")
        default_sym = st.session_state.cfg.get("default_symbol", "AAPL")
        if default_sym not in all_symbols:
            default_sym = all_symbols[0] if all_symbols else "AAPL"

        symbol = st.selectbox("Select Trading Symbol", all_symbols, index=all_symbols.index(default_sym) if default_sym in all_symbols else 0)

        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        default_int = st.session_state.cfg.get("market_interval", "1h")
        interval = st.selectbox("Timeframe", intervals, index=intervals.index(default_int) if default_int in intervals else 4)

        # Chart data source selection
        data_source_options = ["Market Data Service (Yahoo Finance)", "Simulated Data"]
        default_source = st.session_state.cfg.get("chart_data_source", "Market Data Service (Yahoo Finance)")
        chart_data_source = st.selectbox("📊 Chart Data Source", data_source_options, 
                                       index=data_source_options.index(default_source) if default_source in data_source_options else 0)

        # Trading configuration
        st.write("**⚙️ Trading Configuration**")
        account_balance = st.number_input("Account Balance ($)", min_value=100, max_value=1000000, value=10000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, 0.5)
        stop_loss_pct = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5)
        take_profit_pct = st.slider("Take Profit (%)", 1.0, 20.0, 5.0, 0.5)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Analyze & Get Signals", type="primary"):
                st.session_state.analyze_triggered = True
        with c2:
            if st.button("💾 Save Settings"):
                st.session_state.cfg["default_symbol"] = symbol
                st.session_state.cfg["market_interval"] = interval
                st.session_state.cfg["account_balance"] = account_balance
                st.session_state.cfg["risk_per_trade"] = risk_per_trade
                st.session_state.cfg["chart_data_source"] = chart_data_source
                _save_config()
                st.success("Settings saved!")

    with col_b:
        st.write("#### 🎯 Trading Signals & Position Suggestion")
        
        if st.session_state.get('analyze_triggered', False) or 'last_analysis' in st.session_state:
            try:
                # Fetch data for analysis
                end = datetime.utcnow().date().isoformat()
                start = (datetime.utcnow().date() - pd.Timedelta(days=7)).isoformat()
                # Use market data service
                df = mds.fetch_ohlc(symbol, interval, start=start, end=end)
                if df is not None:
                    st.info("📊 Using market data service data")
                
                if df is not None and not df.empty:
                    # Store analysis for persistence and connect to other sections
                    st.session_state.last_analysis = {
                        'symbol': symbol,
                        'df': df,
                        'account_balance': account_balance,
                        'risk_per_trade': risk_per_trade,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    # Generate trading signals and suggestions
                    latest_price = df['Close'].iloc[-1]
                    
                    # Update shared trading data
                    st.session_state.trading_data.update({
                        'current_symbol': symbol,
                        'current_price': latest_price,
                        'last_analysis': st.session_state.last_analysis,
                        'analysis_time': datetime.utcnow().isoformat()
                    })
                    
                    # Technical indicators
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['MA50'] = df['Close'].rolling(window=50).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    df['MACD'] = calculate_macd(df['Close'])
                    
                    # Generate trading signal
                    signal = generate_trading_signal(df, latest_price)
                    
                    # Calculate position size
                    position_size = (account_balance * risk_per_trade / 100)
                    stop_loss_price = latest_price * (1 - stop_loss_pct/100)
                    take_profit_price = latest_price * (1 + take_profit_pct/100)
                    
                    if 'Close' in df.columns:
                        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    else:
                        volume = 0
                    
                    # Display trading suggestion
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Current Price", f"${latest_price:.4f}")
                        st.metric("📊 Signal", signal['action'], delta=None)
                        if signal['confidence']:
                            st.metric("🎯 Confidence", f"{signal['confidence']:.0f}%")
                        
                    with col2:
                        st.metric("💸 Stop Loss", f"${stop_loss_price:.4f}")
                        st.metric("🎯 Take Profit", f"${take_profit_price:.4f}")
                        st.metric("📈 Risk/Reward", f"{(take_profit_pct/stop_loss_pct):.1f}x")
                    
                    # Entry/Exit Price Suggestions
                    st.write("#### 🎯 Trading Recommendations")
                    
                    # Calculate suggested holding period based on interval
                    interval_periods = {
                        '1m': 'hours',
                        '5m': 'hours', 
                        '15m': 'hours to 1 day',
                        '30m': 'hours to 1-2 days',
                        '1h': '1-3 days',
                        '4h': '2-5 days',
                        '1d': '5-15 days',
                        '1w': 'weeks to months'
                    }
                    suggested_period = interval_periods.get(interval, 'days')
                    
                    if signal['action'] == "BUY":
                        entry_price = latest_price * (1 - 0.002)  # Enter slightly below current price
                        price_change_needed = ((take_profit_price - entry_price) / entry_price * 100)
                        
                        st.success(f"🟢 **BUY SIGNAL** - Entry at ${entry_price:.4f}")
                        st.info(f"📊 **Position Size**: ${position_size:.2f} ({risk_per_trade}% of account)")
                        st.info(f"🛡️ **Stop Loss**: ${stop_loss_price:.4f} (risk: ${position_size * stop_loss_pct/100:.2f})")
                        st.info(f"🎯 **Take Profit**: ${take_profit_price:.4f} (profit: ${position_size * take_profit_pct/100:.2f})")
                        st.info(f"⏱️ **Expected Timeframe**: Monitor for {suggested_period} | Need {price_change_needed:.2f}% move to hit TP")
                        
                    elif signal['action'] == "SELL":
                        entry_price = latest_price * (1 + 0.002)  # Enter slightly above current price
                        price_change_needed = ((entry_price - take_profit_price) / entry_price * 100)
                        
                        st.error(f"🔴 **SELL SIGNAL** - Entry at ${entry_price:.4f}")
                        st.info(f"📊 **Position Size**: ${position_size:.2f} ({risk_per_trade}% of account)")
                        st.info(f"🛡️ **Stop Loss**: ${take_profit_price:.4f} (risk: ${position_size * stop_loss_pct/100:.2f})")
                        st.info(f"🎯 **Take Profit**: ${stop_loss_price:.4f} (profit: ${position_size * take_profit_pct/100:.2f})")
                        st.info(f"⏱️ **Expected Timeframe**: Monitor for {suggested_period} | Need {price_change_needed:.2f}% move to hit TP")
                        
                    else:
                        st.warning("🟡 **HOLD** - Wait for better setup")
                    
                    # Position Suggestion for selected symbol only
                    st.write("#### 📈 Recommended Positions to Monitor")
                    # Only analyze the SELECTED symbol
                    recommended_positions = get_recommended_positions([symbol], df, latest_price)
                    
                    if recommended_positions:
                        st.info(f"💡 Showing recommendation for selected symbol: {symbol}")
                        for rec in recommended_positions[:3]:  # Show top 3 for selected symbol
                            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
                            with rec_col1:
                                st.metric(rec['symbol'], f"${rec['price']:.4f}")
                            with rec_col2:
                                st.metric("Signal", rec['signal'])
                            with rec_col3:
                                st.metric("Entry", f"${rec['entry']:.4f}")
                            with rec_col4:
                                st.metric("SL", f"${rec['stop_loss']:.4f}")
                    else:
                        st.info(f"ℹ️ No clear signal for {symbol} - market in neutral zone")
                    
                else:
                    st.error(f"❌ No data available for {symbol}")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

    # Real-time price display and trading chart
    
    # Chart with current analysis
    analyze_triggered = st.session_state.get('analyze_triggered', False)
    if analyze_triggered or 'last_analysis' in st.session_state:
        with st.spinner(f"📊 Analyzing {symbol}..."):
            try:
                end = datetime.utcnow().date().isoformat()
                start = (datetime.utcnow().date() - pd.Timedelta(days=7)).isoformat()
                
                # Use selected data source
                if chart_data_source == "Market Data Service (Yahoo Finance)":
                    df = mds.fetch_ohlc(symbol, interval, start=start, end=end)
                    if df is not None:
                        st.success(f"📊 Using {chart_data_source}")
                else:
                    # Simulated data for testing
                    import numpy as np
                    np.random.seed(42)
                    dates = pd.date_range(start=start, periods=168, freq='H')  # 7 days hourly
                    base_price = df['Close'].iloc[-1] if df is not None else 100.0
                    prices = []
                    for i in range(168):
                        change = np.random.normal(0, base_price * 0.01)
                        prices.append(max(base_price + change, 1.0))
                        base_price = prices[-1]
                    
                    df = pd.DataFrame({
                        'Open': prices,
                        'High': [p * 1.01 for p in prices],
                        'Low': [p * 0.99 for p in prices],
                        'Close': prices,
                        'Volume': [10000] * 168
                    }, index=dates)
                    st.warning(f"📊 Using {chart_data_source}")
                
                # Track data source for display
                data_source = chart_data_source
                
                if df is not None and not df.empty:
                    # Calculate indicators
                    df['MA20'] = df['Close'].rolling(window=min(20, len(df))).mean()
                    df['MA50'] = df['Close'].rolling(window=min(50, len(df))).mean()
                    df['MA200'] = df['Close'].rolling(window=min(200, len(df))).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    
                    latest_price = df['Close'].iloc[-1]
                    latest_rsi = df['RSI'].iloc[-1] if not df['RSI'].empty else 50
                    
                    # Price metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("💰 Current Price", f"${latest_price:.5f}")
                    col2.metric("📊 RSI", f"{latest_rsi:.1f}")
                    col3.metric("📈 MA20", f"${df['MA20'].iloc[-1]:.5f}")
                    col4.metric("📉 MA50", f"${df['MA50'].iloc[-1]:.5f}")
                    
                    # Enhanced price chart
                    st.write("#### 📈 Price Chart with Entry/Exit Levels")
                    st.caption(f"📡 Data Source: {data_source} | Interval: {interval}")
                    
                    try:
                        import altair as alt
                        df_viz = df.reset_index()
                        time_col = df_viz.columns[0]
                        df_viz = df_viz.rename(columns={time_col: 'time'})

                        # Base chart
                        base = alt.Chart(df_viz).encode(
                            x=alt.X('time:T', title='Time'),
                            tooltip=[
                                alt.Tooltip('time:T', title='Time', format='%Y-%m-%d %H:%M'),
                                alt.Tooltip('Close:Q', title='Price', format='$,.5f')
                            ]
                        ).properties(height=400, width='container')

                        price_line = base.mark_line(color='blue', strokeWidth=2).encode(
                            y=alt.Y('Close:Q', title=f'Price ({symbol})', scale=alt.Scale(zero=False))
                        )

                        ma20_line = base.mark_line(strokeDash=[5, 5], color='orange').encode(y='MA20:Q')
                        ma50_line = base.mark_line(strokeDash=[3, 3], color='green').encode(y='MA50:Q')

                        chart_layers = [price_line, ma20_line, ma50_line]

                        # Add entry/exit suggestions based on current analysis
                        if 'last_analysis' in st.session_state:
                            analysis = st.session_state.last_analysis
                            if analysis.get('account_balance') and analysis.get('symbol') == symbol:
                                current_price = latest_price
                                risk_pct = analysis.get('risk_per_trade', 2.0) / 100
                                sl_pct = analysis.get('stop_loss_pct', 2.0) / 100
                                tp_pct = analysis.get('take_profit_pct', 5.0) / 100
                                
                                # Calculate entry/exit prices
                                if analysis.get('signal', {}).get('action') == 'BUY':
                                    entry_price = current_price * (1 - 0.002)  # Slightly below current
                                    sl_price = entry_price * (1 - sl_pct)
                                    tp_price = entry_price * (1 + tp_pct)
                                    
                                    # Calculate liquidation price
                                    liquidation_price = entry_price - (account_balance * sl_pct)
                                    
                                    entry_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': entry_price,
                                        'label': 'ENTRY'
                                    })).mark_rule(color='green', strokeWidth=3).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    sl_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': sl_price,
                                        'label': 'STOP LOSS'
                                    })).mark_rule(color='red', strokeDash=[5, 5]).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    tp_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': tp_price,
                                        'label': 'TAKE PROFIT'
                                    })).mark_rule(color='green', strokeDash=[3, 3]).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    liq_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': liquidation_price,
                                        'label': 'LIQUIDATION'
                                    })).mark_rule(color='yellow', strokeWidth=4).encode(
                                        y='Close:Q',
                                        size=alt.value(3)
                                    )
                                    
                                    chart_layers.extend([entry_level, sl_level, tp_level, liq_level])
                                    
                                    st.info(f"🟢 **BUY SETUP**")
                                    st.write(f"• **Entry Price**: ${entry_price:.5f}")
                                    st.write(f"• **Stop Loss**: ${sl_price:.5f} (risk: {sl_pct*100:.1f}%)")
                                    st.write(f"• **Take Profit**: ${tp_price:.5f} (target: {tp_pct*100:.1f}%)")
                                    st.write(f"• **Liquidation Price**: ${liquidation_price:.5f} (margin call level)")
                                    
                                elif analysis.get('signal', {}).get('action') == 'SELL':
                                    entry_price = current_price * (1 + 0.002)  # Slightly above current
                                    sl_price = entry_price * (1 + sl_pct)
                                    tp_price = entry_price * (1 - tp_pct)
                                    
                                    # Calculate liquidation price
                                    liquidation_price = entry_price + (account_balance * sl_pct)
                                    
                                    entry_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': entry_price,
                                        'label': 'EXIT'
                                    })).mark_rule(color='red', strokeWidth=3).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    liq_level = alt.Chart(pd.DataFrame({
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': liquidation_price,
                                        'label': 'LIQUIDATION'
                                    })).mark_rule(color='yellow', strokeWidth=4).encode(
                                        y='Close:Q',
                                        size=alt.value(3)
                                    )
                                    
                                    chart_layers.append(entry_level)
                                    chart_layers.append(liq_level)
                                    
                                    st.error(f"🔴 **SELL SETUP**")
                                    st.write(f"• **Exit Price**: ${entry_price:.5f}")
                                    st.write(f"• **Stop Loss**: ${tp_price:.5f}")
                                    st.write(f"• **Take Profit**: ${sl_price:.5f}")
                                    st.write(f"• **Liquidation Price**: ${liquidation_price:.5f} (margin call level)")
                                    
                                    st.error(f"🔴 **SELL SETUP**")
                                    st.write(f"• **Exit Price**: ${entry_price:.5f}")
                                    st.write(f"• **Stop Loss**: ${tp_price:.5f}")
                                    st.write(f"• **Take Profit**: ${sl_price:.5f}")

                        # Combine charts
                        chart = alt.layer(*chart_layers).interactive().properties(
                            title=f'📈 {symbol} - Trading Analysis ({interval})',
                            height=400, width='container'
                        )

                        st.altair_chart(chart, use_container_width=True)
                        
                        # Legend
                        st.markdown("""
                        **Chart Legend:**
                        - 🔵 **Price** - Current price action
                        - 🟠 **MA20** - Short-term trend
                        - 🟢 **MA50** - Medium-term trend
                        - 🟢 **Green Line** - Suggested Entry/Exit Price
                        - 🔴 **Red Dashed** - Stop Loss Level
                        - 🟢 **Green Dashed** - Take Profit Level
                        """)
                        
                    except Exception as e:
                        st.warning(f"Chart error: {e}")
                        st.line_chart(df[['Close', 'MA20', 'MA50']])

                    # Clean unified analysis display
                    st.write("#### 📊 Complete Technical Analysis")
                    st.caption(f"📡 Data Source: {data_source} | Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    
                    latest_price = df['Close'].iloc[-1]
                    latest_rsi = df['RSI'].iloc[-1] if not df['RSI'].empty else 50
                    
                    # Main metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Price", f"${latest_price:.5f}")
                    with col2:
                        st.metric("📊 RSI", f"{latest_rsi:.1f}", 
                                delta="Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral")
                    with col3:
                        trend = "UP" if latest_price > df['MA50'].iloc[-1] else "DOWN"
                        st.metric("📈 Trend", trend)
                    with col4:
                        change_pct = ((latest_price / df['Close'].iloc[0] - 1) * 100)
                        st.metric("📊 Change", f"{change_pct:+.2f}%")
                    
                    # Charts row
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.write("**📈 Technical Indicators**")
                        
                        # Calculate additional indicators
                        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                        bb_std = df['Close'].rolling(window=20).std()
                        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                        
                        # MACD Calculation
                        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                        df['MACD'] = exp1 - exp2
                        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                        
                        # Bollinger Bands
                        st.write("**📈 Bollinger Bands**")
                        bb_data = df[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']].tail(50).dropna()
                        if not bb_data.empty:
                            # Normalize to percentage for better scaling
                            bb_middle = bb_data['BB_Middle']
                            bb_normalized = pd.DataFrame({
                                'Price %': ((bb_data['Close'] - bb_middle) / bb_middle * 100),
                                'Upper Band %': ((bb_data['BB_Upper'] - bb_middle) / bb_middle * 100),
                                'Middle Band %': 0,
                                'Lower Band %': ((bb_data['BB_Lower'] - bb_middle) / bb_middle * 100)
                            })
                            st.line_chart(bb_normalized, use_container_width=True)
                            st.caption("📊 % deviation from middle band (±2 std dev)")
                        
                        # RSI Chart
                        rsi_df = df[['RSI']].copy().dropna()
                        if not rsi_df.empty:
                            st.write("**📊 RSI**")
                            st.line_chart(rsi_df.tail(50), use_container_width=True)
                    
                    with chart_col2:
                        st.write("**📉 MACD & Volume**")
                        
                        # Volume Chart
                        if 'Volume' in df.columns:
                            st.write("**📊 Volume**")
                            volume_data = df[['Volume']].tail(30)
                            st.bar_chart(volume_data, use_container_width=True)
                        
                        # MACD Signal Status
                        macd_data = df[['MACD', 'Signal_Line']].tail(50).dropna()
                        if not macd_data.empty:
                            current_macd = df['MACD'].iloc[-1]
                            current_signal = df['Signal_Line'].iloc[-1]
                            if current_macd > current_signal:
                                st.success("🟢 MACD above Signal - Bullish")
                            else:
                                st.error("🔴 MACD below Signal - Bearish")
                        
                        # BB Position indicator
                        bb_data = df[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']].tail(50).dropna()
                        if not bb_data.empty:
                            bb_position = (latest_price - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])
                            if bb_position > 0.8:
                                st.warning("⚠️ Price near upper band - Overbought")
                            elif bb_position < 0.2:
                                st.success("✅ Price near lower band - Oversold")
                            else:
                                st.info("📊 Price in middle range")
                    
                    # Volume Analysis with moving average
                    st.write("**📊 Volume Analysis**")
                    vol_col1, vol_col2, vol_col3 = st.columns(3)
                    
                    with vol_col1:
                        if 'Volume' in df.columns:
                            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                            volume_chart_data = df[['Volume', 'Volume_MA20']].tail(30).dropna()
                            if not volume_chart_data.empty:
                                st.line_chart(volume_chart_data, use_container_width=True)
                    
                    with vol_col2:
                        if 'Volume' in df.columns:
                            current_vol = df['Volume'].iloc[-1]
                            avg_vol = df['Volume'].tail(20).mean()
                            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
                            
                            st.metric("Current Volume", f"{current_vol:,.0f}")
                            st.metric("vs 20D Avg", f"{vol_ratio:.2f}x")
                            
                            if vol_ratio > 1.5:
                                st.success("🔥 High Volume")
                            elif vol_ratio < 0.5:
                                st.warning("📉 Low Volume")
                            else:
                                st.info("📊 Normal Volume")
                    
                    with vol_col3:
                        # Price Performance Metrics
                        price_change_1d = ((latest_price / df['Close'].iloc[-2] - 1) * 100) if len(df) > 1 else 0
                        price_change_5d = ((latest_price / df['Close'].iloc[-5] - 1) * 100) if len(df) > 5 else 0
                        price_change_20d = ((latest_price / df['Close'].iloc[-20] - 1) * 100) if len(df) > 20 else 0
                        
                        st.metric("1D Change", f"{price_change_1d:+.2f}%")
                        st.metric("5D Change", f"{price_change_5d:+.2f}%")
                        st.metric("20D Change", f"{price_change_20d:+.2f}%")
                    
                    # Add new chart: Price Change Distribution
                    st.write("**📊 Price Change Distribution (Multi-Timeframe)**")
                    st.caption("View price performance across different timeframes")
                    
                    try:
                        import altair as alt
                        
                        # Calculate price changes for different periods
                        periods = [1, 5, 10, 20]  # days
                        price_changes = []
                        
                        for period in periods:
                            if len(df) > period:
                                change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-period]) / df['Close'].iloc[-period]) * 100
                                price_changes.append({
                                    'period': f'{period}D',
                                    'change': change_pct,
                                    'color': 'green' if change_pct > 0 else 'red'
                                })
                        
                        if price_changes:
                            change_df = pd.DataFrame(price_changes)
                            
                            # Create bar chart
                            base = alt.Chart(change_df).encode(
                                x=alt.X('period:N', title='Period'),
                                y=alt.Y('change:Q', title='Change %'),
                                color=alt.Color('color:N', legend=None)
                            ).properties(height=200, width='container')
                            
                            bar_chart = base.mark_bar(size=40)
                            
                            # Add horizontal line at 0
                            rule = alt.Chart(pd.DataFrame({'x': ['0%']})).mark_rule(
                                color='gray', strokeDash=[2, 2], strokeWidth=1
                            ).encode(x='x')
                            
                            st.altair_chart(bar_chart + rule, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Price distribution chart error: {e}")
                    
                    # Trend Momentum Gauge
                    st.write("**📊 Trend Strength & Momentum**")
                    
                    momentum_col1, momentum_col2, momentum_col3 = st.columns(3)
                    
                    with momentum_col1:
                        # Calculate ADX-like trend strength
                        df['DM_Plus'] = df['High'].diff()
                        df['DM_Minus'] = -df['Low'].diff()
                        df['DM_Plus'] = df['DM_Plus'].where(df['DM_Plus'] > 0, 0)
                        df['DM_Minus'] = df['DM_Minus'].where(df['DM_Minus'] > 0, 0)
                        
                        # Simple trend strength based on price action
                        price_range = df['High'].tail(14).max() - df['Low'].tail(14).min()
                        if price_range > 0:
                            trend_strength = min(100, (abs(df['Close'].iloc[-1] - df['Close'].iloc[-14]) / price_range) * 100)
                        else:
                            trend_strength = 0
                        
                        st.metric("Trend Strength", f"{trend_strength:.0f}%")
                        
                        if trend_strength > 70:
                            st.success("💪 Strong Trend")
                        elif trend_strength > 40:
                            st.info("📊 Moderate Trend")
                        else:
                            st.warning("😴 Weak/No Trend")
                    
                    with momentum_col2:
                        # Price momentum (rate of change)
                        roc_5 = ((latest_price / df['Close'].iloc[-6] - 1) * 100) if len(df) > 5 else 0
                        roc_10 = ((latest_price / df['Close'].iloc[-11] - 1) * 100) if len(df) > 10 else 0
                        
                        st.metric("5-Period Momentum", f"{roc_5:+.2f}%")
                        st.metric("10-Period Momentum", f"{roc_10:+.2f}%")
                        
                        if roc_5 > 0 and roc_10 > 0:
                            st.success("📈 Bullish Momentum")
                        elif roc_5 < 0 and roc_10 < 0:
                            st.error("📉 Bearish Momentum")
                        else:
                            st.warning("⚖️ Mixed Signals")
                    
                    with momentum_col3:
                        # Volatility trend
                        recent_volatility = df['Close'].pct_change().tail(10).std() * 100
                        overall_volatility = df['Close'].pct_change().std() * 100
                        
                        st.metric("Recent Volatility", f"{recent_volatility:.2f}%")
                        st.metric("Overall Volatility", f"{overall_volatility:.2f}%")
                        
                        if recent_volatility > overall_volatility * 1.5:
                            st.warning("⚠️ Volatility Increasing")
                        elif recent_volatility < overall_volatility * 0.5:
                            st.info("✅ Volatility Decreasing")
                        else:
                            st.info("📊 Stable Volatility")
                    
                    # Candlestick Pattern Detection (Simple)
                    st.write("**🕯️ Recent Candlestick Patterns**")
                    
                    def detect_patterns(df):
                        patterns = []
                        if len(df) >= 3:
                            # Check for Doji
                            last_candle = df.iloc[-1]
                            body = abs(last_candle['Close'] - last_candle['Open'])
                            range_candle = last_candle['High'] - last_candle['Low']
                            if range_candle > 0 and body / range_candle < 0.1:
                                patterns.append("Doji - Indecision")
                            
                            # Check for Hammer
                            if last_candle['Low'] < last_candle[['Open', 'Close']].min():
                                lower_shadow = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
                                body_size = abs(last_candle['Close'] - last_candle['Open'])
                                if lower_shadow > body_size * 2:
                                    patterns.append("Hammer - Potential Reversal")
                            
                            # Check for Bullish Engulfing
                            if len(df) >= 2:
                                prev = df.iloc[-2]
                                curr = df.iloc[-1]
                                if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']:
                                    if curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']:
                                        patterns.append("Bullish Engulfing")
                        
                        return patterns
                    
                    patterns = detect_patterns(df.tail(5))
                    if patterns:
                        for pattern in patterns:
                            st.info(f"🔍 {pattern}")
                    else:
                        st.info("ℹ️ No clear patterns detected in recent candles")
                    
                    # Key levels
                    st.write("**🎯 Key Trading Levels**")
                    level_col1, level_col2, level_col3 = st.columns(3)
                    
                    with level_col1:
                        high_20d = df['High'].tail(20).max()
                        low_20d = df['Low'].tail(20).min()
                        st.metric("📈 20D High", f"${high_20d:.4f}")
                        st.metric("📉 20D Low", f"${low_20d:.4f}")
                    
                    with level_col2:
                        recent_high = df['High'].tail(10).max()
                        recent_low = df['Low'].tail(10).min()
                        st.metric("🔴 Resistance", f"${recent_high:.4f}")
                        st.metric("🟢 Support", f"${recent_low:.4f}")
                    
                    with level_col3:
                        volatility = df['Close'].pct_change().tail(20).std() * 100
                        st.metric("📊 Volatility", f"{volatility:.2f}%")
                        
                        if 'Volume' in df.columns:
                            current_volume = df['Volume'].iloc[-1]
                            st.metric("📊 Volume", f"{current_volume:,.0f}")

            except Exception as e:
                st.error(f"❌ Failed to analyze {symbol}: {e}")

# Simple Connection Status
st.write("---")

# Main Status
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 Trading Status")
    
    current_symbol = st.session_state.trading_data.get('current_symbol', '')
    if current_symbol:
        st.success(f"🎯 Symbol: {current_symbol}")
        current_price = st.session_state.trading_data.get('current_price', 0)
        st.info(f"💰 Price: ${current_price:.5f}")
        
        if st.session_state.trading_data.get('last_analysis'):
            analysis = st.session_state.trading_data['last_analysis']
            action = analysis.get('signal', {}).get('action', 'N/A')
            if action == 'BUY':
                st.success(f"📈 Signal: {action}")
            elif action == 'SELL':
                st.error(f"📉 Signal: {action}")
            else:
                st.warning(f"⏸️ Signal: {action}")
    else:
        st.info("🔍 Select symbol to begin")

with col2:
    st.subheader("🔗 System Status")
    
    # Status indicators
    chart_status = "✅" if current_symbol else "⚠️"
    analysis_status = "✅" if st.session_state.trading_data.get('last_analysis') else "⚠️"
    news_status = "✅" if current_symbol else "⚠️"
    backtest_status = "✅" if st.session_state.trading_data.get('backtest_results') else "⚠️"
    
    st.write(f"📊 Chart: {chart_status}")
    st.write(f"🔍 Analysis: {analysis_status}")
    st.write(f"📰 News: {news_status}")
    st.write(f"🧪 Backtest: {backtest_status}")
    
    # Overall status
    if all([chart_status == "✅", analysis_status == "✅", news_status == "✅"]):
        st.success("🎯 All Systems Ready!")
    else:
        st.info("🔗 Complete setup for full analysis")



# ------------------------
# Advanced Backtesting Engine
# ------------------------
st.write("---")
st.write("### 🧪 Strategy Backtesting")

# Backtesting configuration - connected to main trading data
with st.expander("⚙️ Backtest Configuration", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        # Use current symbol from main trading if available
        current_trade_symbol = st.session_state.trading_data.get('current_symbol', '')
        if current_trade_symbol and current_trade_symbol in all_symbols:
            default_backtest_symbol = current_trade_symbol
            default_index = all_symbols.index(current_trade_symbol)
        else:
            default_backtest_symbol = all_symbols[0] if all_symbols else "AAPL"
            default_index = 0
            
        backtest_symbol = st.selectbox("Backtest Symbol", all_symbols, index=default_index)
        backtest_days = st.number_input("Backtest Period (days)", min_value=7, max_value=365, value=30)
        initial_balance = st.number_input("Initial Balance", min_value=1000, max_value=1000000, value=10000)
        
    with col2:
        strategy_type = st.selectbox("Strategy Type", ["MA Crossover", "Mean Reversion", "Breakout", "RSI Strategy"])
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
        max_drawdown = st.slider("Max Drawdown (%)", 5.0, 50.0, 20.0, 1.0)

    # Advanced configuration
    st.write("**🎯 Starting Price Options**")
    use_custom_start_price = st.checkbox("Use Custom Starting Price", value=False, help="Override the first data point price with your specified starting price")
    
    col1, col2 = st.columns(2)
    with col1:
        if use_custom_start_price:
            custom_start_price = st.number_input(
                "Custom Starting Price", 
                min_value=0.01, 
                max_value=1000000.0, 
                value=100.0, 
                step=0.01,
                help="The price to start backtesting from instead of using the first data point price"
            )
        else:
            custom_start_price = None
            
    with col2:
        start_price_option = st.selectbox(
            "Price Strategy",
            ["First Data Point", "Current Price", "Custom Price"],
            index=0,
            help="How to determine the starting price for backtesting",
            disabled=use_custom_start_price
        )
    
    # Display starting price info
    if use_custom_start_price:
        st.info(f"🎯 Backtesting will start with your custom price: ${custom_start_price:.4f}")
    else:
        if start_price_option == "Current Price":
            st.info("📈 Backtesting will use current price as starting point")
        else:
            st.info("📊 Backtesting will use first data point price as starting point")

if st.button("🚀 Run Backtest", type="primary"):
    if mds:
        # Store current backtest symbol for other sections
        st.session_state.trading_data['backtest_symbol'] = backtest_symbol
        st.session_state.trading_data['backtest_results'] = None
        with st.spinner(f"Running backtest for {backtest_symbol}..."):
            try:
                # Fetch historical data
                end_date = datetime.utcnow().date().isoformat()
                start_date = (datetime.utcnow().date() - pd.Timedelta(days=backtest_days)).isoformat()
                
                df = mds.fetch_ohlc(backtest_symbol, "1h", start=start_date, end=end_date)
                if df is None or df.empty:
                    st.error(f"No data available for {backtest_symbol}")
                else:
                    # Apply starting price strategy
                    original_first_price = df['Close'].iloc[0]
                    
                    if use_custom_start_price:
                        # Override with custom price
                        price_multiplier = custom_start_price / original_first_price
                        df['Open'] = df['Open'] * price_multiplier
                        df['High'] = df['High'] * price_multiplier  
                        df['Low'] = df['Low'] * price_multiplier
                        df['Close'] = df['Close'] * price_multiplier
                        st.success(f"🎯 Starting price set to ${custom_start_price:.4f} (was ${original_first_price:.4f})")
                        
                    elif start_price_option == "Current Price":
                        current_price = df['Close'].iloc[-1]
                        price_multiplier = current_price / original_first_price
                        df['Open'] = df['Open'] * price_multiplier
                        df['High'] = df['High'] * price_multiplier
                        df['Low'] = df['Low'] * price_multiplier
                        df['Close'] = df['Close'] * price_multiplier
                        st.success(f"📈 Starting price set to current price: ${current_price:.4f} (was ${original_first_price:.4f})")
                    
                    else:  # First Data Point
                        st.info(f"📊 Using original starting price: ${original_first_price:.4f}")
                    
                    # Generate trading signals based on strategy
                    signals = generate_trading_signals(df, strategy_type)
                    
                     # Calculate returns
                    returns = calculate_backtest_returns(df, signals, initial_balance, risk_per_trade/100)
                     
                         # Track backtest data source
                    backtest_data_source = "Market Data Service (Yahoo Finance)"
                     
                     # Display results
                    st.success("✅ Backtest completed!")
                    st.caption(f"📡 Data Source: {backtest_data_source} | Symbol: {backtest_symbol} | Period: {backtest_days} days")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total_return = ((returns['final_balance'] / initial_balance) - 1) * 100
                    col1.metric("💰 Final Balance", f"${returns['final_balance']:.2f}")
                    col2.metric("📈 Total Return", f"{total_return:.2f}%")
                    col3.metric("📊 Total Trades", returns['total_trades'])
                    col4.metric("🎯 Win Rate", f"{returns['win_rate']:.1f}%")
                    
# Enhanced Equity Curve with Entry/Exit Points
                    st.write("#### 📈 Equity Curve with Entry/Exit Points")
                    if returns['equity_curve']:
                        equity_df = pd.DataFrame(returns['equity_curve'])
                        # Ensure proper datetime index for chart
                        if 'time' in equity_df.columns:
                            equity_df['time'] = pd.to_datetime(equity_df['time'])
                            equity_df = equity_df.set_index('time')
                        
                        # Create enhanced chart with entry/exit markers
                        try:
                            import altair as alt
                            
                            # Main equity line
                            equity_chart = alt.Chart(equity_df.reset_index()).mark_line(
                                color='blue', strokeWidth=2
                            ).encode(
                                x=alt.X('time:T', title='Time'),
                                y=alt.Y('balance:Q', title='Balance ($)', scale=alt.Scale(zero=False)),
                                tooltip=[alt.Tooltip('time:T', title='Time'), alt.Tooltip('balance:Q', title='Balance', format='$.2f')]
                            )
                            
                            # Add entry/exit points
                            if returns['trades']:
                                trade_points = []
                                for trade in returns['trades']:
                                    entry_time = pd.to_datetime(trade['entry_time'])
                                    exit_time = pd.to_datetime(trade['exit_time'])
                                    entry_balance = None
                                    exit_balance = None
                                    
                                    # Find balance at trade times
                                    for i, row in equity_df.iterrows():
                                        if abs((i - entry_time).total_seconds()) < 3600:
                                            entry_balance = row['balance']
                                        if abs((i - exit_time).total_seconds()) < 3600:
                                            exit_balance = row['balance']
                                    
                                    if entry_balance:
                                        trade_points.append({
                                            'time': entry_time, 
                                            'balance': entry_balance, 
                                            'type': 'Buy', 
                                            'price': trade['entry_price'],
                                            'profit_loss': trade['profit_loss']
                                        })
                                    if exit_balance:
                                        trade_points.append({
                                            'time': exit_time, 
                                            'balance': exit_balance, 
                                            'type': 'Sell', 
                                            'price': trade['exit_price'],
                                            'profit_loss': trade['profit_loss']
                                        })
                                
                                if trade_points:
                                    trades_df_chart = pd.DataFrame(trade_points)
                                    
                                    # Entry points
                                    buy_points = alt.Chart(trades_df_chart[trades_df_chart['type'] == 'Buy']).mark_circle(
                                        size=60, color='green', opacity=0.8
                                    ).encode(
                                        x='time:T', y='balance:Q',
                                        tooltip=[
                                            alt.Tooltip('time:T', title='Buy Time', format='%Y-%m-%d %H:%M'),
                                            alt.Tooltip('balance:Q', title='Balance at Buy', format='$.2f'),
                                            alt.Tooltip('price:Q', title='Buy Price', format='$,.4f'),
                                            alt.Tooltip('profit_loss:Q', title='P&L', format='$,.2f')
                                        ]
                                    )
                                    
                                    # Exit points
                                    sell_points = alt.Chart(trades_df_chart[trades_df_chart['type'] == 'Sell']).mark_circle(
                                        size=60, color='red', opacity=0.8
                                    ).encode(
                                        x='time:T', y='balance:Q',
                                        tooltip=[
                                            alt.Tooltip('time:T', title='Sell Time', format='%Y-%m-%d %H:%M'),
                                            alt.Tooltip('balance:Q', title='Balance at Sell', format='$.2f'),
                                            alt.Tooltip('price:Q', title='Sell Price', format='$,.4f'),
                                            alt.Tooltip('profit_loss:Q', title='P&L', format='$,.2f')
                                        ]
                                    )
                                    
                                    # Combine charts
                                    combined_chart = (equity_chart + buy_points + sell_points).properties(
                                        height=400, width='container',
                                        title=f'Equity Curve - {backtest_symbol} ({strategy_type})'
                                    ).interactive()
                                    
                                    st.altair_chart(combined_chart, use_container_width=True)
                                else:
                                    st.line_chart(equity_df[['balance']], use_container_width=True)
                            else:
                                st.line_chart(equity_df[['balance']], use_container_width=True)
                                
                        except Exception as e:
                            st.warning(f"Enhanced chart unavailable: {e}")
                            if 'balance' in equity_df.columns:
                                st.line_chart(equity_df[['balance']], use_container_width=True)
                    else:
                        st.warning("No equity curve data available")
                    
                    # Trade details
                    if returns['trades']:
                        st.write("#### 📋 Trade Details")
                        trades_df = pd.DataFrame(returns['trades'])
                        
                        # Format the dataframe for better display
                        if not trades_df.empty:
                            # Format datetime columns
                            for col in ['entry_time', 'exit_time']:
                                if col in trades_df.columns:
                                    trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime('%Y-%m-%d %H:%M')
                            
                            # Add duration column
                            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                                entry_times = pd.to_datetime(trades_df['entry_time'])
                                exit_times = pd.to_datetime(trades_df['exit_time'])
                                duration = exit_times - entry_times
                                trades_df['duration'] = duration.dt.total_seconds() / 3600  # in hours
                                trades_df['duration'] = trades_df['duration'].round(1).astype(str) + 'h'
                            
                            # Reorder columns for better readability
                            cols_order = ['entry_time', 'exit_time', 'duration', 'entry_price', 'exit_price', 'shares', 'profit_loss', 'return_pct']
                            available_cols = [col for col in cols_order if col in trades_df.columns]
                            trades_df = trades_df[available_cols]
                            
                            # Rename columns for clarity
                            trades_df = trades_df.rename(columns={
                                'entry_time': '🟢 Entry Time',
                                'exit_time': '🔴 Exit Time', 
                                'duration': '⏱️ Duration',
                                'entry_price': '💰 Entry Price',
                                'exit_price': '💸 Exit Price',
                                'shares': '📊 Shares',
                                'profit_loss': '📈 P&L',
                                'return_pct': '📊 Return %'
                            })
                            
                            # Format numeric columns
                            for col in ['💰 Entry Price', '💸 Exit Price', '📊 Shares']:
                                if col in trades_df.columns:
                                    trades_df[col] = trades_df[col].round(4)
                            
                            for col in ['📈 P&L', '📊 Return %']:
                                if col in trades_df.columns:
                                    trades_df[col] = trades_df[col].round(2)
                            
                            st.caption(f"📡 Data Source: Simulated Trading based on {backtest_data_source} historical data")
                            st.dataframe(trades_df, use_container_width=True)
                            
                            # Show signals summary
                            st.write("#### 📊 Trading Signals Summary")
                            buy_signals = [s for s in signals if s['signal'] == 'BUY']
                            sell_signals = [s for s in signals if s['signal'] == 'SELL']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("🟢 Buy Signals", len(buy_signals))
                            col2.metric("🔴 Sell Signals", len(sell_signals))
                            col3.metric("📊 Executed Trades", len(returns['trades']))
                            col4.metric("💰 Best Trade", f"${max([t['profit_loss'] for t in returns['trades']]) if returns['trades'] else 0:.2f}")
                            
                            # Show recent signals
                            if signals:
                                st.write("#### 🎯 Recent Signals")
                                recent_signals = signals[:10]  # Show last 10 signals
                                signals_df = pd.DataFrame(recent_signals)
                                signals_df['time'] = pd.to_datetime(signals_df['time']).dt.strftime('%Y-%m-%d %H:%M')
                                signals_df = signals_df.rename(columns={
                                    'time': '⏰ Time',
                                    'signal': '📊 Signal', 
                                    'price': '💰 Price'
                                })
                                st.caption("📡 Data Source: Generated from backtest strategy signals")
                                st.dataframe(signals_df, use_container_width=True)
                    
                        # Save results to session state for status update
                    st.session_state.trading_data['backtest_results'] = returns
                    st.session_state.trading_data['backtest_symbol'] = backtest_symbol
                    st.session_state.trading_data['backtest_timestamp'] = datetime.utcnow().isoformat()
                    
                    # Force page refresh to update status indicators
                    st.rerun()
                    
                    # Save results to file
                    try:
                        out_path = Path("db") / "backtest_results"
                        out_path.mkdir(parents=True, exist_ok=True)
                        results_data = {
                            "symbol": backtest_symbol,
                            "strategy": strategy_type,
                            "period_days": backtest_days,
                            "metrics": returns,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        filename = f"backtest_{backtest_symbol}_{strategy_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(out_path / filename, "w") as f:
                            json.dump(results_data, f, indent=2, default=str)
                        st.success(f"Results saved to {filename}")
                    except Exception as save_error:
                        st.warning(f"Could not save results: {save_error}")
                        
            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")
    else:
        st.error("❌ Market data service not available")

# Load previous results
st.write("---")
st.write("### 📁 Previous Backtest Results")
try:
    out_path = Path("db") / "backtest_results"
    if out_path.exists():
        result_files = sorted(out_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if result_files:
            selected_file = st.selectbox("Select a result file", result_files, format_func=lambda x: x.name)
            if st.button("📂 Load Selected Result"):
                try:
                    with open(selected_file, "r") as f:
                        data = json.load(f)
                    
                    st.write(f"**Symbol:** {data.get('symbol', 'N/A')}")
                    st.write(f"**Strategy:** {data.get('strategy', 'N/A')}")
                    st.write(f"**Date:** {data.get('timestamp', 'N/A')}")
                    
                    metrics = data.get('metrics', {})
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Final Balance", f"${metrics.get('final_balance', 0):.2f}")
                        col2.metric("Total Trades", metrics.get('total_trades', 0))
                        col3.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
                        col4.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        
                        # Show equity curve if available
                        if 'equity_curve' in metrics and metrics['equity_curve']:
                            st.line_chart(metrics['equity_curve'])
                except Exception as e:
                    st.error(f"Failed to load results: {e}")
        else:
            st.info("No previous backtest results found")
except Exception as e:
    st.info(f"Could not load previous results: {e}")



# ------------------------
# News Section
# ------------------------
st.write("---")
st.write("### 📰 Market News")

if st.session_state.trading_data.get('current_symbol'):
    try:
        from services.news_service import NewsService
        news_service = NewsService()
        
        current_symbol = st.session_state.trading_data.get('current_symbol', '')
        
        # Show available news sources
        st.write("#### 📡 Available News Sources")
        
        # Check which sources are configured
        source_status = []
        try:
            from services.news_service import FINVIZ_AVAILABLE
            if FINVIZ_AVAILABLE:
                source_status.append("✅ Finviz")
            else:
                source_status.append("⚠️ Finviz (install: pip install finvizfinance)")
        except:
            source_status.append("❌ Finviz")
        
        # Check all API keys
        api_sources = [
            ("Massive", news_service.MASSIVE_KEY, "MASSIVE_KEY"),
            ("NewsAPI", news_service.NEWSAPI_KEY, "NEWSAPI_KEY"),
            ("Alpha Vantage", news_service.ALPHA_VANTAGE_KEY, "ALPHA_VANTAGE_KEY"),
            ("GNews", news_service.GNEWS_KEY, "GNEWS_KEY"),
            ("Currents", news_service.CURRENTS_KEY, "CURRENTS_KEY"),
            ("NY Times", news_service.NYT_KEY, "NYT_KEY"),
            ("Guardian", news_service.GUARDIAN_KEY, "GUARDIAN_KEY"),
            ("Benzinga", news_service.BENZINGA_KEY, "BENZINGA_KEY"),
        ]
        
        for name, key, env_name in api_sources:
            if key:
                source_status.append(f"✅ {name}")
            else:
                source_status.append(f"⚠️ {name}")
        
        # Check RSS feed sources (no API key needed)
        rss_sources = []
        try:
            import feedparser
            rss_sources.append("✅ MarketWatch")
            rss_sources.append("✅ Yahoo Finance")
            rss_sources.append("✅ Investing.com")
        except ImportError:
            rss_sources.append("⚠️ RSS feeds (pip install feedparser)")
        
        # Social/Community sources
        social_status = []
        social_status.append("✅ StockTwits")
        social_status.append("✅ Reddit (WSB/Investing)")
        
        # Display all sources in expandable section
        with st.expander(f"📡 News Sources ({len([s for s in source_status if '✅' in s])} active)"):
            st.write("**Premium APIs:**")
            api_cols = st.columns(4)
            for i, status in enumerate(source_status[:8]):
                with api_cols[i % 4]:
                    st.caption(status)
            
            st.write("**RSS Feeds:**")
            for status in rss_sources:
                st.caption(status)
            
            st.write("**Social/Community:**")
            for status in social_status:
                st.caption(status)
            
            st.info("💡 Add API keys to .env file to activate more sources. See NEWS_API_SETUP.md")
        
        st.caption("📡 Data aggregated from all available sources")
        
        if st.button("📰 Refresh News", type="secondary"):
            with st.spinner("Fetching news for current symbol..."):
                # Focus on current symbol + related symbols
                filter_symbols = [current_symbol] + ["AAPL", "GOOGL", "MSFT", "BTC-USD"][:4]
                try:
                    headlines = news_service.fetch_headlines(tickers=filter_symbols)
                    if headlines:
                        st.success(f"📰 {len(headlines)} headlines fetched")
                except Exception as e:
                    st.warning(f"Could not fetch headlines: {e}")

        # Display symbol-specific news
        try:
            from services.news_service import NEWS_DB
            # Load more news items (increased from 10 to 100 to show more results)
            news_items = _load_recent_news(NEWS_DB, days=7, limit=100)
            
            if news_items:
                # Parse symbol to extract actual company/asset name
                def parse_symbol_to_keywords(symbol):
                    """Parse various symbol formats and return relevant keywords"""
                    symbol = symbol.upper().strip()
                    
                    # Handle exchange-prefixed symbols (e.g., NASDAQ-TSLA, NYSE-AAPL)
                    if '-' in symbol:
                        exchange, ticker = symbol.split('-', 1)
                        # Map exchanges to their markets
                        exchange = exchange.strip()
                        ticker = ticker.strip()
                    else:
                        exchange = None
                        ticker = symbol
                    
                    # Comprehensive symbol to keywords mapping
                    keyword_map = {
                        # Commodities
                        'GC=F': ['gold', 'xau', 'precious metal', 'bullion', 'commodity'],
                        'SI=F': ['silver', 'precious metal', 'commodity'],
                        'CL=F': ['oil', 'crude', 'petroleum', 'commodity', 'energy'],
                        'NG=F': ['natural gas', 'gas', 'energy', 'commodity'],
                        'PL=F': ['platinum', 'precious metal', 'commodity'],
                        # Crypto
                        'BTC-USD': ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'digital currency'],
                        'ETH-USD': ['ethereum', 'eth', 'crypto', 'cryptocurrency'],
                        'XRP-USD': ['ripple', 'xrp', 'crypto'],
                        'LTC-USD': ['litecoin', 'ltc', 'crypto'],
                        # Forex/Majors
                        'EURUSD=X': ['eur/usd', 'euro', 'eurusd', 'currency', 'forex'],
                        'GBPUSD=X': ['gbp/usd', 'pound', 'gbpusd', 'sterling', 'currency', 'forex'],
                        'USDJPY=X': ['usd/jpy', 'yen', 'japan', 'currency', 'forex'],
                        'AUDUSD=X': ['aud/usd', 'aussie', 'australia', 'currency', 'forex'],
                        'USDCAD=X': ['usd/cad', 'canada', 'loonie', 'currency', 'forex'],
                        'USDCHF=X': ['usd/chf', 'swiss', 'franc', 'currency', 'forex'],
                        'NZDUSD=X': ['nzd/usd', 'kiwi', 'new zealand', 'currency', 'forex'],
                        # Major Tech Stocks
                        'AAPL': ['apple', 'aapl', 'iphone', 'mac', 'technology'],
                        'GOOGL': ['google', 'alphabet', 'googl', 'search', 'technology'],
                        'MSFT': ['microsoft', 'msft', 'windows', 'cloud', 'technology'],
                        'TSLA': ['tesla', 'tsla', 'elon musk', 'ev', 'electric vehicle'],
                        'AMZN': ['amazon', 'amzn', 'e-commerce', 'cloud', 'aws'],
                        'META': ['meta', 'facebook', 'instagram', 'social media', 'technology'],
                        'NVDA': ['nvidia', 'nvda', 'gpu', 'ai', 'semiconductor'],
                        'NFLX': ['netflix', 'nflx', 'streaming', 'entertainment'],
                        'AMD': ['amd', 'semiconductor', 'cpu', 'technology'],
                        'INTC': ['intel', 'semiconductor', 'chip', 'technology'],
                        'CRM': ['salesforce', 'cloud', 'software'],
                        'ADBE': ['adobe', 'software', 'creative'],
                        # Other Major Stocks
                        'JPM': ['jpmorgan', 'bank', 'finance', 'jpm'],
                        'BAC': ['bank of america', 'bank', 'finance', 'bac'],
                        'WFC': ['wells fargo', 'bank', 'finance'],
                        'GS': ['goldman sachs', 'bank', 'investment', 'finance'],
                        'V': ['visa', 'payment', 'fintech', 'credit card'],
                        'MA': ['mastercard', 'payment', 'fintech', 'credit card'],
                        'DIS': ['disney', 'entertainment', 'streaming'],
                        'KO': ['coca-cola', 'beverage', 'consumer'],
                        'PEP': ['pepsico', 'beverage', 'consumer'],
                        'MCD': ['mcdonalds', 'fast food', 'restaurant'],
                        'SBUX': ['starbucks', 'coffee', 'restaurant'],
                        'NKE': ['nike', 'sportswear', 'apparel'],
                        'JNJ': ['johnson & johnson', 'pharma', 'healthcare'],
                        'PFE': ['pfizer', 'pharma', 'healthcare', 'vaccine'],
                        'MRNA': ['moderna', 'pharma', 'biotech', 'vaccine'],
                        'XOM': ['exxon', 'oil', 'energy'],
                        'CVX': ['chevron', 'oil', 'energy'],
                        'BA': ['boeing', 'aviation', 'aerospace'],
                        'GE': ['general electric', 'industrial'],
                        'F': ['ford', 'automotive', 'car'],
                        'GM': ['general motors', 'automotive', 'car'],
                        # ETFs
                        'SPY': ['spy', 's&p 500', 'sp500', 'etf', 'market'],
                        'QQQ': ['qqq', 'nasdaq', 'etf', 'technology'],
                        'IWM': ['iwm', 'russell', 'small cap', 'etf'],
                        'DIA': ['dia', 'dow jones', 'dow', 'etf'],
                        'VTI': ['vti', 'total market', 'etf'],
                        'VXUS': ['vxus', 'international', 'etf'],
                        'BND': ['bnd', 'bond', 'fixed income', 'etf'],
                        'ARKK': ['arkk', 'innovation', 'etf', 'cathie wood'],
                        # Indices
                        'SPX': ['s&p 500', 'sp500', 'market', 'index'],
                        'DJI': ['dow jones', 'dow', 'market', 'index'],
                        'IXIC': ['nasdaq', 'composite', 'market', 'index'],
                        'FTSE': ['ftse', 'uk', 'london', 'index'],
                        'DAX': ['dax', 'germany', 'europe', 'index'],
                        'N225': ['nikkei', 'japan', 'asia', 'index'],
                        'HSI': ['hang seng', 'hong kong', 'asia', 'index'],
                    }
                    
                    # Check direct match first
                    if ticker in keyword_map:
                        keywords = keyword_map[ticker]
                        # Add exchange context if available
                        if exchange:
                            keywords = keywords + [exchange.lower()]
                        return keywords
                    
                    # If no direct match, try to extract meaningful parts
                    # Remove common suffixes/prefixes
                    clean_symbol = ticker.replace('=X', '').replace('-USD', '').replace('.L', '')
                    
                    # Return ticker as fallback with exchange
                    if exchange:
                        return [ticker.lower(), exchange.lower(), clean_symbol.lower()]
                    else:
                        return [ticker.lower(), clean_symbol.lower()]
                
                # Get keywords for current symbol
                keywords = parse_symbol_to_keywords(current_symbol)
                
                # Add filter options
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    show_all_news = st.checkbox("Show all news (no keyword filter)", value=False)
                with col_filter2:
                    with_url_only = st.checkbox("Only show news with links", value=True)
                
                # Filter for current symbol using keywords (unless show_all_news is checked)
                symbol_news = []
                for item in news_items:
                    if show_all_news:
                        # Show all news, but prioritize with URLs if option checked
                        if with_url_only:
                            if item.get('url', ''):
                                symbol_news.append(item)
                        else:
                            symbol_news.append(item)
                    else:
                        # Filter by keywords
                        headline = item.get('headline', '').lower()
                        if any(keyword in headline for keyword in keywords):
                            if with_url_only:
                                if item.get('url', ''):
                                    symbol_news.append(item)
                            else:
                                symbol_news.append(item)
                
                # Sort: prioritize items with URLs
                symbol_news.sort(key=lambda x: 1 if x.get('url', '') else 0, reverse=True)
                
                # Show summary
                filter_type = "All news" if show_all_news else f"Keywords: {', '.join(keywords)}"
                url_filter = ", with links only" if with_url_only else ""
                st.write(f"**📊 Found {len(symbol_news)} news items for {current_symbol}**")
                st.caption(f"🔍 Filter: {filter_type}{url_filter}")
                
                if symbol_news:
                    st.write("#### 📰 Symbol News")
                    
                    # Show up to 10 items with an expander for more
                    display_count = min(10, len(symbol_news))
                    for i, item in enumerate(symbol_news[:display_count]):
                        headline = item.get('headline', 'No headline')
                        source = item.get('source', 'Unknown')
                        timestamp = item.get('timestamp', '')
                        url = item.get('url', '')
                        sentiment = item.get('sentiment', 'neutral')
                        sentiment_score = item.get('sentiment_score', 0)
                        
                        # Format sentiment display with colored text
                        if sentiment in ['positive', 'bullish']:
                            sentiment_emoji = '🟢'
                            sentiment_color = 'green'
                        elif sentiment in ['negative', 'bearish']:
                            sentiment_emoji = '🔴'
                            sentiment_color = 'red'
                        else:
                            sentiment_emoji = '⚪'
                            sentiment_color = 'gray'
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # Make headline clickable if URL available
                            if url:
                                st.markdown(f"[**{headline[:80]}**]({url}) <span style='color:{sentiment_color};font-weight:bold;'> {sentiment_emoji}</span>")
                            else:
                                st.markdown(f"**{headline[:80]}** <span style='color:{sentiment_color};font-weight:bold;'> {sentiment_emoji}</span>")
                            st.caption(f"📡 {source} | <span style='color:{sentiment_color};'>{sentiment.capitalize()}</span>")
                        with col2:
                            st.caption(timestamp[:16] if timestamp else 'N/A')
                    
                    # Show more in expander if there are more items
                    if len(symbol_news) > display_count:
                        with st.expander(f"📰 Show {len(symbol_news) - display_count} more news items"):
                            for item in symbol_news[display_count:]:
                                headline = item.get('headline', 'No headline')
                                source = item.get('source', 'Unknown')
                                url = item.get('url', '')
                                sentiment = item.get('sentiment', 'neutral')
                                
                                # Format sentiment display with colored text
                                if sentiment in ['positive', 'bullish']:
                                    sentiment_color = 'green'
                                elif sentiment in ['negative', 'bearish']:
                                    sentiment_color = 'red'
                                else:
                                    sentiment_color = 'gray'
                                
                                if url:
                                    st.markdown(f"• [{headline[:60]}...]({url}) - *{source}* <span style='color:{sentiment_color};font-weight:bold;'> | {sentiment.upper()}</span>")
                                else:
                                    st.markdown(f"• {headline[:60]}... - *{source}* <span style='color:{sentiment_color};font-weight:bold;'> | {sentiment.upper()}</span>")
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"📰 {current_symbol} News", len(symbol_news))
                    with col2:
                        sources = set(item.get('source', '').split(':')[0] for item in symbol_news)
                        st.metric("📡 Sources", len(sources))
                    with col3:
                        # Show sentiment if available
                        sentiments = [item.get('sentiment', 'neutral') for item in symbol_news if 'sentiment' in item]
                        if sentiments:
                            positive = sum(1 for s in sentiments if s in ['positive', 'bullish'])
                            negative = sum(1 for s in sentiments if s in ['negative', 'bearish'])
                            st.metric("📊 Sentiment", f"+{positive}/-{negative}")
                else:
                    st.info(f"No recent news for {current_symbol}")
                    st.caption(f"💡 Try searching with different keywords or check back later")
            else:
                st.warning("No news data available in database")
        except Exception as e:
            st.error(f"News display error: {e}")
    except Exception as e:
        st.error(f"News service error: {e}")
else:
    st.info("🔍 Select a symbol and run analysis to see market news")
    
st.write("---")
st.write("🎯 **Dashboard Ready** | Start adding symbols from the sidebar to begin your analysis!")
