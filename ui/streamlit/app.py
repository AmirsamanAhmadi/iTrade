import json
from datetime import datetime
import os, sys
from pathlib import Path

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
from services.mt5_service import MT5Service
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
        "BTC-USD", "ETH-USD", "XAUUSD=X", "GC=F", "EURUSD=X", "GBPUSD=X", 
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
    return DEFAULT_CONFIG

if 'cfg' not in st.session_state:
    st.session_state.cfg = load_config()

# Initialize Services
if 'mt5_service' not in st.session_state:
    st.session_state.mt5_service = MT5Service()
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

def _save_config():
    try:
        os.makedirs(os.path.dirname(UI_STATE_PATH), exist_ok=True)
        with open(UI_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.sidebar.error(f"Failed to save config: {e}")

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
    commodities = ["XAUUSD=X", "GC=F"]
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
st.write("---")
st.write("### 🎯 Trading Dashboard & Position Suggestions")
mds = None
mt5_service = None
try:
    from services.market_data import MarketDataService
    mds = MarketDataService()
    mt5_service = st.session_state.mt5_service
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
                _save_config()
                st.success("Settings saved!")

    with col_b:
        st.write("#### 🎯 Trading Signals & Position Suggestion")
        
        if analyze_btn or 'last_analysis' in st.session_state:
            try:
                # Fetch data for analysis
                end = datetime.utcnow().date().isoformat()
                start = (datetime.utcnow().date() - pd.Timedelta(days=7)).isoformat()
                
                df = None
                # Try MetaTrader 5 first for live data
                if mt5_service and mt5_service.connected:
                    try:
                        mt5_data = mt5_service.get_symbol_data(symbol, interval, 100)
                        if mt5_data:
                            df = pd.DataFrame(mt5_data)
                            df.index = pd.to_datetime(df['time'])
                            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                            st.success("📡 Using live MetaTrader 5 data!")
                    except Exception as mt5_error:
                        st.warning(f"MT5 data error: {mt5_error}, falling back to market data service")
                
                # Fallback to market data service
                if df is None:
                    df = mds.fetch_ohlc(symbol, interval, start=start, end=end)
                    if df is not None:
                        st.info("📊 Using market data service data")
                
                if df is not None and not df.empty:
                    # Store analysis for persistence
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
                    current_price = df['Close'].iloc[-1]
                    
                    # Technical indicators
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['MA50'] = df['Close'].rolling(window=50).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    df['MACD'] = calculate_macd(df['Close'])
                    
                    # Generate trading signal
                    signal = generate_trading_signal(df, current_price)
                    
                    # Calculate position size
                    position_size = (account_balance * risk_per_trade / 100)
                    stop_loss_price = current_price * (1 - stop_loss_pct/100)
                    take_profit_price = current_price * (1 + take_profit_pct/100)
                    
                    if 'Close' in df.columns:
                        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    else:
                        volume = 0
                    
                    # Display trading suggestion
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Current Price", f"${current_price:.4f}")
                        st.metric("📊 Signal", signal['action'], delta=None)
                        if signal['confidence']:
                            st.metric("🎯 Confidence", f"{signal['confidence']:.0f}%")
                        
                    with col2:
                        st.metric("💸 Stop Loss", f"${stop_loss_price:.4f}")
                        st.metric("🎯 Take Profit", f"${take_profit_price:.4f}")
                        st.metric("📈 Risk/Reward", f"{(take_profit_pct/stop_loss_pct):.1f}x")
                    
                    # Entry/Exit Price Suggestions
                    st.write("#### 🎯 Trading Recommendations")
                    
                    if signal['action'] == "BUY":
                        entry_price = current_price * (1 - 0.002)  # Enter slightly below current price
                        st.success(f"🟢 **BUY SIGNAL** - Entry at ${entry_price:.4f}")
                        st.info(f"📊 **Position Size**: ${position_size:.2f} ({risk_per_trade}% of account)")
                        st.info(f"🛡️ **Stop Loss**: ${stop_loss_price:.4f} (risk: ${position_size * stop_loss_pct/100:.2f})")
                        st.info(f"🎯 **Take Profit**: ${take_profit_price:.4f} (profit: ${position_size * take_profit_pct/100:.2f})")
                        
                    elif signal['action'] == "SELL":
                        entry_price = current_price * (1 + 0.002)  # Enter slightly above current price
                        st.error(f"🔴 **SELL SIGNAL** - Entry at ${entry_price:.4f}")
                        st.info(f"📊 **Position Size**: ${position_size:.2f} ({risk_per_trade}% of account)")
                        st.info(f"🛡️ **Stop Loss**: ${take_profit_price:.4f} (risk: ${position_size * stop_loss_pct/100:.2f})")
                        st.info(f"🎯 **Take Profit**: ${stop_loss_price:.4f} (profit: ${position_size * take_profit_pct/100:.2f})")
                        
                    else:
                        st.warning("🟡 **HOLD** - Wait for better setup")
                    
                    # Position Suggestion with multiple symbols
                    st.write("#### 📈 Recommended Positions to Monitor")
                    recommended_symbols = get_recommended_positions(all_symbols, df, current_price)
                    
                    for rec in recommended_symbols[:5]:  # Show top 5
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
                    st.error(f"❌ No data available for {symbol}")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

# Real-time price display and trading chart
    if mt5_service and mt5_service.connected:
        st.write("#### 📡 Live Trading - MetaTrader 5")
        
        # Get live price from MT5
        live_price_data = mt5_service.get_live_price(symbol)
        if live_price_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Bid", f"${live_price_data['bid']:.5f}")
            with col2:
                st.metric("💸 Ask", f"${live_price_data['ask']:.5f}")
            with col3:
                st.metric("📊 Spread", f"{live_price_data['spread']:.5f}")
            with col4:
                st.metric("⏰ Last Update", live_price_data['time'].strftime('%H:%M:%S'))
        else:
            st.warning("📡 Live price data unavailable")
    
    # Chart with current analysis
    analyze_triggered = st.session_state.get('analyze_triggered', False)
    if analyze_triggered or 'last_analysis' in st.session_state:
        with st.spinner(f"📊 Analyzing {symbol}..."):
            try:
                end = datetime.utcnow().date().isoformat()
                start = (datetime.utcnow().date() - pd.Timedelta(days=7)).isoformat()
                
                # Try MT5 first for live data
                df = None
                if mt5_service and mt5_service.connected:
                    try:
                        mt5_data = mt5_service.get_symbol_data(symbol, interval, 100)
                        if mt5_data:
                            df = pd.DataFrame(mt5_data)
                            df.index = pd.to_datetime(df['time'])
                            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                            st.success("📡 Using live MetaTrader 5 data!")
                    except Exception as mt5_error:
                        st.warning(f"MT5 data error: {mt5_error}")
                
                # Fallback to market data service
                if df is None:
                    df = mds.fetch_ohlc(symbol, interval, start=start, end=end)
                    if df is not None:
                        st.info("📊 Using market data service data")
                
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
                                    
                                    entry_level = alt.Chart(pd.DataFrame([{
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': entry_price,
                                        'label': 'ENTRY'
                                    }])).mark_rule(color='green', strokeWidth=3).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    sl_level = alt.Chart(pd.DataFrame([{
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': sl_price,
                                        'label': 'STOP LOSS'
                                    }])).mark_rule(color='red', strokeDash=[5, 5]).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    tp_level = alt.Chart(pd.DataFrame([{
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': tp_price,
                                        'label': 'TAKE PROFIT'
                                    }])).mark_rule(color='green', strokeDash=[3, 3]).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    chart_layers.extend([entry_level, sl_level, tp_level])
                                    
                                    # Display entry/exit info
                                    st.info(f"🟢 **BUY SETUP**")
                                    st.write(f"• **Entry Price**: ${entry_price:.5f}")
                                    st.write(f"• **Stop Loss**: ${sl_price:.5f} (risk: {sl_pct*100:.1f}%)")
                                    st.write(f"• **Take Profit**: ${tp_price:.5f} (target: {tp_pct*100:.1f}%)")
                                    
                                elif analysis.get('signal', {}).get('action') == 'SELL':
                                    entry_price = current_price * (1 + 0.002)  # Slightly above current
                                    sl_price = entry_price * (1 + sl_pct)
                                    tp_price = entry_price * (1 - tp_pct)
                                    
                                    entry_level = alt.Chart(pd.DataFrame([{
                                        'time': df_viz['time'].iloc[-1],
                                        'Close': entry_price,
                                        'label': 'EXIT'
                                    }])).mark_rule(color='red', strokeWidth=3).encode(
                                        y='Close:Q',
                                        size=alt.value(2)
                                    )
                                    
                                    chart_layers.append(entry_level)
                                    
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

                    # Recent price data
                    st.write("#### 📋 Recent Price Action")
                    display_df = df.tail(10).copy()
                    display_df['Change %'] = ((display_df['Close'] / display_df['Close'].shift(1) - 1) * 100).round(3)
                    display_df['RSI'] = df['RSI'].tail(10).round(1)
                    st.dataframe(display_df[['Close', 'Change %', 'RSI', 'Volume']].round(5))

            except Exception as e:
                st.error(f"❌ Failed to analyze {symbol}: {e}")



# ------------------------
# Advanced Backtesting Engine
# ------------------------
st.write("---")
st.write("### 🧪 Strategy Backtesting")

# Backtesting configuration
with st.expander("⚙️ Backtest Configuration", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        backtest_symbol = st.selectbox("Backtest Symbol", all_symbols, index=0)
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
                    
                    # Display results
                    st.success("✅ Backtest completed!")
                    
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
                                st.dataframe(signals_df, use_container_width=True)
                    
                    # Save results
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
# News Summary Section
# ------------------------
st.write("---")
st.write("### 📰 Market News Summary")

try:
    from services.news_service import NewsService
    news_service = NewsService()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("📰 Refresh News", type="secondary"):
            with st.spinner("Fetching latest market news..."):
                try:
                    # Try to get recent headlines for major symbols
                    major_symbols = ["AAPL", "GOOGL", "MSFT", "BTC-USD", "XAUUSD=X"][:5]  # Limit to avoid rate limits
                    headlines = news_service.fetch_headlines(tickers=major_symbols)
                    if headlines:
                        st.success(f"📰 Fetched {len(headlines)} latest headlines")
                    else:
                        st.info("📰 Using cached news data")
                except Exception as e:
                    st.warning(f"News fetch error: {e}")
    
    with col2:
        news_limit = st.number_input("Max headlines", min_value=5, max_value=50, value=20)
    
    with col3:
        news_keyword = st.text_input("🔍 Filter by keyword", placeholder="e.g., earnings, crypto")
    
    # Display news headlines
    try:
        from services.news_service import NEWS_DB
        news_items = _load_recent_news(NEWS_DB, days=3, limit=news_limit)
        
        if news_items:
            # Apply keyword filter if provided
            if news_keyword:
                news_items = [item for item in news_items if news_keyword.lower() in item.get('headline', '').lower()]
            
            if news_items:
                st.write(f"#### 📰 Latest {len(news_items)} Headlines")
                
                # Create a nicer display for news
                news_data = []
                for i, item in enumerate(news_items[:15]):  # Show top 15
                    headline = item.get('headline', 'No headline')
                    timestamp = item.get('timestamp', '')
                    source = item.get('source', 'Unknown')
                    mapped_currencies = item.get('mapped_currencies', [])
                    
                    # Format timestamp
                    try:
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%b %d, %I:%M %p')
                        else:
                            time_str = 'Unknown'
                    except:
                        time_str = timestamp[:16] if timestamp else 'Unknown'
                    
                    # Add signal indicators if keywords match trading signals
                    signal_indicators = []
                    headline_lower = headline.lower()
                    
                    if any(word in headline_lower for word in ['buy', 'bullish', 'rally', 'surge', 'jump', 'rise']):
                        signal_indicators.append('🟢')
                    if any(word in headline_lower for word in ['sell', 'bearish', 'drop', 'fall', 'decline', 'plunge']):
                        signal_indicators.append('🔴')
                    
                    signal_str = ' '.join(signal_indicators) if signal_indicators else '🟡'
                    
                    news_data.append({
                        '📰': signal_str,
                        '⏰': time_str,
                        '📊 Headline': headline[:100] + '...' if len(headline) > 100 else headline,
                        '📡 Source': source,
                        '💱 Related': ', '.join(mapped_currencies[:3]) if mapped_currencies else 'General'
                    })
                
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    st.dataframe(news_df, use_container_width=True)
                else:
                    st.info("No news items match the current filters")
                    
                # News summary stats
                buy_signals = len([item for item in news_items if any(word in item.get('headline', '').lower() for word in ['buy', 'bullish', 'rally', 'surge'])])
                sell_signals = len([item for item in news_items if any(word in item.get('headline', '').lower() for word in ['sell', 'bearish', 'drop', 'fall'])])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Bullish News", buy_signals)
                col2.metric("🔴 Bearish News", sell_signals) 
                col3.metric("🟡 Neutral News", len(news_items) - buy_signals - sell_signals)
                
            else:
                st.info(f"No headlines found matching '{news_keyword}' if specified")
        else:
            st.info("📰 No recent news available. Click 'Refresh News' to fetch latest headlines, or try adding specific stock symbols to your watchlist.")
            st.info("💡 **Tip**: News fetching requires internet connection and may need specific API configurations.")
            
    except ImportError:
        st.warning("📰 News service not available. Install required dependencies to enable news features.")
    except Exception as e:
        st.warning(f"📰 News summary unavailable: {e}")

except ImportError:
    st.info("📰 News features require additional services to be installed.")

st.write("---")
st.write("🎯 **Dashboard Ready** | Start adding symbols from the sidebar to begin your analysis!")
