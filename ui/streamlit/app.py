import json
from datetime import datetime

import pandas as pd
import streamlit as st
from pathlib import Path

from services.news_signal import NewsSignalService
from backend.backtesting.engine import BacktestEngine, BacktestConfig, SlippageModel
from backend.risk.risk_manager import RiskEngine

st.set_page_config(page_title="Forex Bot — Control Panel")
st.title("Forex Bot — Control Panel")

# ------------------------
# Sidebar: controls (UI only)
# ------------------------
st.sidebar.header("System Controls")
system_on = st.sidebar.checkbox("System ON", value=True, help="Enable/disable the system (UI only)")
mode = st.sidebar.selectbox("Mode", ["Paper", "Live"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk settings (UI only)")
risk_per_trade_pct = st.sidebar.slider("Risk per trade (%)", 0.0, 5.0, 1.0, 0.1)
max_daily_loss_pct = st.sidebar.slider("Max daily loss (%)", 0.0, 20.0, 2.0, 0.1)
max_drawdown_pct = st.sidebar.slider("Max drawdown (%)", 0.0, 50.0, 20.0, 0.5)

st.sidebar.markdown("---")
news_lock = st.sidebar.checkbox("News lock (block new entries)", value=False)
kill_switch = st.sidebar.checkbox("Kill switch (manual)", value=False)

if st.sidebar.button("Apply settings"):
    # persist UI settings to disk (db/ui_state.json)
    cfg = {
        "system_on": bool(system_on),
        "mode": mode,
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "max_daily_loss_pct": float(max_daily_loss_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "news_lock": bool(news_lock),
        "kill_switch": bool(kill_switch),
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        with open("db/ui_state.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        st.sidebar.success("Settings saved (UI only)")
    except Exception as e:
        st.sidebar.error(f"Failed to save settings: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Emergency Stop"):
    st.sidebar.warning("Emergency stop (UI only): system set to OFF")
    system_on = False

st.write("### Overview")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("System ON", "Yes" if system_on else "No")
    st.metric("Mode", mode)
    st.metric("News lock", "On" if news_lock else "Off")
    st.metric("Kill switch", "ENGAGED" if kill_switch else "Off")

# ------------------------
# Sentiment dashboard
# ------------------------
st.write("---")
st.write("### Sentiment dashboard 🔎")
ns = NewsSignalService()
latest = ns.latest_snapshot()
if not latest:
    st.info("No news signal snapshots found. Run the News → Signal pipeline to produce snapshots under `db/news_signals/`.")
else:
    st.write(f"Snapshot: {latest.get('timestamp')}")
    by_currency = latest.get("by_currency", {})
    # display per-currency cards
    c1, c2 = st.columns(len(by_currency) or 1)
    for i, (cur, vals) in enumerate(by_currency.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            avg = vals.get("avg_score", 0.0)
            cnt = vals.get("count", 0)
            col.metric(f"{cur} Avg Sentiment", f"{avg:.2f}")
            col.write(f"sample size: {cnt}")

# ------------------------
# Backtest (simulated) controls & outputs
# ------------------------
st.write("---")
st.write("### Trade log & equity (simulation only)")
btn_run = st.button("Generate example backtest")

if btn_run:
    st.info("Running a short simulated backtest (UI-only simulation)")
    # create synthetic data (same approach as tests)
    import numpy as np
    from datetime import timedelta

    now = datetime.utcnow()
    idx_h = pd.date_range(start=now - timedelta(days=5), periods=200, freq="1H")
    close_h = 100.0 + np.linspace(0, 2, len(idx_h))
    open_h = close_h - 0.1
    high_h = close_h + 0.2
    low_h = close_h - 0.2
    vol = np.ones_like(close_h) * 100
    df1h = pd.DataFrame({"Open": open_h, "High": high_h, "Low": low_h, "Close": close_h, "Volume": vol}, index=idx_h)
    df4h = df1h["Close"].resample("4H").ohlc()
    df4h.rename(columns=str.capitalize, inplace=True)
    df4h["Open"] = df1h["Open"].resample("4H").first()
    df4h["High"] = df1h["High"].resample("4H").max()
    df4h["Low"] = df1h["Low"].resample("4H").min()
    df4h["Volume"] = df1h["Volume"].resample("4H").sum()
    df4h = df4h[["Open", "High", "Low", "Close", "Volume"]]

    cfg = BacktestConfig(start_balance=10000.0, slippage=SlippageModel(spread_pips=0.0001, slippage_pct=0.0000))
    risk = RiskEngine(start_balance=10000.0, risk_per_trade_pct=risk_per_trade_pct / 100.0, max_daily_loss_pct=max_daily_loss_pct / 100.0, max_drawdown_pct=max_drawdown_pct / 100.0)
    engine = BacktestEngine(df1h, df4h, cfg=cfg, risk_engine=risk)
    # pass a news lock event if user enabled news_lock
    news_events = []
    if news_lock:
        news_events = [{"timestamp": df1h.index[-1].isoformat(), "mapped_currencies": ["USD"]}]
    res = engine.run(news_events=news_events, max_bars=150)

    # save last run result for convenience
    try:
        out_path = Path("db") / "backtest_results"
        out_path.mkdir(parents=True, exist_ok=True)
        fname = out_path / f"backtest_{datetime.utcnow().isoformat()}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"metrics": res.metrics, "trades": res.trades}, f, default=str)
    except Exception:
        pass

    # show trade log
    if res.trades:
        df_trades = pd.DataFrame(res.trades)
        st.write("#### Trade log")
        st.dataframe(df_trades)
    else:
        st.info("No trades executed in this simulated run.")

    # equity curve
    if not res.equity_curve.empty:
        st.write("#### Equity curve")
        # equity_curve index may not be a clean datetime index for streamlit, ensure it's a series
        eq = res.equity_curve.copy()
        eq.index = pd.to_datetime(eq.index)
        eq = eq.sort_index()
        st.line_chart(eq['balance'])
        st.write("Metrics:")
        st.json(res.metrics)

# Allow user to load last backtest file (if any)
from pathlib import Path
out_path = Path("db") / "backtest_results"
if out_path.exists():
    files = sorted(out_path.glob("backtest_*.jsonl"))
    if files:
        last = files[-1]
        if st.button("Load last backtest result"):
            try:
                j = json.loads(last.read_text())
                if j.get("trades"):
                    st.write("#### Last trade log")
                    st.dataframe(pd.DataFrame(j["trades"]))
                if j.get("metrics"):
                    st.write("#### Last metrics")
                    st.json(j["metrics"])
            except Exception as e:
                st.error(f"Failed to load last result: {e}")

st.write("---")
st.write("_Note: This control panel is UI-only and will never send live trade orders._")
