"""Simple event-driven backtesting engine.

Features:
- Step through 1H bar series and check multi-timeframe alignment (4H + 1H)
- Integrate `compute_bias` and `detect_setup` from strategy
- News lock simulation: skip entries when recent news maps to involved currencies
- Slippage & spread models (deterministic for repeatability)
- Uses `PaperTrader` + optional `RiskEngine` for execution and risk enforcement
- Produces equity curve and basic metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.strategy.bias import compute_bias, BiasConfig
from backend.strategy.setups import detect_setup, SetupConfig
from backend.execution.trading import PaperTrader, ExecutionConfig
from backend.execution.broker import Broker
from backend.risk.risk_manager import RiskEngine


@dataclass
class SlippageModel:
    spread_pips: float = 0.0001  # price units
    slippage_pct: float = 0.0000  # fraction of price to add/subtract deterministically

    def execution_price(self, current_price: float, direction: str) -> float:
        # For LONG, price paid is current + half spread + slippage
        half_spread = self.spread_pips / 2.0
        slippage = current_price * self.slippage_pct
        if direction == 'LONG':
            return current_price + half_spread + slippage
        else:
            return current_price - half_spread - slippage


from dataclasses import field

@dataclass
class BacktestConfig:
    start_balance: float = 10000.0
    slippage: SlippageModel = field(default_factory=SlippageModel)
    news_lock_minutes: int = 30
    freq: str = '1H'  # data frequency


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class BacktestEngine:
    def __init__(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, symbol: str = 'EURUSD', cfg: BacktestConfig = BacktestConfig(), risk_engine: Optional[RiskEngine] = None, classifier: Optional[Any] = None, confidence_threshold: float = 0.5):
        """Initialize engine with aligned 1H and 4H OHLC DataFrames (indexed by UTC datetime).

        classifier: optional object exposing `predict_proba(x: np.ndarray) -> float`.
        confidence_threshold: probability threshold under which signals are skipped.
        """
        # normalize columns to Title-case expected by downstream code (Open, High, Low, Close, Volume)
        def _normalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            mapping = {}
            for c in df.columns:
                if c.lower() in ("open", "high", "low", "close", "volume"):
                    mapping[c] = c.capitalize()
            if mapping:
                df = df.rename(columns=mapping)
            return df

        self.df_1h = _normalize(df_1h)
        self.df_4h = _normalize(df_4h)
        self.symbol = symbol
        self.cfg = cfg
        self.risk_engine = risk_engine
        self.broker = Broker(balance=cfg.start_balance)
        self.trader = PaperTrader(self.broker, ExecutionConfig(risk_per_trade_pct=(risk_engine.risk_per_trade_pct if risk_engine else 0.01) if risk_engine else ExecutionConfig().risk_per_trade_pct), risk_engine=risk_engine)
        self.trades: List[Dict[str, Any]] = []
        self.equity_records: List[Tuple[datetime, float]] = []
        self.classifier = classifier
        self.confidence_threshold = float(confidence_threshold)

    def _news_recent(self, news_events: List[Dict], check_time: datetime) -> bool:
        # news_events: list of {'timestamp': ISO str, 'mapped_currencies': ['USD','EUR']}
        if not news_events:
            return False
        cutoff = check_time - timedelta(minutes=self.cfg.news_lock_minutes)
        for ev in news_events:
            try:
                t = datetime.fromisoformat(ev.get('timestamp'))
            except Exception:
                continue
            if t >= cutoff and t <= check_time:
                # if event maps to either EUR or USD we lock
                if any(c in ev.get('mapped_currencies', []) for c in ['USD', 'EUR']):
                    return True
        return False

    def _record_equity(self, ts: datetime):
        self.equity_records.append((ts, float(self.broker.balance)))

    def run(self, news_events: Optional[List[Dict]] = None, max_bars: Optional[int] = None) -> BacktestResult:
        news_events = news_events or []
        index = list(self.df_1h.index)
        if max_bars:
            index = index[-max_bars:]
        # start equity record
        if index:
            self._record_equity(index[0])

        for ts in index:
            # skip if no corresponding 4H window (we'll pick last 4H up to ts)
            ohlc_4h = self.df_4h[self.df_4h.index <= ts]
            if ohlc_4h.empty:
                self._record_equity(ts)
                continue
            # take last segment for 4H
            last_4h = ohlc_4h.iloc[-40:]  # pass a window
            last_1h = self.df_1h[self.df_1h.index <= ts].iloc[-120:]

            # Multi-timeframe bias
            bias = compute_bias(last_4h, last_1h, BiasConfig())
            # no bias -> skip
            if bias == 'NEUTRAL':
                self._record_equity(ts)
                continue

            # news lock
            if self._news_recent(news_events, ts):
                self._record_equity(ts)
                continue

            # detect setup on 1H series (we can pass last_1h)
            setup_state, payload = detect_setup(last_1h, bias)
            if setup_state != 'SETUP_FOUND' or not payload:
                self._record_equity(ts)
                continue

            # simulate spread & slippage by adjusting execution price
            current_price = float(last_1h['Close'].iloc[-1])
            exec_price = self.cfg.slippage.execution_price(current_price, bias)
            # adjust payload entry to represent executed price
            payload_adj = payload.copy()
            payload_adj['entry'] = float(exec_price)

            # if a classifier is provided, obtain feature and check confidence
            take_trade = True
            if self.classifier is not None:
                # build feature vector consistent with trainer.features_from_payload
                from backend.ml.trainer import features_from_payload

                x = features_from_payload(payload_adj)
                try:
                    p = self.classifier.predict_proba(x)
                except Exception:
                    p = 0.0
                # attach confidence to payload for auditability
                payload_adj['model_confidence'] = float(p)
                if p < self.confidence_threshold:
                    take_trade = False

            if not take_trade:
                # annotate skipped trade
                self.trades.append({'timestamp': ts.isoformat(), 'action': 'SKIPPED_LOW_CONFIDENCE', 'payload': payload_adj, 'model_confidence': payload_adj.get('model_confidence', 0.0)})
            else:
                res = self.trader.process_setup(self.symbol, bias, payload_adj, exec_price)
                if res == 'TRADE_EXECUTED':
                    # record open position metadata
                    pos = self.trader.open_positions.get(self.symbol)
                    self.trades.append({'timestamp': ts.isoformat(), 'action': 'OPEN', 'payload': payload_adj, 'entry_price': pos.entry_price, 'side': pos.side, 'units': pos.units, 'stop': pos.stop})

            # simulate price tick for stop checking using last low/high as quick proxy
            pos = self.trader.open_positions.get(self.symbol)
            if pos:
                # check stop against current bar extremes
                bar_low = float(last_1h['Low'].iloc[-1])
                bar_high = float(last_1h['High'].iloc[-1])
                # for long: if low <= stop, trigger
                if pos.side == 'long' and bar_low <= pos.stop:
                    pnl = self.trader.on_price(self.symbol, pos.stop)
                    self.trades.append({'timestamp': ts.isoformat(), 'action': 'CLOSE', 'exit_price': pos.stop, 'pnl': pnl})
                if pos.side == 'short' and bar_high >= pos.stop:
                    pnl = self.trader.on_price(self.symbol, pos.stop)
                    self.trades.append({'timestamp': ts.isoformat(), 'action': 'CLOSE', 'exit_price': pos.stop, 'pnl': pnl})

            # record equity at this time
            self._record_equity(ts)

        # build equity curve
        eq_df = pd.DataFrame(self.equity_records, columns=['timestamp', 'balance']).set_index(pd.to_datetime(pd.Series([r[0] for r in self.equity_records])))
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

        metrics = self._compute_metrics(eq_df)
        return BacktestResult(equity_curve=eq_df, trades=self.trades, metrics=metrics)

    def _compute_metrics(self, eq: pd.DataFrame) -> Dict[str, Any]:
        balances = eq['balance'].values
        if len(balances) == 0:
            return {}
        returns = np.diff(balances) / balances[:-1]
        total_return = (balances[-1] / balances[0]) - 1.0
        # max drawdown
        peak = balances[0]
        max_dd = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / max(1e-9, peak)
            if dd > max_dd:
                max_dd = dd
        # simple Sharpe (annualized) assuming hourly bars
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 6.5 * 1)
        else:
            sharpe = 0.0
        # trade stats
        wins = sum(1 for t in self.trades if t.get('action') == 'CLOSE' and t.get('pnl', 0) and t.get('pnl') > 0)
        losses = sum(1 for t in self.trades if t.get('action') == 'CLOSE' and t.get('pnl', 0) and t.get('pnl') <= 0)
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_pnl = np.mean([t.get('pnl') for t in self.trades if t.get('action') == 'CLOSE']) if total_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe': float(sharpe),
            'total_trades': int(total_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(win_rate),
            'avg_pnl': float(avg_pnl),
        }
