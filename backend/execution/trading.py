"""Paper trading engine: entry triggers, stop placement, position sizing, risk enforcement.

Key concepts:
- Entry trigger: accept the detected setup payload entry price and place a market order at current_price if hardware meets side/price logic
- Stop placement: use payload['stop']; support optional buffer
- Position sizing: size so that per-trade risk = balance * risk_per_trade_pct
- Risk cap: enforce max_total_risk_pct across open positions
- Paper execution: uses `Broker` simulator from `backend.execution.broker` to place/close positions

Note: units are in price units such that P&L = (exit - entry) * units (see Broker implementation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

from backend.execution.broker import Broker

TRADE_STATE = Literal['TRADE_EXECUTED', 'TRADE_REJECTED_RISK', 'NO_TRADE']


@dataclass
class ExecutionConfig:
    risk_per_trade_pct: float = 0.01  # 1% per trade
    max_total_risk_pct: float = 0.05  # max 5% of balance at risk across open positions
    max_positions: int = 5
    stop_buffer_pct: float = 0.0005  # small buffer added to stop (e.g., 0.05%)


@dataclass
class Position:
    symbol: str
    side: str
    units: float
    entry_price: float
    stop: float
    risk_amount: float
    payload: Dict


class PaperTrader:
    def __init__(self, broker: Broker, cfg: ExecutionConfig = ExecutionConfig(), risk_engine=None):
        self.broker = broker
        self.cfg = cfg
        self.open_positions: Dict[str, Position] = {}
        self.risk_engine = risk_engine

    def _compute_size(self, balance: float, entry: float, stop: float) -> float:
        # dollars (or price units) risk per unit = abs(entry - stop)
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0:
            return 0.0
        risk_budget = balance * self.cfg.risk_per_trade_pct
        units = risk_budget / per_unit_risk
        return float(units)

    def _total_risk_if_open(self, additional_risk: float = 0.0) -> float:
        current = sum(pos.risk_amount for pos in self.open_positions.values())
        return current + additional_risk

    def _balance(self) -> float:
        return float(self.broker.balance)

    def process_setup(self, symbol: str, direction: Literal['LONG', 'SHORT'], payload: Dict, current_price: float) -> TRADE_STATE:
        """Evaluate setup and possibly place a paper trade.

        Rules:
        - Use payload['entry'] as intended entry price and payload['stop'] for stop.
        - For LONG: require current_price >= entry*0.995 (allow tiny slippage) to accept market entry.
        - For SHORT: require current_price <= entry*1.005
        - Compute units based on risk% and check aggregate risk cap.
        - Respect max_positions.
        """
        if symbol in self.open_positions:
            return 'NO_TRADE'

        entry = float(payload.get('entry', current_price))
        stop = float(payload.get('stop', current_price))

        # basic acceptance logic
        if direction == 'LONG' and current_price < entry * 0.995:
            return 'NO_TRADE'
        if direction == 'SHORT' and current_price > entry * 1.005:
            return 'NO_TRADE'

        units = self._compute_size(self._balance(), entry, stop)
        if units <= 0 or len(self.open_positions) >= self.cfg.max_positions:
            return 'NO_TRADE'

        # risk amount in currency
        risk_amount = abs(entry - stop) * units
        # enforce max_total_risk_pct
        if self._total_risk_if_open(risk_amount) > self._balance() * self.cfg.max_total_risk_pct:
            return 'TRADE_REJECTED_RISK'

        # apply small buffer to stop to avoid immediate stop-outs
        if direction == 'LONG':
            stop_adj = stop - stop * self.cfg.stop_buffer_pct
        else:
            stop_adj = stop + stop * self.cfg.stop_buffer_pct

        # consult risk engine if present
        if self.risk_engine is not None:
            approved, reason = self.risk_engine.approve_trade(risk_amount)
            if not approved:
                return 'TRADE_REJECTED_RISK'

        # place order in broker
        side = 'long' if direction == 'LONG' else 'short'
        order = self.broker.place_order(symbol, side, units, current_price)

        pos = Position(symbol=symbol, side=side, units=units, entry_price=current_price, stop=stop_adj, risk_amount=risk_amount, payload=payload)
        self.open_positions[symbol] = pos

        # reserve risk in risk engine
        if self.risk_engine is not None:
            self.risk_engine.on_trade_open(risk_amount)

        return 'TRADE_EXECUTED'

    def on_price(self, symbol: str, price: float) -> Optional[float]:
        """Call on each price tick. If a stop triggers, close the position and return P&L."""
        pos = self.open_positions.get(symbol)
        if not pos:
            return None
        # stop check depending on side
        if pos.side == 'long' and price <= pos.stop:
            pnl = self.broker.close_position(symbol, price)
            self.open_positions.pop(symbol, None)
            if self.risk_engine is not None:
                self.risk_engine.on_trade_closed(pnl, reserved_risk_released=pos.risk_amount)
            return float(pnl)
        if pos.side == 'short' and price >= pos.stop:
            pnl = self.broker.close_position(symbol, price)
            self.open_positions.pop(symbol, None)
            if self.risk_engine is not None:
                self.risk_engine.on_trade_closed(pnl, reserved_risk_released=pos.risk_amount)
            return float(pnl)
        return None

    def close_all(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        results = {}
        for sym in list(self.open_positions.keys()):
            price = market_prices.get(sym)
            if price is None:
                continue
            pos = self.open_positions[sym]
            pnl = self.broker.close_position(sym, price)
            results[sym] = pnl
            self.open_positions.pop(sym, None)
            if self.risk_engine is not None:
                self.risk_engine.on_trade_closed(pnl, reserved_risk_released=pos.risk_amount)
        return results
