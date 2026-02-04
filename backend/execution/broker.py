"""Broker simulator for backtesting and paper trading"""
from dataclasses import dataclass

@dataclass
class Broker:
    balance: float = 1000.0
    positions: dict = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

    def place_order(self, symbol: str, side: str, units: float, price: float):
        """Place an order in the simulator (simplified)."""
        order = {
            'symbol': symbol,
            'side': side,
            'units': units,
            'entry_price': price
        }
        self.positions[symbol] = order
        return order

    def close_position(self, symbol: str, exit_price: float):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        # simplified P&L calc
        pnl = (exit_price - pos['entry_price']) * pos['units']
        if pos['side'] == 'short':
            pnl = -pnl
        self.balance += pnl
        return pnl
