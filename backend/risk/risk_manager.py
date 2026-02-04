from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class RiskEngine:
    """Simple, explainable risk engine that can approve/reject trades and track P&L.

    Features:
    - Enforce per-trade risk (pct of balance)
    - Track daily P&L and max daily loss cap
    - Track peak-to-trough drawdown
    - Track consecutive losses and enforce kill-switch
    - Placeholder for correlation blocking (future)
    """
    start_balance: float = 10000.0
    risk_per_trade_pct: float = 0.01  # 1%
    max_daily_loss_pct: float = 0.03  # 3% max loss per day
    max_drawdown_pct: float = 0.10    # 10% drawdown kills
    max_consecutive_losses: int = 5

    # runtime state
    day: date = field(default_factory=lambda: date.today())
    daily_pnl: float = 0.0
    peak_balance: float = field(init=False)
    current_balance: float = field(init=False)
    consecutive_losses: int = 0
    killed: bool = False
    kill_reason: Optional[str] = None
    reserved_risk: float = 0.0  # sum of potential risk reserved for open trades

    def __post_init__(self):
        self.peak_balance = self.start_balance
        self.current_balance = self.start_balance

    def reset_daily(self):
        self.day = date.today()
        self.daily_pnl = 0.0

    def update_balance(self, pnl: float):
        self.current_balance += pnl
        self.daily_pnl += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        # check drawdown
        drawdown = (self.peak_balance - self.current_balance) / max(1e-9, self.peak_balance)
        if drawdown >= self.max_drawdown_pct:
            self.killed = True
            self.kill_reason = f"drawdown {drawdown:.3f} >= {self.max_drawdown_pct}"
        # check daily loss
        if self.daily_pnl <= -abs(self.max_daily_loss_pct * self.peak_balance):
            self.killed = True
            self.kill_reason = f"daily loss {self.daily_pnl:.2f} <= -{self.max_daily_loss_pct * self.peak_balance:.2f}"

    def on_trade_open(self, reserved_risk: float) -> None:
        """Reserve risk when a trade is executed (conservative)."""
        self.reserved_risk += reserved_risk

    def release_reserved(self, reserved_risk: float) -> None:
        self.reserved_risk = max(0.0, self.reserved_risk - reserved_risk)

    def on_trade_closed(self, pnl: float, reserved_risk_released: float = 0.0):
        """Call when a trade is closed to update P&L and counters and release reserved risk."""
        # reset daily at day boundary
        if date.today() != self.day:
            self.reset_daily()
        # release reserved risk first
        self.release_reserved(reserved_risk_released)
        self.update_balance(pnl)
        # update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.killed = True
                self.kill_reason = f"consecutive losses {self.consecutive_losses} >= {self.max_consecutive_losses}"
        else:
            self.consecutive_losses = 0

    def approve_trade(self, proposed_risk_amount: float) -> (bool, Optional[str]):
        """Decide whether a proposed trade is allowed. Returns (approved, reason_if_rejected)."""
        if self.killed:
            return False, f"killed: {self.kill_reason}"
        # per-trade risk enforcement
        if proposed_risk_amount > self.current_balance * self.risk_per_trade_pct:
            return False, f"per-trade risk {proposed_risk_amount:.2f} > {self.risk_per_trade_pct*100:.2f}% of balance"
        # consider reserved risk plus proposed
        if (self.reserved_risk + proposed_risk_amount) > abs(self.max_daily_loss_pct * self.peak_balance):
            return False, f"would exceed available daily risk budget: reserved {self.reserved_risk:.2f} + proposed {proposed_risk_amount:.2f} > {self.max_daily_loss_pct * self.peak_balance:.2f}"
        # placeholder: correlation blocking could go here
        return True, None

    def manual_reset_kill(self):
        self.killed = False
        self.kill_reason = None
        self.consecutive_losses = 0
        self.reserved_risk = 0.0

    def status(self) -> dict:
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'daily_pnl': self.daily_pnl,
            'reserved_risk': self.reserved_risk,
            'consecutive_losses': self.consecutive_losses,
            'killed': self.killed,
            'kill_reason': self.kill_reason,
        }
