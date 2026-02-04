from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

class UIState(BaseModel):
    system_on: bool
    mode: Literal['Paper', 'Live']
    risk_per_trade_pct: float = Field(..., ge=0.0, le=5.0)
    max_daily_loss_pct: float = Field(..., ge=0.0, le=100.0)
    max_drawdown_pct: float = Field(..., ge=0.0, le=100.0)
    news_lock: bool
    kill_switch: bool
    updated_at: Optional[str] = None

    @validator('risk_per_trade_pct', 'max_daily_loss_pct', 'max_drawdown_pct')
    def two_decimal_places(cls, v):
        # normalize to two decimals
        return round(float(v), 4)
