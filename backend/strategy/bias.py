"""Multi-timeframe bias helpers

Simple, explainable rules:
- 4H trend detection using short/long moving average crossover + slope
- detect ranging by measuring range vs mean price over lookback
- 1H confirmation requires price above/below 1H MA and short-term momentum
- Final bias: LONG / SHORT / NEUTRAL (NO ENTRIES implied for NEUTRAL)

All logic is intentionally simple and deterministic for explainability.
"""
from typing import Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd

BIAS = Literal['LONG', 'SHORT', 'NEUTRAL']


@dataclass
class BiasConfig:
    ma_short: int = 5
    ma_long: int = 20
    range_lookback: int = 20
    range_threshold: float = 0.02  # 2% range considered small -> ranging
    ma_slope_window: int = 3
    confirm_ma: int = 20  # 1H confirmation MA period


def _is_trending(df: pd.DataFrame, cfg: BiasConfig) -> BIAS:
    """Detect trend direction on the provided OHLC (assumed to be 4H timeframe).

    Returns 'LONG' or 'SHORT' or 'NEUTRAL' depending on MA crossover and slope.
    """
    if df is None or len(df) < cfg.ma_long + cfg.ma_slope_window:
        return 'NEUTRAL'

    close = df['Close'].astype(float)
    ma_s = close.rolling(cfg.ma_short).mean()
    ma_l = close.rolling(cfg.ma_long).mean()

    # latest values
    m_s = ma_s.iloc[-1]
    m_l = ma_l.iloc[-1]
    if np.isnan(m_s) or np.isnan(m_l):
        return 'NEUTRAL'

    # slope (difference over window)
    slope_s = m_s - ma_s.iloc[-1 - cfg.ma_slope_window]
    slope_l = m_l - ma_l.iloc[-1 - cfg.ma_slope_window]

    price = close.iloc[-1]

    # Conservative checks: short MA above long MA, both slopes in same direction, price confirming
    if (m_s > m_l) and (slope_s > 0) and (slope_l >= 0) and (price > m_s):
        return 'LONG'
    if (m_s < m_l) and (slope_s < 0) and (slope_l <= 0) and (price < m_s):
        return 'SHORT'

    return 'NEUTRAL'


def _is_ranging(df: pd.DataFrame, cfg: BiasConfig) -> bool:
    """Detect low-range market: if (max-min)/mean over lookback is small, consider ranging."""
    if df is None or len(df) < cfg.range_lookback:
        return False
    close = df['Close'].astype(float).iloc[-cfg.range_lookback:]
    rng = (close.max() - close.min()) / max(1e-9, close.mean())
    return rng < cfg.range_threshold


def _confirm_1h(ohlc_1h: pd.DataFrame, direction: BIAS, cfg: BiasConfig) -> bool:
    """Confirm 4H bias on 1H timeframe: require price relative to 1H MA and short-term momentum."""
    if ohlc_1h is None or len(ohlc_1h) < cfg.confirm_ma + 3:
        return False
    close = ohlc_1h['Close'].astype(float)
    ma = close.rolling(cfg.confirm_ma).mean()
    ma_last = ma.iloc[-1]
    if np.isnan(ma_last):
        return False

    # recent momentum: last delta of close
    delta = close.diff().iloc[-3:]
    momentum = delta.sum()

    price = close.iloc[-1]

    if direction == 'LONG':
        return (price > ma_last) and (momentum > 0)
    if direction == 'SHORT':
        return (price < ma_last) and (momentum < 0)
    return False


def compute_bias(ohlc_4h: pd.DataFrame, ohlc_1h: pd.DataFrame, cfg: BiasConfig = BiasConfig()) -> BIAS:
    """Compute final market bias (LONG / SHORT / NEUTRAL).

    Rules:
    - If 4H is ranging -> NEUTRAL
    - Else detect 4H trend using MA crossover and slope
    - Require 1H confirmation for final LONG/SHORT, otherwise NEUTRAL (no entries)
    """
    # Safety checks
    if ohlc_4h is None or len(ohlc_4h) < 10:
        return 'NEUTRAL'

    if _is_ranging(ohlc_4h, cfg):
        return 'NEUTRAL'

    trend = _is_trending(ohlc_4h, cfg)
    if trend == 'NEUTRAL':
        return 'NEUTRAL'

    # require 1H confirmation
    if _confirm_1h(ohlc_1h, trend, cfg):
        return trend

    # if no confirmation, be cautious
    return 'NEUTRAL'
