"""Setup detection logic

Supported setup types (kept to 2 simple, explainable ones):
- EMA_PULLBACK: price in a trending market pulls back to an EMA (e.g., 50 EMA) and shows a bounce
- BREAK_RETEST: price breaks a prior resistance/support and successfully retests it

Outputs: 'SETUP_FOUND' or 'NO_SETUP' and a small payload when found (setup_type, entry, stop, reason)

Rules are intentionally simple and deterministic for auditability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

SetupState = Literal['SETUP_FOUND', 'NO_SETUP']


@dataclass
class SetupConfig:
    ema_period: int = 50
    ema_slope_window: int = 3
    pullback_max_pct: float = 0.03  # max depth of pullback relative to recent high (3%)
    pullback_retest_bars: int = 5
    break_lookback: int = 40
    retest_max_drop_pct: float = 0.015  # allowed drop on retest (1.5%)
    confirm_bars: int = 2  # number of bars confirming the retest bounce


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _ema_slope(ema: pd.Series, window: int) -> float:
    if len(ema) < window + 1:
        return 0.0
    return ema.iloc[-1] - ema.iloc[-1 - window]


def detect_ema_pullback(df: pd.DataFrame, cfg: SetupConfig, direction: Literal['LONG', 'SHORT']) -> Tuple[bool, Optional[Dict]]:
    """Detect EMA pullback setup.

    LONG rules (mirrored for SHORT):
    - Trend: EMA slope > 0
    - Recent high over lookback = max close over lookback
    - Pullback: there exists a low within last `pullback_retest_bars` that is within pullback_max_pct of recent high (not deeper)
    - Price currently above EMA and showing a small bounce (last close > last open for LONG)
    """
    if df is None or len(df) < max(cfg.ema_period, cfg.pullback_retest_bars) + 3:
        return False, None

    close = df['Close'].astype(float)
    ema = _ema(close, cfg.ema_period)
    slope = _ema_slope(ema, cfg.ema_slope_window)

    if direction == 'LONG' and slope <= 0:
        return False, None
    if direction == 'SHORT' and slope >= 0:
        return False, None

    lookback = cfg.pullback_retest_bars
    recent_high = close.iloc[-lookback:].max()
    recent_low = close.iloc[-lookback:].min()
    c_last = close.iloc[-1]

    # measure pullback depth relative to recent high for LONG
    if direction == 'LONG':
        pullback_pct = (recent_high - recent_low) / max(1e-9, recent_high)
        if pullback_pct > cfg.pullback_max_pct:
            return False, None
        # require price above EMA and a small bullish candle
        if (c_last <= ema.iloc[-1]) or (df['Close'].iloc[-1] <= df['Open'].iloc[-1]):
            return False, None
        entry = c_last
        stop = recent_low - 0.0001 * recent_low
        payload = {'setup': 'EMA_PULLBACK', 'direction': 'LONG', 'entry': float(entry), 'stop': float(stop), 'pullback_pct': float(pullback_pct)}
        return True, payload

    # SHORT
    pullback_pct = (recent_low - recent_high) / max(1e-9, recent_low)  # negative
    pullback_pct = abs(pullback_pct)
    if pullback_pct > cfg.pullback_max_pct:
        return False, None
    if (c_last >= ema.iloc[-1]) or (df['Close'].iloc[-1] >= df['Open'].iloc[-1]):
        return False, None
    entry = c_last
    stop = recent_high + 0.0001 * recent_high
    payload = {'setup': 'EMA_PULLBACK', 'direction': 'SHORT', 'entry': float(entry), 'stop': float(stop), 'pullback_pct': float(pullback_pct)}
    return True, payload


def detect_break_retest(df: pd.DataFrame, cfg: SetupConfig, direction: Literal['LONG', 'SHORT']) -> Tuple[bool, Optional[Dict]]:
    """Detect break & retest setup.

    Returns payload with optional 'break_gap_pct' (magnitude of breakout vs historical level) when found.
    """
    """Detect break & retest setup.

    Simplified logic:
    - Find a breakout level as the max (LONG) or min (SHORT) of previous `break_lookback` bars excluding the final few bars
    - Verify there was a breakout (a close beyond the level), then a retest where price dipped back near the level but not beyond allowed drop
    - Confirm bounce: after retest, `confirm_bars` closes in the direction
    """
    n = len(df)
    if df is None or n < cfg.break_lookback + cfg.confirm_bars + 3:
        return False, None

    close = df['Close'].astype(float)
    lookback = cfg.break_lookback
    # Find breakout as a close that exceeds the max (LONG) / min (SHORT) of the previous `lookback` bars
    breakout_idx = None
    level = None

    # search for breakout in the recent window excluding the final confirm bars
    search_end = n - cfg.confirm_bars
    search_start = max(lookback, 0)

    if direction == 'LONG':
        breakout_idx = None
        level = None
        for i in range(search_start, search_end):
            prior_window = close.iloc[max(0, i - lookback):i]
            if len(prior_window) < 3:
                continue
            prior_level = prior_window.max()
            if close.iloc[i] > prior_level:
                breakout_idx = i
                level = float(prior_level)
                break
        if breakout_idx is None:
            return False, None
        # compute breakout gap relative to level
        break_gap_pct = (float(close.iloc[breakout_idx]) - level) / max(1e-9, level)
        # after breakout, within a short window expect a retest low near the level
        retest_window_end = min(n - 1, breakout_idx + cfg.confirm_bars + 3)
        segment = df.iloc[breakout_idx + 1:retest_window_end + 1]
        if segment.empty:
            return False, None
        retest_low = float(segment['Low'].min())
        drop_pct = (level - retest_low) / max(1e-9, level)
        if drop_pct > cfg.retest_max_drop_pct:
            return False, None
        # confirm bounce: last `confirm_bars` closes > level
        if not all(close.iloc[-cfg.confirm_bars:] > level):
            return False, None
        entry = float(close.iloc[-1])
        stop = float(retest_low - 0.0001 * retest_low)
        payload = {'setup': 'BREAK_RETEST', 'direction': 'LONG', 'entry': entry, 'stop': stop, 'level': level, 'break_gap_pct': float(break_gap_pct)}
        return True, payload

    # SHORT mirror
    for i in range(search_start, search_end):
        prior_window = close.iloc[max(0, i - lookback):i]
        if len(prior_window) < 3:
            continue
        prior_level = prior_window.min()
        if close.iloc[i] < prior_level:
            breakout_idx = i
            level = float(prior_level)
            break
    if breakout_idx is None:
        return False, None
    break_gap_pct = (level - float(close.iloc[breakout_idx])) / max(1e-9, level)
    segment = df.iloc[breakout_idx + 1: min(n - 1, breakout_idx + cfg.confirm_bars + 3) + 1]
    if segment.empty:
        return False, None
    retest_high = float(segment['High'].max())
    rise_pct = (retest_high - level) / max(1e-9, level)
    if rise_pct > cfg.retest_max_drop_pct:
        return False, None
    if not all(close.iloc[-cfg.confirm_bars:] < level):
        return False, None
    entry = float(close.iloc[-1])
    stop = float(retest_high + 0.0001 * retest_high)
    payload = {'setup': 'BREAK_RETEST', 'direction': 'SHORT', 'entry': entry, 'stop': stop, 'level': level, 'break_gap_pct': float(break_gap_pct)}
    return True, payload


def detect_setup(df: pd.DataFrame, direction: Literal['LONG', 'SHORT'], cfg: SetupConfig = SetupConfig()) -> Tuple[SetupState, Optional[Dict]]:
    """Top-level detection. Returns ('SETUP_FOUND', payload) or ('NO_SETUP', None).

    Order of checks: EMA pullback first (preferred in trend), then break & retest.
    """
    # Validate df
    if df is None or len(df) < 10:
        return 'NO_SETUP', None

    # compute recent pullback magnitude and if it's too deep, invalidate setups
    lookback = cfg.pullback_retest_bars
    if len(df) >= lookback:
        recent_high = df['Close'].astype(float).iloc[-lookback:].max()
        recent_low = df['Close'].astype(float).iloc[-lookback:].min()
        recent_pullback_pct = (recent_high - recent_low) / max(1e-9, recent_high)
        if recent_pullback_pct > cfg.pullback_max_pct * 1.5:
            # market had a deep pullback; avoid signalling setups
            return 'NO_SETUP', None

    # Evaluate both detections and prefer sensible tie-breaks
    ok_break, payload_break = detect_break_retest(df, cfg, direction)
    ok_ema, payload_ema = detect_ema_pullback(df, cfg, direction)

    if ok_break and not ok_ema:
        payload_break['method'] = 'BREAK_RETEST'
        return 'SETUP_FOUND', payload_break
    if ok_ema and not ok_break:
        payload_ema['method'] = 'EMA_PULLBACK'
        return 'SETUP_FOUND', payload_ema

    if ok_break and ok_ema:
        # tie-break: prefer BREAK_RETEST if breakout gap sizeable, else prefer EMA_PULLBACK
        gap = payload_break.get('break_gap_pct', 0.0)
        if gap >= 0.01:
            payload_break['method'] = 'BREAK_RETEST'
            return 'SETUP_FOUND', payload_break
        payload_ema['method'] = 'EMA_PULLBACK'
        return 'SETUP_FOUND', payload_ema

    return 'NO_SETUP', None
