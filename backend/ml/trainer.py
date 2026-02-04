"""Lightweight ML utilities: feature engineering + simple logistic classifier (numpy-based).

We implement a small logistic regression trainer using Newton-Raphson to avoid external deps.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def features_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    """Extract a numeric feature vector from a setup payload.

    Features (simple, explainable):
    - method: EMA_PULLBACK=1, BREAK_RETEST=0 (one numeric flag)
    - direction: LONG=1, SHORT=0
    - pullback_pct (0 if absent)
    - break_gap_pct (0 if absent)
    - risk_pct = abs(entry - stop)/entry if present else 0
    """
    method_flag = 1.0 if payload.get('method', payload.get('setup')) == 'EMA_PULLBACK' else 0.0
    direction_flag = 1.0 if payload.get('direction') == 'LONG' else 0.0
    pullback = float(payload.get('pullback_pct') or 0.0)
    break_gap = float(payload.get('break_gap_pct') or 0.0)
    entry = float(payload.get('entry') or 0.0)
    stop = float(payload.get('stop') or 0.0)
    risk_pct = 0.0
    if entry and stop:
        risk_pct = abs(entry - stop) / max(1e-9, entry)
    return np.array([method_flag, direction_flag, pullback, break_gap, risk_pct], dtype=float)


def pair_opens_closes(trades: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
    """Pair OPEN and CLOSE trade events to produce labeled examples (payload, pnl).

    Assumes trades are appended in chronological order with 'action' in {'OPEN','CLOSE'}.
    Returns list of (open_payload, pnl) tuples.
    """
    opens = []
    results = []
    for t in trades:
        a = t.get('action')
        if a == 'OPEN':
            opens.append(t)
        elif a == 'CLOSE':
            # match to last open (LIFO) if exists
            if opens:
                o = opens.pop(0)  # earliest open
                payload = o.get('payload') or {}
                pnl = float(t.get('pnl') or 0.0)
                results.append((payload, pnl))
    return results


@dataclass
class LogisticModel:
    weights: np.ndarray  # including intercept as last element

    def predict_proba(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        X = np.hstack([x, np.ones((x.shape[0], 1))])
        logits = X.dot(self.weights)
        proba = _sigmoid(logits)
        return float(proba.ravel()[0])


def train_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> LogisticModel:
    """Train logistic regression via Newton-Raphson (IRLS).
    X shape (n, m). Returns LogisticModel.
    """
    n, m = X.shape
    # add intercept column
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(m + 1)
    for _ in range(max_iter):
        logits = Xb.dot(w)
        p = _sigmoid(logits)
        # gradient
        g = Xb.T.dot(p - y)
        # Hessian
        R = p * (1 - p)
        # avoid singular H by adding small diag
        H = Xb.T.dot(Xb * R.reshape(-1, 1)) + np.eye(m + 1) * 1e-6
        # solve
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - delta
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return LogisticModel(weights=w)


def build_dataset_from_trades(trades: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    pairs = pair_opens_closes(trades)
    X = []
    y = []
    for payload, pnl in pairs:
        feat = features_from_payload(payload)
        X.append(feat)
        y.append(1 if pnl > 0 else 0)
    if not X:
        return np.zeros((0, 5)), np.zeros((0,))
    return np.vstack(X), np.array(y, dtype=float)


def save_model(model: LogisticModel, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> LogisticModel:
    with open(path, 'rb') as f:
        return pickle.load(f)
