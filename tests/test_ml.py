import numpy as np
from backend.ml.trainer import features_from_payload, train_logistic, LogisticModel, build_dataset_from_trades


def make_synthetic_trades(n=200):
    trades = []
    for i in range(n):
        # create payload with pullback_pct drawn from mixture
        if i < n // 2:
            pull = 0.01 + np.random.rand() * 0.005  # shallow pullbacks -> profitable
            pnl = 10.0
        else:
            pull = 0.04 + np.random.rand() * 0.01  # deep pullbacks -> losing
            pnl = -5.0
        payload = {'payload': {'method': 'EMA_PULLBACK', 'direction': 'LONG', 'entry': 100.0, 'stop': 99.0, 'pullback_pct': float(pull), 'method': 'EMA_PULLBACK'}}
        trades.append({'action': 'OPEN', **payload})
        trades.append({'action': 'CLOSE', 'pnl': float(pnl)})
    return trades


def test_feature_and_training():
    trades = make_synthetic_trades(200)
    X, y = build_dataset_from_trades(trades)
    assert X.shape[0] == len(trades) // 2
    # train
    model = train_logistic(X, y)
    # check predictions: average prob for first half > second half
    first_x = X[:len(X)//2]
    second_x = X[len(X)//2:]
    probs_first = [model.predict_proba(x) for x in first_x]
    probs_second = [model.predict_proba(x) for x in second_x]
    assert np.mean(probs_first) > np.mean(probs_second)
