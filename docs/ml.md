# Machine Learning Helpers

This section describes the lightweight ML helpers used to build an explainable classifier used by the backtesting engine.

## Model registry & persistence 🔒

Trained models are persisted under `db/models/<model_name>/` with automatic versioning. Each save produces:

- `model_v{n}.pkl` — the pickled model object
- `meta_v{n}.json` — human-readable metadata (name, version, timestamp, notes)

APIs:

- `backend.ml.registry.save_model(model, name, metadata=None)` — saves and returns integer version
- `backend.ml.registry.list_models(name=None)` — lists metadata entries (all or for a given name)
- `backend.ml.registry.load_model(name, version=None)` — loads model object (latest if version omitted)

This enables reproducible experiments and safe rollbacks to previous model versions. Store any additional information (training dataset ID, performance metrics) in the metadata when saving.

## Example usage

1. Train and save model:

```python
from backend.ml.trainer import build_dataset_from_trades, train_logistic
from backend.ml.registry import save_model

X, y = build_dataset_from_trades(trades)
model = train_logistic(X, y)
ver = save_model(model, 'my_classifier', metadata={'notes':'trained on sample backtest v1'})
print('Saved model version', ver)
```

2. Load latest model for integration in backtest:

```python
from backend.ml.registry import load_model
m = load_model('my_classifier')
engine = BacktestEngine(df1h, df4h, classifier=m, confidence_threshold=0.6)
```