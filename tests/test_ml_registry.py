from backend.ml.trainer import train_logistic
from backend.ml.registry import save_model, list_models, load_model
import numpy as np


def test_registry_save_load(tmp_path, monkeypatch):
    # isolate models dir
    monkeypatch.setenv('PYTEST_TMP', '1')
    monkeypatch.setattr('backend.ml.registry.MODELS_DIR', tmp_path / 'models')

    # synthetic tiny dataset
    X = np.array([[1.0, 1.0, 0.01, 0.0, 0.01], [0.0, 1.0, 0.05, 0.0, 0.01]])
    y = np.array([1.0, 0.0])
    model = train_logistic(X, y)
    ver = save_model(model, 'testmodel', metadata={'notes': 'unit test save'})
    assert ver == 1
    models = list_models('testmodel')
    assert len(models) == 1
    loaded = load_model('testmodel', version=1)
    assert hasattr(loaded, 'predict_proba')
    p = loaded.predict_proba(X[0])
    assert 0.0 <= p <= 1.0
