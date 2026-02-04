"""Model registry for saving, versioning and loading trained models.

Files are stored under `db/models/<name>/` with naming convention:
- model_v{version}.pkl   (pickled model object)
- meta_v{version}.json   (metadata with timestamp, name, version, notes)

Provides simple APIs:
- save_model(model, name, metadata) -> version (int)
- list_models(name=None) -> list of metadata dicts
- load_model(name, version=None) -> model (latest if version not provided)
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MODELS_DIR = Path(__file__).resolve().parents[2] / "db" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_dir(name: str) -> Path:
    d = MODELS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_version(name: str) -> int:
    d = _model_dir(name)
    existing = [p.name for p in d.glob("model_v*.pkl")]
    versions = []
    for e in existing:
        try:
            v = int(e.split("_v")[-1].split(".")[0])
            versions.append(v)
        except Exception:
            continue
    return max(versions) + 1 if versions else 1


def save_model(model: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    meta = metadata.copy() if metadata else {}
    ver = _next_version(name)
    meta.update({"name": name, "version": ver, "saved_at": datetime.utcnow().isoformat()})
    d = _model_dir(name)
    model_path = d / f"model_v{ver}.pkl"
    meta_path = d / f"meta_v{ver}.json"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return ver


def list_models(name: Optional[str] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    dirs = [_model_dir(name)] if name else [p for p in MODELS_DIR.iterdir() if p.is_dir()]
    for d in dirs:
        for meta_f in sorted(d.glob("meta_v*.json")):
            try:
                with open(meta_f, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    results.append(m)
            except Exception:
                continue
    # sort by saved_at
    results = sorted(results, key=lambda x: x.get("saved_at", ""))
    return results


def load_model(name: str, version: Optional[int] = None) -> Any:
    d = _model_dir(name)
    if version is None:
        # find latest
        metas = list(sorted(d.glob("meta_v*.json")))
        if not metas:
            raise FileNotFoundError(f"No models found for {name}")
        last_meta = metas[-1]
        version = int(last_meta.name.split("_v")[-1].split(".")[0])
    model_path = d / f"model_v{version}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {name} v{version} not found")
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    return m
