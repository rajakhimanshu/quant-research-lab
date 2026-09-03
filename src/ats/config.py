from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    return _load_yaml(CONFIG_DIR / "settings.yaml")


def load_hypotheses() -> list[dict[str, Any]]:
    raw = _load_yaml(CONFIG_DIR / "hypotheses.yaml")
    items = raw.get("hypotheses", [])
    for item in items:
        why = (item.get("why") or "").strip()
        if not why:
            raise ValueError(f"Hypothesis {item.get('id')} has no why — refused.")
    return items
