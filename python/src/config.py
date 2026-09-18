"""Environment/runtime configuration loading (config.yaml + .env).

This is distinct from the *strategy design* (see `src.schema`), which is
serialized separately as strategy_config.json.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # python/
REPO_ROOT = PROJECT_ROOT.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Real environment variables win over .env.
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    artifacts_dir: Path
    schema_dir: Path

    def ensure(self) -> None:
        for p in (self.data_dir, self.artifacts_dir, self.schema_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    """Parsed config.yaml plus resolved paths."""

    raw: dict[str, Any]
    paths: Paths

    # -- convenience accessors -------------------------------------------------
    @property
    def market(self) -> dict[str, Any]:
        return self.raw.get("market", {})

    @property
    def costs(self) -> dict[str, Any]:
        return self.raw.get("costs", {})

    @property
    def taxes(self) -> dict[str, Any]:
        return self.raw.get("taxes", {})

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw.get("universe", {})

    @property
    def data_sources(self) -> dict[str, Any]:
        return self.raw.get("data_sources", {})

    @property
    def trading_days(self) -> int:
        return int(self.market.get("trading_days_per_year", 252))

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested key, e.g. cfg.get('costs.slippage_bps')."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


@lru_cache(maxsize=4)
def load_config(config_path: str | None = None) -> AppConfig:
    """Load config.yaml and .env once per path (cached)."""
    _load_dotenv(REPO_ROOT / ".env")

    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    p = raw.get("paths", {})
    paths = Paths(
        data_dir=_resolve(PROJECT_ROOT, p.get("data_dir", "data")),
        artifacts_dir=_resolve(PROJECT_ROOT, os.environ.get("ARTIFACTS_DIR") or p.get("artifacts_dir", "../artifacts")),
        schema_dir=_resolve(PROJECT_ROOT, p.get("schema_dir", "../schema")),
    )
    paths.ensure()

    _configure_logging(raw.get("logging", {}))
    return AppConfig(raw=raw, paths=paths)


def _configure_logging(log_cfg: dict[str, Any]) -> None:
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def env(key: str, default: str | None = None) -> str | None:
    _load_dotenv(REPO_ROOT / ".env")
    return os.environ.get(key, default)
