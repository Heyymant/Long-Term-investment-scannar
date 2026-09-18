"""Schema and artifact contract tests.

These protect the seam between the Python analytics and the Rust dashboard:
if Python stops emitting a file or renames a column, the dashboard silently
shows nothing. Better to fail here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.presets import get_preset, list_presets
from src.schema import SCHEMA_VERSION, StrategyConfig, export_json_schema


# -------------------------------------------------------------- schema
def test_all_presets_validate():
    for name in list_presets():
        cfg = get_preset(name)
        assert isinstance(cfg, StrategyConfig)
        assert cfg.schema_version == SCHEMA_VERSION
        assert cfg.active_sleeves()


def test_config_roundtrips_through_json(tmp_path):
    cfg = get_preset("full_composite")
    path = cfg.save(tmp_path / "strategy_config.json")
    loaded = StrategyConfig.load(path)
    assert loaded.config_hash() == cfg.config_hash()


def test_config_hash_ignores_name_but_tracks_parameters():
    a = get_preset("full_composite")
    b = get_preset("full_composite")
    b.name = "renamed"
    assert a.config_hash() == b.config_hash()

    b.sleeve_a.top_n += 1
    assert a.config_hash() != b.config_hash()


def test_unknown_fields_are_rejected():
    """extra='forbid' stops a typo'd parameter silently doing nothing."""
    with pytest.raises(Exception):
        StrategyConfig.model_validate({"schema_version": SCHEMA_VERSION, "typo_field": 1})


def test_allocations_must_not_exceed_one():
    with pytest.raises(Exception):
        StrategyConfig.model_validate({
            "allocation": {"sleeve_a": 0.8, "sleeve_b": 0.5, "sleeve_c": 0.1}
        })


def test_drawdown_thresholds_must_descend():
    with pytest.raises(Exception):
        StrategyConfig.model_validate({
            "risk": {"drawdown_thresholds": [-0.25, -0.15], "drawdown_exposures": [0.5, 0.75]}
        })


def test_json_schema_exports_for_rust(tmp_path):
    path = export_json_schema(tmp_path / "strategy_config.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["title"] == "StrategyConfig"
    assert "properties" in schema
    for key in ("sleeve_a", "sleeve_b", "sleeve_c", "risk", "allocation"):
        assert key in schema["properties"]


# ------------------------------------------------------------ artifacts
def test_artifact_writer_emits_the_contract(tmp_path, dataset):
    from src.reports.artifacts import SCHEMA_VERSION as ART_VERSION, ArtifactWriter

    writer = ArtifactWriter(tmp_path / "runs" / "testrun", "testrun")
    cfg = get_preset("full_composite")

    writer.write_config(cfg)
    writer.write_metrics({"combined": {"cagr": 0.12, "sharpe": 0.9}})

    equity = pd.Series(
        [100.0, 101.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3)
    )
    writer.write_equity_curve(equity, benchmark=equity * 0.9)
    writer.write_json("attribution.json", {"capm": {"beta": 0.9}})
    writer.finalize(tmp_path)

    run_dir = tmp_path / "runs" / "testrun"
    for name in ("strategy_config.json", "backtest.json", "equity_curve.parquet",
                 "attribution.json", "manifest.json"):
        assert (run_dir / name).exists(), f"missing artifact {name}"

    metrics = json.loads((run_dir / "backtest.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == ART_VERSION
    assert metrics["run_id"] == "testrun"

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "testrun"


def test_equity_curve_columns_match_dashboard_expectations(tmp_path):
    """The Rust side reads these exact column names."""
    from src.reports.artifacts import ArtifactWriter

    writer = ArtifactWriter(tmp_path, "r1")
    idx = pd.bdate_range("2024-01-01", periods=10)
    equity = pd.Series(range(100, 110), index=idx, dtype=float)

    writer.write_equity_curve(
        equity,
        after_tax=equity * 0.98,
        benchmark=equity * 0.95,
        exposure=pd.Series(1.0, index=idx),
    )
    df = pd.read_parquet(tmp_path / "equity_curve.parquet")
    for col in ("date", "equity", "after_tax_equity", "benchmark", "exposure", "drawdown"):
        assert col in df.columns


def test_json_writer_handles_numpy_and_nan(tmp_path):
    import numpy as np

    from src.reports.artifacts import ArtifactWriter

    writer = ArtifactWriter(tmp_path, "r2")
    path = writer.write_json("test.json", {
        "int": np.int64(5),
        "float": np.float64(1.5),
        "nan": float("nan"),
        "ts": pd.Timestamp("2024-01-01"),
        "arr": np.array([1, 2]),
    })
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["int"] == 5
    assert payload["nan"] is None
    assert payload["arr"] == [1, 2]


def test_registry_counts_trials_for_deflated_sharpe(tmp_path):
    """Deflated Sharpe needs an honest trial count, so every trial must log."""
    from src.backtest.registry import RunRegistry

    registry = RunRegistry(tmp_path)
    for i in range(5):
        rec = registry.start(f"t{i}", "hash1", "data1", 42, study="sweep",
                             kind="optimization_trial")
        registry.complete(rec, {"sharpe": 0.5 + i * 0.1})

    assert registry.trial_count(study="sweep") == 5
    assert registry.trial_count(config_hash="hash1") == 5
    best = registry.best("sweep")
    assert best is not None
    assert best["sharpe"] == pytest.approx(0.9)
