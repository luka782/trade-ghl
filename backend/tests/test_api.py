from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings
from app.storage import Storage


def test_api_smoke_uses_cached_synthetic_data_without_network(
    tmp_path, monkeypatch, synthetic_provider
) -> None:
    data_dir = tmp_path / "data"
    db_path = tmp_path / "quant.sqlite3"
    monkeypatch.setenv("QUANT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("QUANT_DB_PATH", str(db_path))

    from app.main import create_app

    storage = Storage(Settings(data_dir=data_dir, db_path=db_path))
    client = TestClient(create_app(provider=synthetic_provider, storage=storage))
    dates = sorted(synthetic_provider.bars["date"].unique())
    symbols = sorted(synthetic_provider.bars["symbol"].unique())

    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["provider"] == "synthetic"
    assert health["database"] == "ok"
    assert health["version"] == "0.1.0"
    stock_response = client.get("/api/data/stocks?limit=3")
    assert stock_response.status_code == 200
    assert len(stock_response.json()["stocks"]) == 3
    assert "survivorship" in stock_response.json()["universe_warning"].lower()
    calendar = client.get(
        "/api/data/calendar",
        params={
            "start_date": pd.Timestamp(dates[0]).date().isoformat(),
            "end_date": pd.Timestamp(dates[4]).date().isoformat(),
        },
    )
    assert calendar.status_code == 200
    assert calendar.json()["count"] == 5
    factors_response = client.get(
        "/api/data/adjustment-factors/600001",
        params={
            "start_date": pd.Timestamp(dates[0]).date().isoformat(),
            "end_date": pd.Timestamp(dates[4]).date().isoformat(),
        },
    )
    assert factors_response.status_code == 200
    assert len(factors_response.json()["factors"]) == 5

    first_payload = {
        "symbols": [f"{symbol}.SH" for symbol in symbols],
        "start_date": pd.Timestamp(dates[5]).date().isoformat(),
        "end_date": pd.Timestamp(dates[-6]).date().isoformat(),
        "adjust": "qfq",
    }
    first_download = client.post("/api/data/download", json=first_payload)
    assert first_download.status_code == 200
    assert first_download.json()["status"] == "completed"
    assert all(
        item["status"] == "updated"
        for item in first_download.json()["results"]
    )
    assert all(
        item["execution_cache"]["status"] == "updated"
        for item in first_download.json()["results"]
    )

    full_payload = {
        **first_payload,
        "start_date": pd.Timestamp(dates[0]).date().isoformat(),
        "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
    }
    incremental = client.post("/api/data/download", json=full_payload)
    assert incremental.status_code == 200
    assert all(
        item["status"] == "updated" and item["fetched_rows"] > 0
        for item in incremental.json()["results"]
    )
    cached = client.post("/api/data/download", json=full_payload)
    assert all(
        item["status"] == "up_to_date" and item["fetched_rows"] == 0
        for item in cached.json()["results"]
    )
    no_data = client.post(
        "/api/data/download",
        json={**full_payload, "symbols": ["999999"]},
    )
    assert no_data.status_code == 200
    assert no_data.json()["status"] == "failed"
    assert no_data.json()["results"][0]["status"] == "no_data"

    status = client.get("/api/data/status")
    assert status.status_code == 200
    assert status.json()["dataset_count"] == len(symbols) * 2
    assert status.json()["row_count"] == len(synthetic_provider.bars) * 2
    assert all(item["name"] for item in status.json()["datasets"])
    stock_bars = client.get(
        f"/api/data/stocks/{symbols[0]}/bars",
        params={"adjust": "qfq", "limit": 30},
    )
    assert stock_bars.status_code == 200
    stock_bars_json = stock_bars.json()
    assert stock_bars_json["symbol"] == symbols[0]
    assert stock_bars_json["name"] == f"Synthetic {symbols[0]}"
    assert stock_bars_json["count"] == 30
    assert len(stock_bars_json["bars"]) == 30
    assert {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change_pct",
    }.issubset(stock_bars_json["bars"][-1])
    json.dumps(stock_bars_json, allow_nan=False)

    factors = client.get("/api/factors")
    assert factors.status_code == 200
    assert {item["name"] for item in factors.json()["factors"]} == {
        "momentum_20",
        "momentum_60",
        "momentum_252_21",
        "reversal_5",
        "volatility_20",
        "volume_change_20",
        "ma_bias_20",
        "price_position_60",
        "price_position_252",
        "max_return_20",
        "skewness_60",
        "atr_ratio_20",
        "overnight_reversal_20",
        "intraday_strength_20",
        "amount_surprise_20",
        "volume_price_corr_20",
        "beta_252",
        "idio_volatility_60",
        "relative_strength_60",
        "residual_momentum_60",
        "downside_volatility_20",
        "amihud_20",
        "pb",
        "bp",
        "ep",
        "dividend_yield",
        "roe",
        "gross_margin",
        "operating_cashflow_to_assets",
        "accruals",
        "asset_growth",
        "market_cap_size",
        "ma_200",
        "ma_slope_20",
        "distance_to_ma_200",
        "rsi_14",
        "bollinger_mid_20",
        "bollinger_upper_20",
        "bollinger_lower_20",
        "bollinger_percent_b_20",
        "bollinger_bandwidth_20",
    }
    for item in factors.json()["factors"]:
        assert {
            "display_name",
            "display_name_zh",
            "description_zh",
            "direction",
            "direction_label",
            "direction_kind",
            "applicable_assets",
            "lookback",
            "requirements",
        }.issubset(item)

    research_payload = {
        "factor_name": "momentum_20d",
        "symbols": [f"{symbol}.SH" for symbol in symbols],
        "start_date": pd.Timestamp(dates[25]).date().isoformat(),
        "end_date": pd.Timestamp(dates[-8]).date().isoformat(),
        "forward_period": 5,
        "quantiles": 5,
        "adjust": "qfq",
        "preprocess": "winsorize_zscore",
    }
    analysis = client.post("/api/factors/analyze", json=research_payload)
    assert analysis.status_code == 200, analysis.text
    analysis_json = analysis.json()
    assert analysis_json["coverage"]["ratio"] > 0
    assert analysis_json["ic"]["series"]
    assert analysis_json["ic_series"]
    assert analysis_json["summary"]["ic_mean"] is not None
    assert analysis_json["factor_distribution"]
    assert "survivorship" in analysis_json["universe_warning"].lower()
    json.dumps(analysis_json, allow_nan=False)

    pb_payload = {**research_payload, "factor_name": "pb"}
    unavailable_pb = client.post("/api/factors/analyze", json=pb_payload)
    assert unavailable_pb.status_code == 422
    assert "pb" in unavailable_pb.json()["detail"].lower()

    multifactor_config = {
        "name": "API多因子回归",
        "components": [
            {
                "factor_name": "momentum_20",
                "weight": 1.0,
                "enabled": True,
            },
            {
                "factor_name": "volatility_20",
                "weight": 1.0,
                "enabled": True,
            },
        ],
        "mode": "cross_sectional",
        "rolling_window": 20,
        "rolling_min_periods": 10,
    }
    saved_config = client.post(
        "/api/multifactor/configs",
        json=multifactor_config,
    )
    assert saved_config.status_code == 200, saved_config.text
    config_id = saved_config.json()["id"]
    assert client.get(f"/api/multifactor/configs/{config_id}").status_code == 200

    multifactor_analysis = client.post(
        "/api/multifactor/analyze",
        json={
            "config": multifactor_config,
            "symbols": symbols,
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-8]).date().isoformat(),
            "forward_period": 5,
            "quantiles": 5,
            "adjust": "qfq",
            "benchmark": "CSI300",
        },
    )
    assert multifactor_analysis.status_code == 200, multifactor_analysis.text
    multifactor_json = multifactor_analysis.json()
    assert multifactor_json["config_snapshot"]["name"] == "API多因子回归"
    assert "pearson" in multifactor_json["correlation_report"]
    assert "marginal_ic" in multifactor_json["correlation_report"]

    multifactor_backtest = client.post(
        "/api/multifactor/backtests",
        json={
            "config": multifactor_config,
            "symbols": symbols,
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "top_n": 2,
            "rebalance": "M",
            "adjust": "qfq",
            "benchmark": "CSI300",
        },
    )
    assert multifactor_backtest.status_code == 200, multifactor_backtest.text
    multifactor_backtest_json = multifactor_backtest.json()
    assert multifactor_backtest_json["summary"]["trade_count"] > 0
    multifactor_run_id = multifactor_backtest_json["id"]

    timing_config = {
        **multifactor_config,
        "name": "API单标的择时回归",
        "mode": "time_series",
    }
    timing_backtest = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert timing_backtest.status_code == 200, timing_backtest.text
    timing_json = timing_backtest.json()
    assert timing_json["config_snapshot"]["mode"] == "time_series"
    assert timing_json["asset_type"] == "stock"
    assert timing_json["score_trace"]
    timing_run_id = timing_json["id"]

    mean_reversion_backtest = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "options": {"timing_style": "mean_reversion"},
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert mean_reversion_backtest.status_code == 200, mean_reversion_backtest.text
    mean_reversion_json = mean_reversion_backtest.json()
    assert mean_reversion_json["summary"]["timing_style"] == "mean_reversion"
    assert any(
        row["timing_price_position_60"] is not None
        for row in mean_reversion_json["score_trace"]
    )
    mean_reversion_run_id = mean_reversion_json["id"]

    dual_entry_config = {
        **timing_config,
        "name": "API智能买入评分",
        "components": [
            {
                "factor_name": "reversal_5",
                "weight": 1.0,
                "enabled": True,
            }
        ],
    }
    dual_exit_config = {
        **timing_config,
        "name": "API智能卖出评分",
        "components": [
            {
                "factor_name": "ma_bias_20",
                "weight": 1.0,
                "enabled": True,
            }
        ],
    }
    identical_dual = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "entry_config": dual_entry_config,
            "exit_config": dual_entry_config,
            "options": {"timing_style": "factor_dual"},
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert identical_dual.status_code == 422
    assert "买入配置与卖出配置完全相同" in identical_dual.json()["detail"]

    factor_dual_backtest = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "entry_config": dual_entry_config,
            "exit_config": dual_exit_config,
            "options": {
                "timing_style": "factor_dual",
                "entry_score_threshold": 0.3,
                "exit_score_threshold": 0.3,
            },
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert factor_dual_backtest.status_code == 200, factor_dual_backtest.text
    factor_dual_json = factor_dual_backtest.json()
    assert factor_dual_json["summary"]["timing_style"] == "factor_dual"
    assert factor_dual_json["entry_config_snapshot"]["name"] == "API智能买入评分"
    assert factor_dual_json["exit_config_snapshot"]["name"] == "API智能卖出评分"
    assert any(
        row["entry_score"] is not None and row["exit_score"] is not None
        for row in factor_dual_json["score_trace"]
    )
    factor_dual_run_id = factor_dual_json["id"]

    regime_backtest = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "entry_config": dual_entry_config,
            "exit_config": dual_exit_config,
            "options": {
                "timing_style": "regime_reversion",
                "ma_period": 20,
                "ma_slope_period": 5,
                "rsi_period": 7,
                "bollinger_window": 10,
            },
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert regime_backtest.status_code == 200, regime_backtest.text
    regime_json = regime_backtest.json()
    assert regime_json["summary"]["timing_style"] == "regime_reversion"
    assert any(row["ma_200"] is not None for row in regime_json["score_trace"])
    assert any(row["rsi_14"] is not None for row in regime_json["score_trace"])
    regime_run_id = regime_json["id"]

    rsi_bollinger_backtest = client.post(
        "/api/timing/backtests",
        json={
            "symbol": symbols[0],
            "config": timing_config,
            "options": {
                "timing_style": "rsi_bollinger",
                "rsi_period": 7,
                "bollinger_window": 10,
            },
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "adjust": "qfq",
            "benchmark": "CSI300",
            "is_etf": False,
        },
    )
    assert rsi_bollinger_backtest.status_code == 200
    rsi_bollinger_json = rsi_bollinger_backtest.json()
    assert rsi_bollinger_json["summary"]["timing_style"] == "rsi_bollinger"
    rsi_bollinger_run_id = rsi_bollinger_json["id"]

    cta_run_ids: list[str] = []
    for style, expected_column in (
        ("donchian_atr", "donchian_upper"),
        ("ma_crossover_atr", "ma_fast"),
    ):
        cta_response = client.post(
            "/api/timing/backtests",
            json={
                "symbol": symbols[0],
                "config": timing_config,
                "options": {
                    "timing_style": style,
                    "donchian_entry_window": 20,
                    "donchian_exit_window": 10,
                    "ma_fast_period": 10,
                    "ma_slow_period": 20,
                    "ma_slope_period": 5,
                    "atr_period": 10,
                },
                "start_date": pd.Timestamp(dates[25]).date().isoformat(),
                "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
                "adjust": "qfq",
                "benchmark": "CSI300",
                "is_etf": False,
            },
        )
        assert cta_response.status_code == 200, cta_response.text
        cta_json = cta_response.json()
        assert cta_json["summary"]["timing_style"] == style
        assert any(
            row[expected_column] is not None
            for row in cta_json["score_trace"]
        )
        cta_run_ids.append(cta_json["id"])

    single_symbol_backtest = client.post(
        "/api/backtests",
        json={
            "factor_name": "momentum_20",
            "symbols": [symbols[0]],
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "top_n": 1,
            "rebalance": "D",
            "benchmark": "CSI300",
            "adjust": "qfq",
        },
    )
    assert single_symbol_backtest.status_code == 422
    assert "universe larger than top_n" in single_symbol_backtest.json()["detail"]

    equal_size_multifactor = client.post(
        "/api/multifactor/backtests",
        json={
            "config": {
                "name": "equal-size rejection",
                "components": [
                    {
                        "factor_name": "momentum_20",
                        "weight": 1,
                        "enabled": True,
                    }
                ],
            },
            "symbols": symbols,
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "top_n": len(symbols),
            "rebalance": "M",
            "benchmark": "CSI300",
            "adjust": "qfq",
        },
    )
    assert equal_size_multifactor.status_code == 422
    assert "universe size > top_n" in equal_size_multifactor.json()["detail"]

    backtest_payload = {
        "factor_name": "momentum_20d",
        "symbols": [f"{symbol}.SH" for symbol in symbols],
        "start_date": pd.Timestamp(dates[25]).date().isoformat(),
        "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "top_n": 2,
        "rebalance": "weekly",
        "commission_rate": 0.0003,
        "stamp_duty_rate": 0.0005,
        "slippage_rate": 0.0005,
        "benchmark": "000300.SH",
        "adjust": "qfq",
    }
    backtest = client.post("/api/backtests", json=backtest_payload)
    assert backtest.status_code == 200, backtest.text
    backtest_json = backtest.json()
    assert backtest_json["trades"]
    assert backtest_json["summary"]["benchmark_return"] is not None
    assert "T+1" in backtest_json["execution_policy"]
    assert backtest_json["factor_name"] == "momentum_20"
    assert backtest_json["equity_curve"][0]["benchmark"] is not None
    assert backtest_json["trades"][0]["market_open"] is not None
    assert any(
        "Historical ST status" in warning
        for warning in backtest_json["warnings"]
    )
    json.dumps(backtest_json, allow_nan=False)

    listing = client.get("/api/backtests")
    assert listing.status_code == 200
    assert listing.json()["count"] == 9
    assert len(listing.json()["backtests"]) == 9
    run_id = backtest_json["id"]
    detail = client.get(f"/api/backtests/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["result"]["id"] == run_id

    cors = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"
    deleted = client.delete(f"/api/backtests/{run_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": run_id, "deleted": True}
    assert client.get(f"/api/backtests/{run_id}").status_code == 404
    assert client.delete(f"/api/backtests/{run_id}").status_code == 404
    assert client.delete(f"/api/backtests/{multifactor_run_id}").status_code == 200
    assert client.delete(f"/api/backtests/{timing_run_id}").status_code == 200
    assert client.delete(f"/api/backtests/{mean_reversion_run_id}").status_code == 200
    assert client.delete(f"/api/backtests/{factor_dual_run_id}").status_code == 200
    assert client.delete(f"/api/backtests/{regime_run_id}").status_code == 200
    assert client.delete(f"/api/backtests/{rsi_bollinger_run_id}").status_code == 200
    for cta_run_id in cta_run_ids:
        assert client.delete(f"/api/backtests/{cta_run_id}").status_code == 200
    assert client.get("/api/backtests").json()["count"] == 0


def test_qfq_backtest_refuses_to_guess_execution_prices(
    tmp_path, synthetic_provider
) -> None:
    from app.main import create_app

    storage = Storage(
        Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "quant.sqlite3",
        )
    )
    dates = sorted(synthetic_provider.bars["date"].unique())
    symbols = sorted(synthetic_provider.bars["symbol"].unique())[:2]
    for symbol in symbols:
        storage.update_symbol(
            synthetic_provider,
            str(symbol),
            pd.Timestamp(dates[0]).date(),
            pd.Timestamp(dates[-1]).date(),
            "qfq",
        )
    client = TestClient(create_app(provider=synthetic_provider, storage=storage))
    response = client.post(
        "/api/backtests",
        json={
            "factor_name": "momentum_20",
            "symbols": [str(symbol) for symbol in symbols],
            "start_date": pd.Timestamp(dates[25]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
            "top_n": 1,
            "rebalance": "M",
            "benchmark": "CSI300",
            "adjust": "qfq",
        },
    )
    assert response.status_code == 400
    assert "Unadjusted execution data is required" in response.json()["detail"]
