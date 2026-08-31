from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data.base import normalize_symbol


Adjustment = Literal["qfq", "none"]
PreprocessMode = Literal["none", "winsorize", "zscore", "winsorize_zscore"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateRangeModel(StrictModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_date_range(self) -> "DateRangeModel":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class SymbolListMixin:
    @staticmethod
    def normalize_symbols(symbols: list[str] | None) -> list[str] | None:
        if symbols is None:
            return None
        normalized = list(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols))
        if not normalized:
            raise ValueError("symbols must not be empty")
        return normalized


class DownloadRequest(DateRangeModel, SymbolListMixin):
    symbols: list[str] = Field(min_length=1, max_length=1000)
    adjust: Adjustment = "qfq"

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        return cls.normalize_symbols(value) or []


class PreprocessOptions(StrictModel):
    winsorize: bool = True
    zscore: bool = True
    industry_neutralize: bool = False
    market_cap_neutralize: bool = False
    winsor_lower: float = Field(default=0.01, ge=0, le=1)
    winsor_upper: float = Field(default=0.99, ge=0, le=1)

    @model_validator(mode="after")
    def validate_winsor_bounds(self) -> "PreprocessOptions":
        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("winsor_lower must be less than winsor_upper")
        return self


class FactorAnalyzeRequest(DateRangeModel, SymbolListMixin):
    factor_name: str
    symbols: list[str] | None = Field(default=None, max_length=1000)
    forward_period: int = Field(default=5, ge=1, le=252)
    quantiles: int = Field(default=5, ge=2, le=20)
    adjust: Adjustment = "qfq"
    benchmark: Literal["CSI300", "CSI500"] = "CSI300"
    preprocess: PreprocessOptions | PreprocessMode = Field(
        default_factory=PreprocessOptions
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str] | None) -> list[str] | None:
        return cls.normalize_symbols(value)


class BacktestRequest(DateRangeModel, SymbolListMixin):
    factor_name: str
    symbols: list[str] | None = Field(default=None, max_length=1000)
    top_n: int = Field(default=10, ge=1, le=500)
    rebalance: Literal["D", "W", "M"] = "W"
    commission_rate: float = Field(default=0.0003, ge=0, lt=1)
    minimum_commission: float = Field(default=5.0, ge=0)
    minimum_trade_notional: float = Field(default=1_000.0, ge=0)
    rebalance_tolerance: float = Field(default=0.001, ge=0, lt=1)
    stamp_duty_rate: float = Field(default=0.0005, ge=0, lt=1)
    historical_stamp_duty: bool = True
    slippage_rate: float = Field(default=0.0005, ge=0, lt=1)
    max_stale_sessions: int = Field(default=20, ge=1, le=252)
    benchmark: Literal["CSI300", "CSI500"] = "CSI300"
    adjust: Adjustment = "qfq"

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str] | None) -> list[str] | None:
        return cls.normalize_symbols(value)

    @field_validator("rebalance", mode="before")
    @classmethod
    def normalize_rebalance(cls, value: str) -> str:
        aliases = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "d": "D",
            "w": "W",
            "m": "M",
        }
        return aliases.get(str(value).strip().lower(), value)

    @field_validator("benchmark", mode="before")
    @classmethod
    def normalize_benchmark(cls, value: str) -> str:
        aliases = {
            "000300": "CSI300",
            "000300.SH": "CSI300",
            "SH.000300": "CSI300",
            "000905": "CSI500",
            "000905.SH": "CSI500",
            "SH.000905": "CSI500",
        }
        text = str(value).strip().upper()
        return aliases.get(text, text)


class FactorComponentRequest(StrictModel):
    factor_name: str
    weight: float = 1.0
    enabled: bool = True
    direction: Literal[-1, 1] | None = None
    normalization: Literal[
        "auto", "cross_sectional", "rolling", "none"
    ] = "auto"
    winsorize: bool = True
    missing_policy: Literal["renormalize", "drop", "zero"] = "renormalize"


class MultiFactorConfigRequest(StrictModel):
    name: str = "自定义多因子"
    components: list[FactorComponentRequest] = Field(min_length=1, max_length=50)
    mode: Literal["cross_sectional", "time_series"] = "cross_sectional"
    rolling_window: int = Field(default=252, ge=20, le=1000)
    rolling_min_periods: int = Field(default=120, ge=10, le=1000)
    zscore_clip: float = Field(default=3.0, gt=0, le=20)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rolling_window(self) -> "MultiFactorConfigRequest":
        if self.rolling_min_periods > self.rolling_window:
            raise ValueError("rolling_min_periods cannot exceed rolling_window")
        if not any(item.enabled and item.weight != 0 for item in self.components):
            raise ValueError("at least one enabled nonzero component is required")
        return self


class MultiFactorAnalyzeRequest(DateRangeModel, SymbolListMixin):
    config: MultiFactorConfigRequest
    symbols: list[str] | None = Field(default=None, max_length=2000)
    forward_period: int = Field(default=5, ge=1, le=252)
    quantiles: int = Field(default=5, ge=2, le=20)
    adjust: Adjustment = "qfq"
    benchmark: Literal["CSI300", "CSI500"] = "CSI300"

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str] | None) -> list[str] | None:
        return cls.normalize_symbols(value)


class MultiFactorBacktestRequest(BacktestRequest):
    factor_name: str = "multifactor"
    config: MultiFactorConfigRequest


class TimingOptions(StrictModel):
    timing_style: Literal[
        "trend",
        "mean_reversion",
        "factor_dual",
        "regime_reversion",
        "rsi_bollinger",
        "rsi_bollinger",
    ] = "trend"
    buy_threshold: float = 0.7
    sell_threshold: float = 0.0
    entry_score_threshold: float = 0.4
    exit_score_threshold: float = 0.5
    setup_expiry_sessions: int = Field(default=30, ge=1, le=252)
    entry_max_price_position: float = Field(default=0.45, ge=0, le=1)
    exit_min_price_position: float = Field(default=0.65, ge=0, le=1)
    ma_period: int = Field(default=200, ge=20, le=500)
    ma_slope_period: int = Field(default=20, ge=1, le=252)
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_oversold: float = Field(default=30.0, ge=0, le=100)
    rsi_overbought: float = Field(default=70.0, ge=0, le=100)
    bollinger_window: int = Field(default=20, ge=5, le=252)
    bollinger_std: float = Field(default=2.0, gt=0, le=10)
    entry_factor_weight: float = Field(default=0.40, ge=0)
    entry_rsi_weight: float = Field(default=0.25, ge=0)
    entry_bollinger_weight: float = Field(default=0.25, ge=0)
    entry_regime_weight: float = Field(default=0.10, ge=0)
    exit_factor_weight: float = Field(default=0.40, ge=0)
    exit_rsi_weight: float = Field(default=0.20, ge=0)
    exit_bollinger_weight: float = Field(default=0.20, ge=0)
    exit_regime_weight: float = Field(default=0.20, ge=0)
    low_zone_threshold: float = Field(default=0.20, ge=0, le=1)
    low_recovery_threshold: float = Field(default=0.25, ge=0, le=1)
    high_reversal_threshold: float = Field(default=0.75, ge=0, le=1)
    high_zone_threshold: float = Field(default=0.80, ge=0, le=1)
    fixed_stop: float = Field(default=0.08, ge=0, lt=1)
    trailing_stop: float = Field(default=0.10, ge=0, lt=1)
    max_holding_sessions: int = Field(default=60, ge=1, le=2000)
    minimum_holding_sessions: int = Field(default=2, ge=1, le=252)
    cooldown_sessions: int = Field(default=5, ge=0, le=252)
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    commission_rate: float = Field(default=0.0003, ge=0, lt=1)
    minimum_commission: float = Field(default=5.0, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0, lt=1)
    minimum_trade_notional: float = Field(default=1_000.0, ge=0)
    lot_size: int = Field(default=100, ge=1, le=10000)
    max_stale_sessions: int = Field(default=20, ge=0, le=252)

    @model_validator(mode="after")
    def validate_mean_reversion_thresholds(self) -> "TimingOptions":
        if self.timing_style == "mean_reversion" and not (
            self.low_zone_threshold
            < self.low_recovery_threshold
            < self.high_reversal_threshold
            < self.high_zone_threshold
        ):
            raise ValueError(
                "mean-reversion thresholds must satisfy low_zone < "
                "low_recovery < high_reversal < high_zone"
            )
        if self.timing_style == "factor_dual" and not (
            self.low_zone_threshold
            < self.low_recovery_threshold
            < self.entry_max_price_position
            < self.exit_min_price_position
        ):
            raise ValueError(
                "factor-dual thresholds must satisfy low_zone < "
                "low_recovery < entry_max_position < exit_min_position"
            )
        if self.timing_style == "regime_reversion":
            if not (
                self.low_zone_threshold
                < self.low_recovery_threshold
                < self.entry_max_price_position
                < self.exit_min_price_position
            ):
                raise ValueError(
                    "regime-reversion thresholds must satisfy low_zone < "
                    "low_recovery < entry_max_position < exit_min_position"
                )
            if self.rsi_oversold >= self.rsi_overbought:
                raise ValueError("rsi_oversold must be less than rsi_overbought")
            if (
                self.entry_factor_weight
                + self.entry_rsi_weight
                + self.entry_bollinger_weight
                + self.entry_regime_weight
                <= 0
                or self.exit_factor_weight
                + self.exit_rsi_weight
                + self.exit_bollinger_weight
                + self.exit_regime_weight
                <= 0
            ):
                raise ValueError("regime-reversion score weights must be positive")
        return self


class TimingBacktestRequest(DateRangeModel):
    symbol: str
    config: MultiFactorConfigRequest
    entry_config: MultiFactorConfigRequest | None = None
    exit_config: MultiFactorConfigRequest | None = None
    options: TimingOptions = Field(default_factory=TimingOptions)
    adjust: Adjustment = "qfq"
    benchmark: Literal["CSI300", "CSI500"] = "CSI300"
    is_etf: bool = False

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


class TimingWalkForwardProtocolRequest(StrictModel):
    evaluation_years: Literal[3] = 3
    locked_oos_months: Literal[12] = 12
    train_months: int = Field(default=6, ge=3, le=18)
    validation_months: int = Field(default=2, ge=1, le=6)
    test_months: int = Field(default=2, ge=1, le=6)
    purge_sessions: int = Field(default=5, ge=0, le=60)
    embargo_sessions: int = Field(default=5, ge=0, le=60)


class TimingWalkForwardRequest(StrictModel, SymbolListMixin):
    symbols: list[str] = Field(min_length=2, max_length=20)
    config: MultiFactorConfigRequest
    entry_config: MultiFactorConfigRequest
    exit_config: MultiFactorConfigRequest
    options: TimingOptions
    adjust: Adjustment = "qfq"
    benchmark: Literal["CSI300", "CSI500"] = "CSI300"
    protocol: TimingWalkForwardProtocolRequest = Field(
        default_factory=TimingWalkForwardProtocolRequest
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        return cls.normalize_symbols(value) or []
