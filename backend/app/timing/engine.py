from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import floor
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..backtest.engine import execution_block_reason
from ..backtest.metrics import calculate_metrics
from ..json_utils import json_safe
from .position_sizing import (
    PositionSizingMode,
    calculate_position_size,
)
from .strategies import (
    RegimeEntryMode,
    donchian_entry,
    donchian_exit,
    moving_average_entry,
    moving_average_exit,
    regime_entry_decision,
)


# 择时引擎是“单证券、只做多”的有限状态机。信号在 T 日收盘后生成，
# 委托只会在下一可交易日开盘模拟成交；买入日收盘形成的卖出信号也只能在
# 再下一交易日开盘执行，满足 A 股买入后下一交易日才可卖出的限制。
Side = Literal["buy", "sell"]
TimingStyle = Literal[
    "trend",
    "mean_reversion",
    "factor_dual",
    "regime_reversion",
    "regime_reversion_legacy",
    "rsi_bollinger",
    "donchian_atr",
    "ma_crossover_atr",
    "buy_and_hold",
    "ma_200",
]

_PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "trade_open",
    "trade_high",
    "trade_low",
    "trade_close",
    "trade_volume",
    "trade_reference_close",
)
_REQUIRED_COLUMNS = ("date", *_PRICE_COLUMNS, "composite_score", "trend_score")

_REASON_CN = {
    "score_cross_up": "买入_综合评分上穿",
    "score_cross_down": "卖出_综合评分下穿",
    "trend_negative": "卖出_趋势转负",
    "fixed_stop": "卖出_固定止损",
    "trailing_stop": "卖出_移动止损",
    "max_holding": "卖出_最长持有期",
    "stale_data": "卖出_行情连续缺失",
    "low_zone_recovery": "买入_低位区反转确认",
    "high_zone_reversal": "卖出_高位区转弱确认",
    "entry_factor_confirmation": "买入_低位区综合因子确认",
    "exit_factor_risk": "卖出_高位区综合风险分触发",
    "regime_entry_confirmation": "买入_趋势反转综合确认",
    "regime_exit_risk": "卖出_综合趋势反转风险分触发",
    "rsi_overbought_reversal": "卖出_RSI超买后转弱",
    "bollinger_upper_reversal": "卖出_布林上轨后转弱",
    "ma200_breakdown": "卖出_长期均线趋势破坏",
    "rsi_bollinger_entry": "买入_RSI与布林带反转确认",
    "donchian_breakout": "买入_Donchian突破",
    "donchian_exit": "卖出_Donchian退出",
    "ma_crossover": "买入_双均线金叉且慢均线向上",
    "ma_crossdown": "卖出_双均线死叉",
    "buy_and_hold_entry": "买入_买入持有基准建仓",
    "ma200_entry": "买入_价格位于MA200上方",
    "ma200_exit": "卖出_价格跌破MA200",
    "atr_initial_stop": "卖出_ATR初始止损",
    "atr_trailing_stop": "卖出_ATR移动止损",
}


@dataclass(frozen=True, slots=True)
class TimingConfig:
    """单标的择时策略、风险控制和真实交易摩擦的不可变配置。

    不同 ``timing_style`` 共享仓位、费用和止损规则，但使用不同的入场/离场
    条件。将它们集中在一个快照中，能让历史结果在日后准确复现。
    """
    timing_style: TimingStyle = "trend"
    buy_threshold: float = 0.7
    sell_threshold: float = 0.0
    entry_score_threshold: float = 0.4
    exit_score_threshold: float = 0.5
    setup_expiry_sessions: int = 30
    entry_max_price_position: float = 0.45
    exit_min_price_position: float = 0.65
    ma_period: int = 200
    ma_slope_period: int = 20
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    entry_factor_weight: float = 0.40
    entry_rsi_weight: float = 0.25
    entry_bollinger_weight: float = 0.25
    entry_regime_weight: float = 0.10
    exit_factor_weight: float = 0.40
    exit_rsi_weight: float = 0.20
    exit_bollinger_weight: float = 0.20
    exit_regime_weight: float = 0.20
    regime_entry_mode: RegimeEntryMode = "confirmation_count"
    regime_confirmation_required: int = 2
    low_zone_threshold: float = 0.20
    low_recovery_threshold: float = 0.25
    high_reversal_threshold: float = 0.75
    high_zone_threshold: float = 0.80
    fixed_stop: float = 0.08
    trailing_stop: float = 0.10
    donchian_entry_window: int = 55
    donchian_exit_window: int = 20
    donchian_trend_filter: bool = False
    ma_fast_period: int = 20
    ma_slow_period: int = 60
    atr_period: int = 20
    atr_stop_multiple: float = 2.0
    atr_trailing_multiple: float = 3.0
    max_holding_sessions: int = 60
    # 0 表示允许买入日收盘形成卖出信号，并在下一交易日开盘执行（A股 T+1）。
    # 更大的值是策略主动延长最短持有期，而不是成交制度要求。
    minimum_holding_sessions: int = 0
    cooldown_sessions: int = 5
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    slippage: float = 0.0005
    minimum_trade_notional: float = 1_000.0
    lot_size: int = 100
    position_sizing: PositionSizingMode = "atr_risk"
    fixed_position_fraction: float = 0.50
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.50
    is_etf: bool = False
    max_stale_sessions: int = 20

    def __post_init__(self) -> None:
        # 在回测开始前校验参数关系，优先阻止“低位阈值高于高位阈值”一类
        # 无意义配置，而不是让状态机在运行时产生难以解释的信号。
        if self.timing_style not in {
            "trend",
            "mean_reversion",
            "factor_dual",
            "regime_reversion",
            "regime_reversion_legacy",
            "rsi_bollinger",
            "donchian_atr",
            "ma_crossover_atr",
            "buy_and_hold",
            "ma_200",
        }:
            raise ValueError(
                "timing_style must be trend, mean_reversion, factor_dual, "
                "regime_reversion, regime_reversion_legacy, rsi_bollinger, "
                "donchian_atr, ma_crossover_atr, buy_and_hold, or ma_200"
            )
        if self.regime_entry_mode not in {
            "legacy_all",
            "confirmation_count",
        }:
            raise ValueError("regime_entry_mode is invalid")
        if self.regime_confirmation_required not in {1, 2, 3}:
            raise ValueError(
                "regime_confirmation_required must be 1, 2, or 3"
            )
        if self.position_sizing not in {"full", "fixed", "atr_risk"}:
            raise ValueError("position_sizing is invalid")
        if not np.isfinite(self.buy_threshold) or not np.isfinite(
            self.sell_threshold
        ):
            raise ValueError("Signal thresholds must be finite")
        if not np.isfinite(self.entry_score_threshold) or not np.isfinite(
            self.exit_score_threshold
        ):
            raise ValueError("Entry and exit score thresholds must be finite")
        for name in ("fixed_stop", "trailing_stop", "commission_rate", "slippage"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in (
            "max_holding_sessions",
            "minimum_holding_sessions",
            "cooldown_sessions",
            "setup_expiry_sessions",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_holding_sessions < self.minimum_holding_sessions:
            raise ValueError(
                "max_holding_sessions cannot be less than minimum_holding_sessions"
            )
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.minimum_commission < 0 or self.minimum_trade_notional < 0:
            raise ValueError("Minimum commission/notional must be non-negative")
        if self.lot_size < 1:
            raise ValueError("lot_size must be at least 1")
        if self.max_stale_sessions < 0:
            raise ValueError("max_stale_sessions must be non-negative")
        for name in (
            "ma_period",
            "ma_slope_period",
            "rsi_period",
            "bollinger_window",
            "donchian_entry_window",
            "donchian_exit_window",
            "ma_fast_period",
            "ma_slow_period",
            "atr_period",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.ma_fast_period >= self.ma_slow_period:
            raise ValueError("ma_fast_period must be less than ma_slow_period")
        for name in ("atr_stop_multiple", "atr_trailing_multiple"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "fixed_position_fraction",
            "risk_per_trade",
            "max_position_fraction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.bollinger_std) or self.bollinger_std <= 0:
            raise ValueError("bollinger_std must be positive")
        thresholds = (
            self.low_zone_threshold,
            self.low_recovery_threshold,
            self.high_reversal_threshold,
            self.high_zone_threshold,
            self.entry_max_price_position,
            self.exit_min_price_position,
        )
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in thresholds):
            raise ValueError("Mean-reversion thresholds must be within [0, 1]")
        if self.timing_style == "mean_reversion" and not (
            self.low_zone_threshold
            < self.low_recovery_threshold
            < self.high_reversal_threshold
            < self.high_zone_threshold
        ):
            raise ValueError(
                "Mean-reversion thresholds must satisfy low_zone < "
                "low_recovery < high_reversal < high_zone"
            )
        if self.timing_style == "factor_dual" and not (
            self.low_zone_threshold
            < self.low_recovery_threshold
            < self.entry_max_price_position
            < self.exit_min_price_position
        ):
            raise ValueError(
                "Factor-dual thresholds must satisfy low_zone < "
                "low_recovery < entry_max_position < exit_min_position"
            )
        if self.timing_style in {
            "regime_reversion",
            "regime_reversion_legacy",
        }:
            if not (
                self.low_zone_threshold
                < self.low_recovery_threshold
                < self.entry_max_price_position
                < self.exit_min_price_position
            ):
                raise ValueError(
                    "Regime-reversion thresholds must satisfy low_zone < "
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
                raise ValueError("regime-reversion weights must sum positive")


def _close_time(value: pd.Timestamp) -> str:
    return f"{pd.Timestamp(value).date().isoformat()}T15:00:00+08:00"


def _open_time(value: pd.Timestamp) -> str:
    return f"{pd.Timestamp(value).date().isoformat()}T09:30:00+08:00"


def _number(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _valid_price(value: Any) -> bool:
    number = _number(value)
    return bool(np.isfinite(number) and number > 0)


def _commission(notional: float, config: TimingConfig) -> float:
    if notional <= 0 or config.commission_rate <= 0:
        return 0.0
    return max(notional * config.commission_rate, config.minimum_commission)


def _stamp_duty_rate(trading_date: pd.Timestamp, config: TimingConfig) -> float:
    if config.is_etf:
        return 0.0
    return 0.001 if trading_date.date() < date(2023, 8, 28) else 0.0005


def _affordable_shares(
    cash: float, execution_price: float, config: TimingConfig
) -> int:
    """计算现金可承受的最大整手数量，并把最低佣金计入可用资金。

    先用费率给出快速上界，再向下逐手修正，避免最低佣金导致实际扣款超过
    账户现金。A 股/ETF 默认一手为 100 股（份）。
    """
    if cash <= 0 or execution_price <= 0:
        return 0
    divisor = execution_price * (1.0 + config.commission_rate)
    shares = int(floor(cash / divisor / config.lot_size) * config.lot_size)
    while shares > 0:
        notional = shares * execution_price
        if notional + _commission(notional, config) <= cash + 1e-9:
            return shares
        shares -= config.lot_size
    return 0


def _contributions(row: pd.Series, columns: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for column in columns:
        value = _number(row[column])
        result[column] = value if np.isfinite(value) else None
    return result


def _prepare_frame(
    signal_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """校验并规范化输入信号，收集用于交易审计的因子明细列。

    这里强制“一日一行、一资产”是因为仓位状态（现金、持仓、止损价）属于单一
    证券；多资产选股应走 ``backtest.engine`` 的组合回测引擎。
    """
    if signal_frame.empty:
        raise ValueError("The timing signal frame is empty")
    missing = [column for column in _REQUIRED_COLUMNS if column not in signal_frame]
    if missing:
        raise ValueError(f"Missing required timing columns: {', '.join(missing)}")

    frame = signal_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Timing frame contains an invalid date")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("Timing frame must contain one row per trading date")
    if "symbol" in frame and frame["symbol"].dropna().astype(str).nunique() > 1:
        raise ValueError("Timing engine accepts exactly one asset")

    for column in (*_PRICE_COLUMNS, "composite_score", "trend_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "timing_price_position_60" in frame:
        frame["timing_price_position_60"] = pd.to_numeric(
            frame["timing_price_position_60"], errors="coerce"
        )
    for column in (
        "entry_score",
        "exit_score",
        "entry_score_final",
        "exit_score_final",
        "ma_200",
        "ma_slope_20",
        "distance_to_ma_200",
        "rsi_14",
        "bollinger_mid_20",
        "bollinger_upper_20",
        "bollinger_lower_20",
        "bollinger_percent_b_20",
        "bollinger_bandwidth_20",
        "atr_20",
        "donchian_upper",
        "donchian_lower",
        "ma_fast",
        "ma_slow",
        "ma_slow_slope",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    contribution_columns = sorted(
        column for column in frame if column.startswith("contribution_")
    )
    detail_columns = sorted(
        column
        for column in frame
        if column.startswith(
            ("factor_", "normalized_", "contribution_", "weight_", "direction_")
        )
    )
    for column in detail_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, contribution_columns, detail_columns


def _blocked_order(
    order: dict[str, Any],
    execution_date: pd.Timestamp | None,
    reason: str,
    requested_shares: int | None = None,
) -> dict[str, Any]:
    """将未成交委托转成可展示、可追溯的审计记录，而非静默丢弃。"""
    return {
        "side": order["side"],
        "reason": reason,
        "reason_code": order["reason_code"],
        "signal_date": order["signal_date"],
        "signal_time": _close_time(order["signal_date"]),
        "execution_date": execution_date,
        "execution_time": (
            _open_time(execution_date) if execution_date is not None else None
        ),
        "requested_shares": requested_shares,
        "composite_score": order.get("composite_score"),
        "trend_score": order.get("trend_score"),
        "timing_style": order.get("timing_style"),
        "timing_price_position_60": order.get("timing_price_position_60"),
        "entry_score": order.get("entry_score"),
        "exit_score": order.get("exit_score"),
        "entry_score_final": order.get("entry_score_final"),
        "exit_score_final": order.get("exit_score_final"),
        "market_regime": order.get("market_regime"),
        "ma_200": order.get("ma_200"),
        "ma_slope_20": order.get("ma_slope_20"),
        "distance_to_ma_200": order.get("distance_to_ma_200"),
        "rsi_14": order.get("rsi_14"),
        "bollinger_percent_b_20": order.get(
            "bollinger_percent_b_20"
        ),
        "atr_20": order.get("atr_20"),
        "donchian_upper": order.get("donchian_upper"),
        "donchian_lower": order.get("donchian_lower"),
        "ma_fast": order.get("ma_fast"),
        "ma_slow": order.get("ma_slow"),
        "ma_slow_slope": order.get("ma_slow_slope"),
        "factor_contributions": order.get("factor_contributions", {}),
        "factor_details": order.get("factor_details", {}),
    }


def run_timing(
    signal_frame: pd.DataFrame,
    config: TimingConfig | None = None,
) -> dict[str, Any]:
    """Run deterministic single-asset long-only timing on precomputed signals.

    Signals are observed after the adjusted close on T and queued exactly once
    for execution at the raw open on the next supplied session. Adjusted
    accounting units map actual raw execution notional onto the adjusted open,
    so holding returns begin at that open.
    This is necessarily approximate across corporate actions because no
    historical share-conversion ledger is supplied.
    """

    config = config or TimingConfig()
    frame, contribution_columns, detail_columns = _prepare_frame(signal_frame)
    # 不让策略在指标缺失时悄悄退化为“永不交易”。每种模式先确认依赖数据
    # 存在且有有效值，错误会被 API 明确返回给使用者。
    if (
        config.timing_style in {
            "mean_reversion",
            "factor_dual",
            "regime_reversion",
            "regime_reversion_legacy",
        }
        and (
            "timing_price_position_60" not in frame
            or not frame["timing_price_position_60"].notna().any()
        )
    ):
        raise ValueError(
            "Mean-reversion timing requires causal timing_price_position_60 values."
        )
    if config.timing_style in {
        "factor_dual",
        "regime_reversion",
        "regime_reversion_legacy",
    }:
        missing_scores = [
            column
            for column in ("entry_score", "exit_score")
            if column not in frame
            or not pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        if missing_scores:
            raise ValueError(
                "Factor-dual timing requires usable "
                + ", ".join(missing_scores)
                + " values."
            )
    if config.timing_style in {
        "regime_reversion",
        "regime_reversion_legacy",
    }:
        required_regime = (
            "entry_score_final",
            "exit_score_final",
            "ma_200",
            "ma_slope_20",
            "rsi_14",
            "bollinger_percent_b_20",
            "market_regime",
        )
        missing_regime = [
            column
            for column in required_regime
            if column not in frame
            or (
                column != "market_regime"
                and not pd.to_numeric(
                    frame[column], errors="coerce"
                ).notna().any()
            )
        ]
        if missing_regime:
            raise ValueError(
                "Regime-reversion timing requires usable "
                + ", ".join(missing_regime)
                + " values."
            )
    if config.timing_style == "rsi_bollinger":
        missing_indicators = [
            column
            for column in (
                "rsi_14",
                "bollinger_percent_b_20",
                "bollinger_mid_20",
                "bollinger_upper_20",
                "bollinger_lower_20",
            )
            if column not in frame
            or not pd.to_numeric(
                frame[column], errors="coerce"
            ).notna().any()
        ]
        if missing_indicators:
            raise ValueError(
                "RSI-Bollinger timing requires usable "
                + ", ".join(missing_indicators)
                + " values."
            )
    if config.position_sizing == "atr_risk" or config.timing_style in {
        "donchian_atr",
        "ma_crossover_atr",
    }:
        if (
            "atr_20" not in frame
            or not pd.to_numeric(
                frame["atr_20"], errors="coerce"
            ).notna().any()
        ):
            raise ValueError(
                "ATR position sizing and CTA strategies require usable atr_20 values."
            )
    strategy_columns = {
        "donchian_atr": ("donchian_upper", "donchian_lower"),
        "ma_crossover_atr": (
            "ma_fast",
            "ma_slow",
            "ma_slow_slope",
        ),
        "ma_200": ("ma_200",),
    }
    missing_strategy = [
        column
        for column in strategy_columns.get(config.timing_style, ())
        if column not in frame
        or not pd.to_numeric(
            frame[column], errors="coerce"
        ).notna().any()
    ]
    if missing_strategy:
        raise ValueError(
            f"{config.timing_style} requires usable "
            + ", ".join(missing_strategy)
            + " values."
        )
    dates = list(pd.DatetimeIndex(frame["date"]))
    symbol = (
        str(frame["symbol"].dropna().iloc[0])
        if "symbol" in frame and not frame["symbol"].dropna().empty
        else None
    )

    # raw_shares 用不复权价格结算真实现金；accounting_quantity 用前复权价格
    # 计算连续收益。两套数量分开保存，避免复权尺度直接污染成交金额。
    cash = float(config.initial_capital)
    raw_shares = 0
    accounting_quantity = 0.0
    entry_index: int | None = None
    entry_adjusted_price = float("nan")
    entry_atr = float("nan")
    entry_raw_notional = 0.0
    entry_total_cost = 0.0
    peak_adjusted_close = float("nan")
    last_exit_index: int | None = None
    stale_sessions = 0
    # pending 只保存 T 日收盘后形成的一笔待执行订单，并于下一交易日尝试成交，
    # 从结构上保证同日不会“看到收盘信号后又按该收盘价成交”。
    pending: dict[str, Any] | None = None
    low_zone_armed = False
    high_zone_armed = False
    low_zone_armed_index: int | None = None
    regime_rsi_recovered = False
    regime_bollinger_recovered = False
    rsi_overbought_armed = False
    bollinger_upper_armed = False

    total_commission = 0.0
    total_stamp_duty = 0.0
    total_slippage_cost = 0.0
    turnover = 0.0
    trades: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    blocked_orders: list[dict[str, Any]] = []
    score_trace: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_adjusted_close = float("nan")
    entry_funnel = {
        "indicator_ready": 0,
        "candidate_zone": 0,
        "regime_allowed": 0,
        "price_allowed": 0,
        "rsi_confirmation": 0,
        "bollinger_confirmation": 0,
        "factor_confirmation": 0,
        "confirmation_passed": 0,
        "orders_created": 0,
        "orders_filled": 0,
    }

    for index, row in frame.iterrows():
        trading_date = pd.Timestamp(row["date"])
        adjusted_open = _number(row["open"])
        adjusted_close = _number(row["close"])
        raw_open = _number(row["trade_open"])
        raw_volume = _number(row["trade_volume"])
        pre_trade_equity = cash + (
            accounting_quantity * last_adjusted_close
            if raw_shares and np.isfinite(last_adjusted_close)
            else 0.0
        )
        daily_notional = 0.0

        if raw_shares:
            if np.isfinite(adjusted_close) and adjusted_close > 0:
                stale_sessions = 0
                last_adjusted_close = adjusted_close
            else:
                stale_sessions += 1

        if pending is not None:
            order = pending
            pending = None
            if not _valid_price(raw_open):
                blocked_orders.append(
                    _blocked_order(order, trading_date, "missing_raw_open")
                )
            elif not np.isfinite(raw_volume) or raw_volume <= 0:
                blocked_orders.append(
                    _blocked_order(order, trading_date, "suspension_or_missing_volume")
                )
            elif symbol is not None and (
                block_reason := execution_block_reason(
                    symbol,
                    row,
                    order["side"],
                    trading_date,
                )
            ) is not None:
                blocked_orders.append(
                    _blocked_order(order, trading_date, block_reason)
                )
            elif order["side"] == "buy":
                execution_price = raw_open * (1.0 + config.slippage)
                signal_atr = _number(order.get("atr_20"))
                size = calculate_position_size(
                    cash=cash,
                    equity=pre_trade_equity,
                    execution_price=execution_price,
                    adjusted_price=adjusted_open,
                    atr=signal_atr,
                    lot_size=config.lot_size,
                    mode=config.position_sizing,
                    fixed_fraction=config.fixed_position_fraction,
                    risk_per_trade=config.risk_per_trade,
                    max_fraction=config.max_position_fraction,
                    atr_stop_multiple=config.atr_stop_multiple,
                )
                affordable = _affordable_shares(
                    min(cash, size.target_notional),
                    execution_price,
                    config,
                )
                shares = min(size.shares, affordable)
                notional = shares * execution_price
                if size.reason is not None:
                    blocked_orders.append(
                        _blocked_order(
                            order, trading_date, size.reason, shares
                        )
                    )
                elif shares <= 0:
                    blocked_orders.append(
                        _blocked_order(
                            order, trading_date, "insufficient_cash", shares
                        )
                    )
                elif notional < config.minimum_trade_notional:
                    blocked_orders.append(
                        _blocked_order(
                            order, trading_date, "below_minimum_trade_notional", shares
                        )
                    )
                elif not _valid_price(adjusted_open):
                    blocked_orders.append(
                        _blocked_order(
                            order, trading_date, "missing_adjusted_open", shares
                        )
                    )
                else:
                    commission = _commission(notional, config)
                    cash -= notional + commission
                    raw_shares = shares
                    accounting_quantity = notional / adjusted_open
                    entry_index = index
                    entry_adjusted_price = adjusted_open
                    entry_atr = signal_atr
                    entry_raw_notional = notional
                    entry_total_cost = notional + commission
                    peak_adjusted_close = adjusted_open
                    last_adjusted_close = adjusted_open
                    stale_sessions = 0
                    low_zone_armed = False
                    high_zone_armed = False
                    low_zone_armed_index = None
                    regime_rsi_recovered = False
                    regime_bollinger_recovered = False
                    rsi_overbought_armed = False
                    bollinger_upper_armed = False
                    slippage_cost = shares * raw_open * config.slippage
                    total_commission += commission
                    total_slippage_cost += slippage_cost
                    daily_notional += notional
                    entry_funnel["orders_filled"] += 1
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "buy",
                            "reason": order["reason"],
                            "reason_code": order["reason_code"],
                            "reason_codes": order["reason_codes"],
                            "composite_score": order.get("composite_score"),
                            "trend_score": order.get("trend_score"),
                            "timing_style": order.get("timing_style"),
                            "timing_price_position_60": order.get(
                                "timing_price_position_60"
                            ),
                            "entry_score": order.get("entry_score"),
                            "exit_score": order.get("exit_score"),
                            "entry_score_final": order.get(
                                "entry_score_final"
                            ),
                            "exit_score_final": order.get(
                                "exit_score_final"
                            ),
                            "market_regime": order.get("market_regime"),
                            "ma_200": order.get("ma_200"),
                            "ma_slope_20": order.get("ma_slope_20"),
                            "distance_to_ma_200": order.get(
                                "distance_to_ma_200"
                            ),
                            "rsi_14": order.get("rsi_14"),
                            "bollinger_percent_b_20": order.get(
                                "bollinger_percent_b_20"
                            ),
                            "atr_20": order.get("atr_20"),
                            "donchian_upper": order.get(
                                "donchian_upper"
                            ),
                            "donchian_lower": order.get(
                                "donchian_lower"
                            ),
                            "ma_fast": order.get("ma_fast"),
                            "ma_slow": order.get("ma_slow"),
                            "ma_slow_slope": order.get(
                                "ma_slow_slope"
                            ),
                            "factor_contributions": order.get(
                                "factor_contributions", {}
                            ),
                            "factor_details": order.get("factor_details", {}),
                            "signal_date": order["signal_date"],
                            "signal_time": _close_time(order["signal_date"]),
                            "execution_date": trading_date,
                            "execution_time": _open_time(trading_date),
                            "execution_session": "T+1 open",
                            "raw_open": raw_open,
                            "raw_price": execution_price,
                            "atr_20": signal_atr,
                            "position_sizing": config.position_sizing,
                            "target_notional": size.target_notional,
                            "risk_cash": size.risk_cash,
                            "stop_distance": size.stop_distance,
                            "position_fraction": (
                                notional / pre_trade_equity
                                if pre_trade_equity > 0
                                else None
                            ),
                            "shares": shares,
                            "lots": shares // config.lot_size,
                            "notional": notional,
                            "commission": commission,
                            "stamp_duty_rate": 0.0,
                            "stamp_duty": 0.0,
                            "fees": commission,
                            "slippage_cost": slippage_cost,
                            "accounting_quantity": accounting_quantity,
                            "holding_sessions": 0,
                            "return": None,
                        }
                    )
            else:
                if raw_shares <= 0:
                    blocked_orders.append(
                        _blocked_order(order, trading_date, "no_position", 0)
                    )
                elif entry_index == index:
                    blocked_orders.append(
                        _blocked_order(order, trading_date, "t_plus_one", raw_shares)
                    )
                else:
                    shares = raw_shares
                    execution_price = raw_open * (1.0 - config.slippage)
                    notional = shares * execution_price
                    commission = _commission(notional, config)
                    stamp_rate = _stamp_duty_rate(trading_date, config)
                    stamp_duty = notional * stamp_rate
                    fees = commission + stamp_duty
                    cash += notional - fees
                    slippage_cost = shares * raw_open * config.slippage
                    holding_sessions = (
                        index - entry_index if entry_index is not None else None
                    )
                    trade_return = (
                        (notional - fees) / entry_total_cost - 1.0
                        if entry_total_cost > 0
                        else float("nan")
                    )
                    realized_pnl = (
                        notional - fees - entry_total_cost
                        if entry_total_cost > 0
                        else float("nan")
                    )
                    total_commission += commission
                    total_stamp_duty += stamp_duty
                    total_slippage_cost += slippage_cost
                    daily_notional += notional
                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "sell",
                            "reason": order["reason"],
                            "reason_code": order["reason_code"],
                            "reason_codes": order["reason_codes"],
                            "composite_score": order.get("composite_score"),
                            "trend_score": order.get("trend_score"),
                            "timing_style": order.get("timing_style"),
                            "timing_price_position_60": order.get(
                                "timing_price_position_60"
                            ),
                            "entry_score": order.get("entry_score"),
                            "exit_score": order.get("exit_score"),
                            "entry_score_final": order.get(
                                "entry_score_final"
                            ),
                            "exit_score_final": order.get(
                                "exit_score_final"
                            ),
                            "market_regime": order.get("market_regime"),
                            "ma_200": order.get("ma_200"),
                            "ma_slope_20": order.get("ma_slope_20"),
                            "distance_to_ma_200": order.get(
                                "distance_to_ma_200"
                            ),
                            "rsi_14": order.get("rsi_14"),
                            "bollinger_percent_b_20": order.get(
                                "bollinger_percent_b_20"
                            ),
                            "atr_20": order.get("atr_20"),
                            "donchian_upper": order.get(
                                "donchian_upper"
                            ),
                            "donchian_lower": order.get(
                                "donchian_lower"
                            ),
                            "ma_fast": order.get("ma_fast"),
                            "ma_slow": order.get("ma_slow"),
                            "ma_slow_slope": order.get(
                                "ma_slow_slope"
                            ),
                            "factor_contributions": order.get(
                                "factor_contributions", {}
                            ),
                            "factor_details": order.get("factor_details", {}),
                            "signal_date": order["signal_date"],
                            "signal_time": _close_time(order["signal_date"]),
                            "execution_date": trading_date,
                            "execution_time": _open_time(trading_date),
                            "execution_session": "T+1 open",
                            "raw_open": raw_open,
                            "raw_price": execution_price,
                            "shares": shares,
                            "lots": shares // config.lot_size,
                            "notional": notional,
                            "commission": commission,
                            "stamp_duty_rate": stamp_rate,
                            "stamp_duty": stamp_duty,
                            "fees": fees,
                            "slippage_cost": slippage_cost,
                            "accounting_quantity": accounting_quantity,
                            "holding_sessions": holding_sessions,
                            "return": trade_return,
                            "pnl": realized_pnl,
                            "realized_pnl": realized_pnl,
                        }
                    )
                    raw_shares = 0
                    accounting_quantity = 0.0
                    entry_index = None
                    entry_adjusted_price = float("nan")
                    entry_atr = float("nan")
                    entry_raw_notional = 0.0
                    entry_total_cost = 0.0
                    peak_adjusted_close = float("nan")
                    stale_sessions = 0
                    last_exit_index = index
                    high_zone_armed = False
                    low_zone_armed = False
                    low_zone_armed_index = None
                    regime_rsi_recovered = False
                    regime_bollinger_recovered = False
                    rsi_overbought_armed = False
                    bollinger_upper_armed = False

        score = _number(row["composite_score"])
        trend = _number(row["trend_score"])
        price_position = (
            _number(row["timing_price_position_60"])
            if "timing_price_position_60" in row
            else float("nan")
        )
        previous_price_position = (
            _number(frame.iloc[index - 1]["timing_price_position_60"])
            if index > 0 and "timing_price_position_60" in frame
            else float("nan")
        )
        entry_score = (
            _number(row["entry_score"])
            if "entry_score" in row
            else float("nan")
        )
        exit_score = (
            _number(row["exit_score"])
            if "exit_score" in row
            else float("nan")
        )
        previous_entry_score = (
            _number(frame.iloc[index - 1]["entry_score"])
            if index > 0 and "entry_score" in frame
            else float("nan")
        )
        entry_score_final = (
            _number(row["entry_score_final"])
            if "entry_score_final" in row
            else float("nan")
        )
        exit_score_final = (
            _number(row["exit_score_final"])
            if "exit_score_final" in row
            else float("nan")
        )
        previous_entry_score_final = (
            _number(frame.iloc[index - 1]["entry_score_final"])
            if index > 0 and "entry_score_final" in frame
            else float("nan")
        )
        rsi_value = (
            _number(row["rsi_14"]) if "rsi_14" in row else float("nan")
        )
        previous_rsi = (
            _number(frame.iloc[index - 1]["rsi_14"])
            if index > 0 and "rsi_14" in frame
            else float("nan")
        )
        percent_b = (
            _number(row["bollinger_percent_b_20"])
            if "bollinger_percent_b_20" in row
            else float("nan")
        )
        atr_value = (
            _number(row["atr_20"]) if "atr_20" in row else float("nan")
        )
        donchian_upper = (
            _number(row["donchian_upper"])
            if "donchian_upper" in row
            else float("nan")
        )
        donchian_lower = (
            _number(row["donchian_lower"])
            if "donchian_lower" in row
            else float("nan")
        )
        ma_fast = (
            _number(row["ma_fast"])
            if "ma_fast" in row
            else float("nan")
        )
        ma_slow = (
            _number(row["ma_slow"])
            if "ma_slow" in row
            else float("nan")
        )
        ma_slow_slope = (
            _number(row["ma_slow_slope"])
            if "ma_slow_slope" in row
            else float("nan")
        )
        previous_ma_fast = (
            _number(frame.iloc[index - 1]["ma_fast"])
            if index > 0 and "ma_fast" in frame
            else float("nan")
        )
        previous_ma_slow = (
            _number(frame.iloc[index - 1]["ma_slow"])
            if index > 0 and "ma_slow" in frame
            else float("nan")
        )
        previous_percent_b = (
            _number(frame.iloc[index - 1]["bollinger_percent_b_20"])
            if index > 0 and "bollinger_percent_b_20" in frame
            else float("nan")
        )
        market_regime = str(row.get("market_regime", "sideways"))
        if (
            config.timing_style in {"mean_reversion", "factor_dual"}
            and np.isfinite(price_position)
        ):
            if raw_shares == 0 and price_position <= config.low_zone_threshold:
                low_zone_armed = True
                low_zone_armed_index = index
            elif (
                config.timing_style == "factor_dual"
                and raw_shares == 0
                and low_zone_armed
                and (
                    low_zone_armed_index is None
                    or index - low_zone_armed_index
                    > config.setup_expiry_sessions
                    or price_position > config.entry_max_price_position
                )
            ):
                low_zone_armed = False
                low_zone_armed_index = None
            if raw_shares:
                high_trigger = (
                    config.exit_min_price_position
                    if config.timing_style == "factor_dual"
                    else config.high_zone_threshold
                )
                if price_position >= high_trigger:
                    high_zone_armed = True
        elif (
            config.timing_style
            in {"regime_reversion", "regime_reversion_legacy"}
            and np.isfinite(price_position)
            and np.isfinite(rsi_value)
            and np.isfinite(percent_b)
        ):
            legacy_regime = (
                config.timing_style == "regime_reversion_legacy"
                or config.regime_entry_mode == "legacy_all"
            )
            regime_setup = raw_shares == 0 and (
                (
                    market_regime != "downtrend"
                    and price_position <= config.low_zone_threshold
                    and rsi_value <= config.rsi_oversold
                    and percent_b <= 0.0
                )
                if legacy_regime
                else (
                    price_position <= config.low_zone_threshold
                    or percent_b <= 0.0
                )
            )
            if regime_setup:
                low_zone_armed = True
                low_zone_armed_index = index
                regime_rsi_recovered = False
                regime_bollinger_recovered = False
            elif (
                raw_shares == 0
                and low_zone_armed
                and (
                    low_zone_armed_index is None
                    or index - low_zone_armed_index
                    > config.setup_expiry_sessions
                    or price_position > config.entry_max_price_position
                    or (legacy_regime and market_regime == "downtrend")
                )
            ):
                low_zone_armed = False
                low_zone_armed_index = None
                regime_rsi_recovered = False
                regime_bollinger_recovered = False
            if low_zone_armed:
                if (
                    np.isfinite(previous_rsi)
                    and previous_rsi <= config.rsi_oversold
                    and rsi_value > config.rsi_oversold
                    and rsi_value > previous_rsi
                ):
                    regime_rsi_recovered = True
                if (
                    np.isfinite(previous_percent_b)
                    and previous_percent_b <= 0.0
                    and percent_b > 0.0
                    and percent_b > previous_percent_b
                ):
                    regime_bollinger_recovered = True
            if raw_shares:
                if rsi_value >= config.rsi_overbought:
                    rsi_overbought_armed = True
                if percent_b >= 1.0:
                    bollinger_upper_armed = True
        elif (
            config.timing_style == "rsi_bollinger"
            and np.isfinite(rsi_value)
            and np.isfinite(percent_b)
        ):
            setup = (
                raw_shares == 0
                and rsi_value <= config.rsi_oversold
                and percent_b <= 0.0
            )
            if setup:
                low_zone_armed = True
                low_zone_armed_index = index
                regime_rsi_recovered = False
                regime_bollinger_recovered = False
            elif (
                raw_shares == 0
                and low_zone_armed
                and (
                    low_zone_armed_index is None
                    or index - low_zone_armed_index
                    > config.setup_expiry_sessions
                )
            ):
                low_zone_armed = False
                low_zone_armed_index = None
            if low_zone_armed:
                if (
                    np.isfinite(previous_rsi)
                    and previous_rsi <= config.rsi_oversold
                    and rsi_value > config.rsi_oversold
                ):
                    regime_rsi_recovered = True
                if (
                    np.isfinite(previous_percent_b)
                    and previous_percent_b <= 0.0
                    and percent_b > 0.0
                ):
                    regime_bollinger_recovered = True
            if raw_shares:
                rsi_overbought_armed |= rsi_value >= config.rsi_overbought
                bollinger_upper_armed |= percent_b >= 1.0
        contributions = _contributions(row, contribution_columns)
        factor_details = _contributions(row, detail_columns)
        regime_decision = None
        if config.timing_style in {
            "regime_reversion",
            "regime_reversion_legacy",
        }:
            indicator_ready = all(
                np.isfinite(value)
                for value in (
                    price_position,
                    rsi_value,
                    percent_b,
                    entry_score_final,
                    previous_entry_score_final,
                )
            )
            factor_confirmation = (
                indicator_ready
                and entry_score_final >= config.entry_score_threshold
                and entry_score_final > previous_entry_score_final
            )
            regime_decision = regime_entry_decision(
                mode=(
                    "legacy_all"
                    if config.timing_style
                    == "regime_reversion_legacy"
                    else config.regime_entry_mode
                ),
                candidate=low_zone_armed,
                regime_allowed=market_regime != "downtrend",
                price_allowed=(
                    np.isfinite(price_position)
                    and price_position <= config.entry_max_price_position
                ),
                rsi_confirmation=regime_rsi_recovered,
                bollinger_confirmation=regime_bollinger_recovered,
                factor_confirmation=factor_confirmation,
                confirmation_required=config.regime_confirmation_required,
            )
            entry_funnel["indicator_ready"] += int(indicator_ready)
            entry_funnel["candidate_zone"] += int(
                regime_decision.candidate
            )
            entry_funnel["regime_allowed"] += int(
                regime_decision.candidate
                and regime_decision.regime_allowed
            )
            entry_funnel["price_allowed"] += int(
                regime_decision.candidate
                and regime_decision.regime_allowed
                and regime_decision.price_allowed
            )
            entry_funnel["rsi_confirmation"] += int(
                regime_decision.rsi_confirmation
            )
            entry_funnel["bollinger_confirmation"] += int(
                regime_decision.bollinger_confirmation
            )
            entry_funnel["factor_confirmation"] += int(
                regime_decision.factor_confirmation
            )
            entry_funnel["confirmation_passed"] += int(
                regime_decision.passed
            )
        trace_row: dict[str, Any] = {
            "date": trading_date,
            "time": _close_time(trading_date),
            "adjusted_close": adjusted_close,
            "composite_score": score,
            "trend_score": trend,
            "buy_threshold": config.buy_threshold,
            "sell_threshold": config.sell_threshold,
            "timing_style": config.timing_style,
            "timing_price_position_60": price_position,
            "low_zone_threshold": config.low_zone_threshold,
            "low_recovery_threshold": config.low_recovery_threshold,
            "high_reversal_threshold": config.high_reversal_threshold,
            "high_zone_threshold": config.high_zone_threshold,
            "low_zone_armed": low_zone_armed,
            "high_zone_armed": high_zone_armed,
            "setup_age_sessions": (
                index - low_zone_armed_index
                if low_zone_armed and low_zone_armed_index is not None
                else None
            ),
            "setup_expiry_sessions": config.setup_expiry_sessions,
            "entry_max_price_position": config.entry_max_price_position,
            "exit_min_price_position": config.exit_min_price_position,
            "entry_score": entry_score,
            "exit_score": exit_score,
            "entry_score_final": entry_score_final,
            "exit_score_final": exit_score_final,
            "entry_score_threshold": config.entry_score_threshold,
            "exit_score_threshold": config.exit_score_threshold,
            "market_regime": market_regime,
            "ma_200": row.get("ma_200"),
            "ma_slope_20": row.get("ma_slope_20"),
            "distance_to_ma_200": row.get("distance_to_ma_200"),
            "rsi_14": rsi_value,
            "rsi_oversold": config.rsi_oversold,
            "rsi_overbought": config.rsi_overbought,
            "bollinger_mid_20": row.get("bollinger_mid_20"),
            "bollinger_upper_20": row.get("bollinger_upper_20"),
            "bollinger_lower_20": row.get("bollinger_lower_20"),
            "bollinger_percent_b_20": percent_b,
            "bollinger_bandwidth_20": row.get("bollinger_bandwidth_20"),
            "atr_20": atr_value,
            "donchian_upper": donchian_upper,
            "donchian_lower": donchian_lower,
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "ma_slow_slope": ma_slow_slope,
            "atr_initial_stop_line": (
                entry_adjusted_price
                - config.atr_stop_multiple * entry_atr
                if raw_shares
                and np.isfinite(entry_adjusted_price)
                and np.isfinite(entry_atr)
                else None
            ),
            "atr_trailing_stop_line": (
                max(peak_adjusted_close, adjusted_close)
                - config.atr_trailing_multiple * atr_value
                if raw_shares
                and np.isfinite(peak_adjusted_close)
                and np.isfinite(adjusted_close)
                and np.isfinite(atr_value)
                else None
            ),
            "rsi_recovered": regime_rsi_recovered,
            "bollinger_recovered": regime_bollinger_recovered,
            "entry_candidate": (
                regime_decision.candidate
                if regime_decision is not None
                else None
            ),
            "entry_regime_allowed": (
                regime_decision.regime_allowed
                if regime_decision is not None
                else None
            ),
            "entry_price_allowed": (
                regime_decision.price_allowed
                if regime_decision is not None
                else None
            ),
            "entry_factor_confirmation": (
                regime_decision.factor_confirmation
                if regime_decision is not None
                else None
            ),
            "entry_confirmation_count": (
                regime_decision.confirmation_count
                if regime_decision is not None
                else None
            ),
            "entry_confirmation_required": (
                regime_decision.confirmation_required
                if regime_decision is not None
                else None
            ),
            "entry_confirmation_passed": (
                regime_decision.passed
                if regime_decision is not None
                else None
            ),
            "position": "long" if raw_shares else "flat",
            "factor_contributions": contributions,
            "factor_details": factor_details,
        }
        trace_row.update(contributions)
        score_trace.append(trace_row)

        if raw_shares and np.isfinite(adjusted_close) and adjusted_close > 0:
            # 新仓在本日开盘建立，日终立即按收盘价估值，使收益从成交开盘开始。
            last_adjusted_close = adjusted_close
            peak_adjusted_close = max(peak_adjusted_close, adjusted_close)

        previous_score = (
            _number(frame.iloc[index - 1]["composite_score"])
            if index > 0
            else float("nan")
        )
        reason_keys: list[str] = []
        trend_buy = (
            config.timing_style == "trend"
            and index > 0
            and np.isfinite(previous_score)
            and np.isfinite(score)
            and previous_score < config.buy_threshold <= score
            and np.isfinite(trend)
            and trend > 0
        )
        mean_reversion_buy = (
            config.timing_style == "mean_reversion"
            and low_zone_armed
            and np.isfinite(previous_price_position)
            and np.isfinite(price_position)
            and price_position >= config.low_recovery_threshold
            and price_position <= config.entry_max_price_position
            and price_position > previous_price_position
        )
        factor_dual_buy = (
            config.timing_style == "factor_dual"
            and low_zone_armed
            and np.isfinite(previous_price_position)
            and np.isfinite(price_position)
            and price_position >= config.low_recovery_threshold
            and price_position <= config.entry_max_price_position
            and price_position > previous_price_position
            and np.isfinite(previous_entry_score)
            and np.isfinite(entry_score)
            and entry_score >= config.entry_score_threshold
            and entry_score > previous_entry_score
        )
        regime_reversion_buy = bool(
            config.timing_style
            in {"regime_reversion", "regime_reversion_legacy"}
            and regime_decision is not None
            and regime_decision.passed
        )
        rsi_bollinger_buy = (
            config.timing_style == "rsi_bollinger"
            and low_zone_armed
            and regime_rsi_recovered
            and regime_bollinger_recovered
        )
        donchian_candidate = (
            config.timing_style == "donchian_atr"
            and np.isfinite(adjusted_close)
            and np.isfinite(donchian_upper)
            and adjusted_close > donchian_upper
        )
        donchian_filter_passed = (
            not config.donchian_trend_filter
            or (
                _valid_price(row.get("ma_200"))
                and adjusted_close > _number(row.get("ma_200"))
                and _number(row.get("ma_slope_20")) > 0
            )
        )
        donchian_buy = (
            donchian_candidate
            and donchian_entry(
                close=adjusted_close,
                upper=donchian_upper,
                trend_filter_passed=donchian_filter_passed,
            )
        )
        ma_cross_candidate = (
            config.timing_style == "ma_crossover_atr"
            and all(
                np.isfinite(value)
                for value in (
                    previous_ma_fast,
                    previous_ma_slow,
                    ma_fast,
                    ma_slow,
                )
            )
            and previous_ma_fast <= previous_ma_slow
            and ma_fast > ma_slow
        )
        ma_crossover_buy = (
            ma_cross_candidate
            and np.isfinite(ma_slow_slope)
            and moving_average_entry(
                previous_fast=previous_ma_fast,
                previous_slow=previous_ma_slow,
                fast=ma_fast,
                slow=ma_slow,
                slow_slope=ma_slow_slope,
            )
        )
        buy_and_hold_buy = (
            config.timing_style == "buy_and_hold"
            and entry_funnel["orders_filled"] == 0
        )
        ma200_buy = (
            config.timing_style == "ma_200"
            and np.isfinite(adjusted_close)
            and _valid_price(row.get("ma_200"))
            and adjusted_close > _number(row.get("ma_200"))
        )
        if config.timing_style == "donchian_atr":
            entry_funnel["indicator_ready"] += int(
                np.isfinite(donchian_upper) and np.isfinite(atr_value)
            )
            entry_funnel["candidate_zone"] += int(donchian_candidate)
            entry_funnel["regime_allowed"] += int(
                donchian_candidate and donchian_filter_passed
            )
            entry_funnel["price_allowed"] += int(donchian_candidate)
            entry_funnel["confirmation_passed"] += int(donchian_buy)
        elif config.timing_style == "ma_crossover_atr":
            entry_funnel["indicator_ready"] += int(
                all(
                    np.isfinite(value)
                    for value in (ma_fast, ma_slow, ma_slow_slope, atr_value)
                )
            )
            entry_funnel["candidate_zone"] += int(ma_cross_candidate)
            entry_funnel["regime_allowed"] += int(
                ma_cross_candidate and ma_slow_slope > 0
            )
            entry_funnel["price_allowed"] += int(ma_cross_candidate)
            entry_funnel["confirmation_passed"] += int(
                ma_crossover_buy
            )
        if (
            pending is None
            and raw_shares == 0
            and (
                trend_buy
                or mean_reversion_buy
                or factor_dual_buy
                or regime_reversion_buy
                or rsi_bollinger_buy
                or donchian_buy
                or ma_crossover_buy
                or buy_and_hold_buy
                or ma200_buy
            )
        ):
            reason_keys = [
                "score_cross_up"
                if trend_buy
                else (
                    "low_zone_recovery"
                    if mean_reversion_buy
                    else (
                        "entry_factor_confirmation"
                        if factor_dual_buy
                        else (
                            "regime_entry_confirmation"
                            if regime_reversion_buy
                            else (
                                "rsi_bollinger_entry"
                                if rsi_bollinger_buy
                                else (
                                    "donchian_breakout"
                                    if donchian_buy
                                    else (
                                        "ma_crossover"
                                        if ma_crossover_buy
                                        else (
                                            "buy_and_hold_entry"
                                            if buy_and_hold_buy
                                            else "ma200_entry"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ]
            cooldown_active = (
                last_exit_index is not None
                and index - last_exit_index <= config.cooldown_sessions
            )
            order = {
                "side": "buy",
                "reason": reason_keys[0],
                "reason_code": _REASON_CN[reason_keys[0]],
                "reason_codes": [_REASON_CN[key] for key in reason_keys],
                "signal_date": trading_date,
                "composite_score": score,
                "trend_score": trend,
                "timing_style": config.timing_style,
                "timing_price_position_60": price_position,
                "entry_score": entry_score,
                "exit_score": exit_score,
                "entry_score_final": entry_score_final,
                "exit_score_final": exit_score_final,
                "market_regime": market_regime,
                "ma_200": row.get("ma_200"),
                "ma_slope_20": row.get("ma_slope_20"),
                "distance_to_ma_200": row.get("distance_to_ma_200"),
                "rsi_14": rsi_value,
                "bollinger_percent_b_20": percent_b,
                "atr_20": atr_value,
                "donchian_upper": donchian_upper,
                "donchian_lower": donchian_lower,
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "ma_slow_slope": ma_slow_slope,
                "factor_contributions": contributions,
                "factor_details": factor_details,
            }
            if cooldown_active:
                blocked_orders.append(
                    _blocked_order(order, dates[index + 1] if index + 1 < len(dates) else None, "cooldown")
                )
            elif index + 1 >= len(dates):
                blocked_orders.append(
                    _blocked_order(order, None, "no_next_session")
                )
            else:
                pending = order
                entry_funnel["orders_created"] += 1
        elif pending is None and raw_shares and entry_index is not None:
            held_sessions = index - entry_index
            if held_sessions >= config.minimum_holding_sessions:
                cta_style = config.timing_style in {
                    "donchian_atr",
                    "ma_crossover_atr",
                }
                signal_only_baseline = config.timing_style in {
                    "buy_and_hold",
                    "ma_200",
                }
                fixed_stop_hit = (
                    not cta_style
                    and not signal_only_baseline
                    and
                    np.isfinite(adjusted_close)
                    and adjusted_close
                    <= entry_adjusted_price * (1.0 - config.fixed_stop)
                )
                trailing_stop_hit = (
                    not cta_style
                    and not signal_only_baseline
                    and
                    np.isfinite(adjusted_close)
                    and adjusted_close
                    <= peak_adjusted_close * (1.0 - config.trailing_stop)
                )
                atr_initial_stop_hit = (
                    cta_style
                    and np.isfinite(adjusted_close)
                    and np.isfinite(entry_atr)
                    and adjusted_close
                    <= entry_adjusted_price
                    - config.atr_stop_multiple * entry_atr
                )
                atr_trailing_stop_hit = (
                    cta_style
                    and np.isfinite(adjusted_close)
                    and np.isfinite(peak_adjusted_close)
                    and np.isfinite(atr_value)
                    and adjusted_close
                    <= peak_adjusted_close
                    - config.atr_trailing_multiple * atr_value
                )
                score_exit = (
                    np.isfinite(previous_score)
                    and np.isfinite(score)
                    and previous_score > config.sell_threshold >= score
                )
                trend_exit = np.isfinite(trend) and trend < 0
                high_reversal_exit = (
                    config.timing_style == "mean_reversion"
                    and high_zone_armed
                    and np.isfinite(previous_price_position)
                    and np.isfinite(price_position)
                    and price_position <= config.high_reversal_threshold
                    and price_position < previous_price_position
                )
                factor_dual_exit = (
                    config.timing_style == "factor_dual"
                    and high_zone_armed
                    and np.isfinite(exit_score)
                    and exit_score >= config.exit_score_threshold
                )
                regime_score_exit = (
                    config.timing_style
                    in {
                        "regime_reversion",
                        "regime_reversion_legacy",
                        "rsi_bollinger",
                    }
                    and np.isfinite(price_position)
                    and price_position >= config.exit_min_price_position
                    and np.isfinite(exit_score_final)
                    and exit_score_final >= config.exit_score_threshold
                )
                rsi_reversal_exit = (
                    config.timing_style
                    in {
                        "regime_reversion",
                        "regime_reversion_legacy",
                        "rsi_bollinger",
                    }
                    and rsi_overbought_armed
                    and np.isfinite(previous_rsi)
                    and np.isfinite(rsi_value)
                    and rsi_value < previous_rsi
                )
                bollinger_reversal_exit = (
                    config.timing_style
                    in {"regime_reversion", "regime_reversion_legacy"}
                    and bollinger_upper_armed
                    and np.isfinite(previous_percent_b)
                    and np.isfinite(percent_b)
                    and percent_b < 1.0
                    and percent_b < previous_percent_b
                )
                ma_breakdown_exit = (
                    config.timing_style
                    in {"regime_reversion", "regime_reversion_legacy"}
                    and market_regime == "downtrend"
                )
                donchian_channel_exit = (
                    config.timing_style == "donchian_atr"
                    and np.isfinite(adjusted_close)
                    and np.isfinite(donchian_lower)
                    and donchian_exit(
                        close=adjusted_close,
                        lower=donchian_lower,
                    )
                )
                ma_cross_exit = (
                    config.timing_style == "ma_crossover_atr"
                    and all(
                        np.isfinite(value)
                        for value in (
                            previous_ma_fast,
                            previous_ma_slow,
                            ma_fast,
                            ma_slow,
                        )
                    )
                    and moving_average_exit(
                        previous_fast=previous_ma_fast,
                        previous_slow=previous_ma_slow,
                        fast=ma_fast,
                        slow=ma_slow,
                    )
                )
                ma200_exit = (
                    config.timing_style == "ma_200"
                    and np.isfinite(adjusted_close)
                    and _valid_price(row.get("ma_200"))
                    and adjusted_close < _number(row.get("ma_200"))
                )
                max_holding_exit = (
                    config.timing_style != "buy_and_hold"
                    and held_sessions >= config.max_holding_sessions
                )
                stale_exit = stale_sessions > config.max_stale_sessions
                reason_keys = [
                    key
                    for key, hit in (
                        ("fixed_stop", fixed_stop_hit),
                        ("trailing_stop", trailing_stop_hit),
                        ("atr_initial_stop", atr_initial_stop_hit),
                        ("atr_trailing_stop", atr_trailing_stop_hit),
                        (
                            "score_cross_down",
                            config.timing_style == "trend" and score_exit,
                        ),
                        (
                            "trend_negative",
                            config.timing_style == "trend" and trend_exit,
                        ),
                        ("high_zone_reversal", high_reversal_exit),
                        ("exit_factor_risk", factor_dual_exit),
                        ("regime_exit_risk", regime_score_exit),
                        ("rsi_overbought_reversal", rsi_reversal_exit),
                        (
                            "bollinger_upper_reversal",
                            bollinger_reversal_exit,
                        ),
                        ("ma200_breakdown", ma_breakdown_exit),
                        ("donchian_exit", donchian_channel_exit),
                        ("ma_crossdown", ma_cross_exit),
                        ("ma200_exit", ma200_exit),
                        ("max_holding", max_holding_exit),
                        ("stale_data", stale_exit),
                    )
                    if hit
                ]
                if reason_keys:
                    order = {
                        "side": "sell",
                        "reason": reason_keys[0],
                        "reason_code": _REASON_CN[reason_keys[0]],
                        "reason_codes": [_REASON_CN[key] for key in reason_keys],
                        "signal_date": trading_date,
                        "composite_score": score,
                        "trend_score": trend,
                        "timing_style": config.timing_style,
                        "timing_price_position_60": price_position,
                        "entry_score": entry_score,
                        "exit_score": exit_score,
                        "entry_score_final": entry_score_final,
                        "exit_score_final": exit_score_final,
                        "market_regime": market_regime,
                        "ma_200": row.get("ma_200"),
                        "ma_slope_20": row.get("ma_slope_20"),
                        "distance_to_ma_200": row.get(
                            "distance_to_ma_200"
                        ),
                        "rsi_14": rsi_value,
                        "bollinger_percent_b_20": percent_b,
                        "atr_20": atr_value,
                        "donchian_upper": donchian_upper,
                        "donchian_lower": donchian_lower,
                        "ma_fast": ma_fast,
                        "ma_slow": ma_slow,
                        "ma_slow_slope": ma_slow_slope,
                        "factor_contributions": contributions,
                        "factor_details": factor_details,
                    }
                    if index + 1 >= len(dates):
                        blocked_orders.append(
                            _blocked_order(order, None, "no_next_session", raw_shares)
                        )
                    else:
                        pending = order

        if reason_keys:
            is_buy_signal = reason_keys[0] in {
                "score_cross_up",
                "low_zone_recovery",
                "entry_factor_confirmation",
                "regime_entry_confirmation",
                "rsi_bollinger_entry",
                "donchian_breakout",
                "ma_crossover",
                "buy_and_hold_entry",
                "ma200_entry",
            }
            signal_row: dict[str, Any] = {
                "symbol": symbol,
                "side": "buy" if is_buy_signal else "sell",
                "signal_date": trading_date,
                "signal_time": _close_time(trading_date),
                "composite_score": score,
                "trend_score": trend,
                "timing_style": config.timing_style,
                "timing_price_position_60": price_position,
                "entry_score": entry_score,
                "exit_score": exit_score,
                "entry_score_final": entry_score_final,
                "exit_score_final": exit_score_final,
                "market_regime": market_regime,
                "ma_200": row.get("ma_200"),
                "ma_slope_20": row.get("ma_slope_20"),
                "rsi_14": rsi_value,
                "bollinger_percent_b_20": percent_b,
                "atr_20": atr_value,
                "donchian_upper": donchian_upper,
                "donchian_lower": donchian_lower,
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "ma_slow_slope": ma_slow_slope,
                "adjusted_close": adjusted_close,
                "reason": reason_keys[0],
                "reason_code": _REASON_CN[reason_keys[0]],
                "reason_codes": [_REASON_CN[key] for key in reason_keys],
                "factor_contributions": contributions,
                "factor_details": factor_details,
            }
            signal_row.update(contributions)
            signals.append(signal_row)

        marked_value = (
            accounting_quantity * last_adjusted_close
            if raw_shares and np.isfinite(last_adjusted_close)
            else 0.0
        )
        equity = cash + marked_value
        day_turnover = daily_notional / pre_trade_equity if pre_trade_equity > 0 else 0.0
        turnover += day_turnover
        equity_rows.append(
            {
                "date": trading_date,
                "equity": equity,
                "net_value": equity / config.initial_capital,
                "cash": cash,
                "position_value": marked_value,
                "raw_shares": raw_shares,
                "accounting_quantity": accounting_quantity,
                "adjusted_close": (
                    last_adjusted_close if np.isfinite(last_adjusted_close) else None
                ),
                "stale_sessions": stale_sessions,
                "turnover": day_turnover,
            }
        )

    equity_curve = pd.DataFrame(equity_rows)
    equity_curve["daily_return"] = equity_curve["equity"].pct_change(fill_method=None)
    metrics = calculate_metrics(equity_curve)
    sell_trades = [trade for trade in trades if trade["side"] == "sell"]
    winning_returns = [
        float(trade["return"])
        for trade in sell_trades
        if trade["return"] is not None and float(trade["return"]) > 0
    ]
    losing_returns = [
        float(trade["return"])
        for trade in sell_trades
        if trade["return"] is not None and float(trade["return"]) < 0
    ]
    holding_sessions = [
        int(trade["holding_sessions"])
        for trade in sell_trades
        if trade["holding_sessions"] is not None
    ]
    total_fees = total_commission + total_stamp_duty
    market_exposure = float(
        np.mean(
            pd.to_numeric(
                equity_curve["position_value"], errors="coerce"
            ).fillna(0.0).gt(0.0)
        )
    )
    evidence_warnings: list[str] = []
    if len(sell_trades) < 5:
        evidence_warnings.append(
            "Fewer than five closed trades; trade statistics are insufficient."
        )
    if market_exposure < 0.10:
        evidence_warnings.append(
            "Market exposure is below 10%; results are dominated by time in cash."
        )
    summary = {
        key: metrics[key]
        for key in (
            "total_return",
            "annualized_return",
            "sharpe",
            "max_drawdown",
            "volatility",
        )
    }
    summary.update(
        {
            "initial_capital": config.initial_capital,
            "timing_style": config.timing_style,
            "position_sizing": config.position_sizing,
            "ending_equity": float(equity_curve["equity"].iloc[-1]),
            "trade_count": len(trades),
            "round_trip_count": len(sell_trades),
            "win_rate": (
                sum(float(trade["return"]) > 0 for trade in sell_trades)
                / len(sell_trades)
                if sell_trades
                else float("nan")
            ),
            "profit_loss_ratio": (
                float(np.mean(winning_returns)) / abs(float(np.mean(losing_returns)))
                if winning_returns and losing_returns
                else float("nan")
            ),
            "average_holding_sessions": (
                float(np.mean(holding_sessions))
                if holding_sessions
                else float("nan")
            ),
            "market_exposure": market_exposure,
            "evidence_sufficient": not evidence_warnings,
            "evidence_warnings": evidence_warnings,
            "entry_funnel": dict(entry_funnel),
            "current_state": "long" if raw_shares else "flat",
            "market_regime": (
                score_trace[-1].get("market_regime")
                if score_trace
                else None
            ),
            "last_reason": (
                signals[-1]["reason_code"] if signals else None
            ),
            "blocked_order_count": len(blocked_orders),
            "turnover": turnover,
            "commission": total_commission,
            "stamp_duty": total_stamp_duty,
            "fees": total_fees,
            "slippage_cost": total_slippage_cost,
            "total_cost": total_fees + total_slippage_cost,
        }
    )

    warnings = [
        "Adjusted-price accounting quantity is derived from raw execution "
        "notional divided by the adjusted open. It is an approximation around "
        "corporate actions because the input has no historical share-conversion ledger.",
        "Signals are assumed to be precomputed causally; the engine never recalculates "
        "them and therefore cannot detect look-ahead embedded by the caller.",
        *evidence_warnings,
    ]
    result = {
        "execution_policy": (
            "Signal at T close; one queued order executes only at the next supplied "
            "session's 09:30 raw open with side-specific slippage. A stock bought "
            "on one session cannot be sold until the next trading session. "
            f"Timing style: {config.timing_style}."
        ),
        "accounting_policy": (
            "NAV uses fixed adjusted accounting quantity derived from actual raw "
            "execution notional; corporate-action share changes are approximate."
        ),
        "warnings": warnings,
        "summary": summary,
        "metrics": summary,
        "equity_curve": equity_curve.to_dict(orient="records"),
        "score_trace": score_trace,
        "signals": signals,
        "trades": trades,
        "blocked_orders": blocked_orders,
        "blocked_trades": blocked_orders,
    }
    return json_safe(result)


run_timing_backtest = run_timing


__all__ = [
    "TimingConfig",
    "run_timing",
    "run_timing_backtest",
]
