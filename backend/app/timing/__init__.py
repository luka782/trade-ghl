from .indicators import (
    bollinger_bands,
    distance_to_ma,
    distance_to_moving_average,
    ma,
    ma_slope,
    moving_average,
    moving_average_slope,
    rsi,
    wilder_rsi,
)


def __getattr__(name: str):
    if name in {"TimingConfig", "run_timing", "run_timing_backtest"}:
        from .engine import TimingConfig, run_timing, run_timing_backtest

        return {
            "TimingConfig": TimingConfig,
            "run_timing": run_timing,
            "run_timing_backtest": run_timing_backtest,
        }[name]
    raise AttributeError(name)

__all__ = [
    "TimingConfig",
    "bollinger_bands",
    "distance_to_ma",
    "distance_to_moving_average",
    "ma",
    "ma_slope",
    "moving_average",
    "moving_average_slope",
    "rsi",
    "run_timing",
    "run_timing_backtest",
    "wilder_rsi",
]
