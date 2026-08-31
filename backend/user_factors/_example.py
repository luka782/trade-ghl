"""Copy this file to a name without a leading underscore to enable it."""

import pandas as pd

from app.factors.base import Factor, FactorMetadata


class Reversal5Factor(Factor):
    metadata = FactorMetadata(
        name="reversal_5",
        description="Negative five-session return, known at signal close T.",
        lookback=5,
        required_columns=("close",),
    )

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        self.validate(bars)
        ordered = bars.sort_values(["symbol", "date"])
        values = ordered.groupby("symbol", sort=False)["close"].transform(
            lambda close: -(close / close.shift(5) - 1.0)
        )
        result = pd.Series(float("nan"), index=bars.index, dtype=float)
        result.loc[ordered.index] = values.to_numpy()
        return result


FACTOR = Reversal5Factor()
