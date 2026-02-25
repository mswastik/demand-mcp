"""
metrics.py — Core metric computation functions.

All functions operate on polars DataFrames and return polars DataFrames.
They do NOT touch I/O or the DataStore — they are pure transformations.

Accuracy formula (per group):
    Accuracy = 1 - Sum(Abs Error) / Sum(Act Orders Vol)

Bias formula (per group):
    Bias = Sum(Act Orders Vol) - Sum(Forecast Vol)

FVA formula (per group):
    FVA = DF Accuracy - Stat Accuracy
    (positive = DF beats Stat, negative = Stat beats DF)
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from data import (
    ACT_VOL,
    ALL_LAGS,
    FORECAST_LEVEL_COL,
    LAG_DF_COLS,
    LAG_STAT_COLS,
    apply_filters,
    filter_date_window,
    filter_to_actuals_period,
)

LagName = Literal["L2", "L1", "L0", "Fcst"]

# ── Low-level metric builders ────────────────────────────────────────────────

def _abs_error_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Append abs error columns for all lags, both DF and Stat."""
    exprs = []
    for lag in ALL_LAGS:
        _, df_vol = LAG_DF_COLS[lag]
        _, stat_vol = LAG_STAT_COLS[lag]
        exprs += [
            (pl.col(ACT_VOL) - pl.col(df_vol)).abs().alias(f"{lag} DF Abs Err"),
            (pl.col(ACT_VOL) - pl.col(stat_vol)).abs().alias(f"{lag} Stat Abs Err"),
            (pl.col(ACT_VOL) - pl.col(df_vol)).alias(f"{lag} DF Bias Raw"),
            (pl.col(ACT_VOL) - pl.col(stat_vol)).alias(f"{lag} Stat Bias Raw"),
        ]
    return df.with_columns(exprs)


def _aggregate_metrics(
    df: pl.DataFrame,
    group_by: list[str],
) -> pl.DataFrame:
    """
    Aggregate sum of actuals, abs errors, and bias raws by group_by columns.
    Returns one row per group with all lag metric sums.
    """
    agg_exprs = [pl.col(ACT_VOL).sum().alias("Sum Act Vol")]
    for lag in ALL_LAGS:
        _, df_vol = LAG_DF_COLS[lag]
        _, stat_vol = LAG_STAT_COLS[lag]
        agg_exprs += [
            pl.col(f"{lag} DF Abs Err").sum().alias(f"{lag} DF Sum Abs Err"),
            pl.col(f"{lag} Stat Abs Err").sum().alias(f"{lag} Stat Sum Abs Err"),
            pl.col(f"{lag} DF Bias Raw").sum().alias(f"{lag} DF Sum Bias"),
            pl.col(f"{lag} Stat Bias Raw").sum().alias(f"{lag} Stat Sum Bias"),
            pl.col(df_vol).sum().alias(f"{lag} DF Sum Fcst"),
            pl.col(stat_vol).sum().alias(f"{lag} Stat Sum Fcst"),
        ]

    return df.group_by(group_by).agg(agg_exprs)


def _compute_derived_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    From summed columns, derive:
      - Accuracy   = 1 - Sum(Abs Err) / Sum(Act Vol)
      - Bias %     = Sum(Bias Raw) / Sum(Act Vol)  [signed % of actuals]
      - Bias Units = Sum(Bias Raw)
      - FVA        = DF Accuracy - Stat Accuracy

    Guards:
      - Sum Act Vol == 0 → Accuracy and Bias % set to null (not inf/NaN)
      - Accuracy clamped to [-1.0, 1.0] to handle extreme outlier rows
        (abs error > 2× actuals gives accuracy < -1, which is uninformative)
    """
    exprs = []
    for lag in ALL_LAGS:
        act = pl.col("Sum Act Vol")
        df_abs = pl.col(f"{lag} DF Sum Abs Err")
        stat_abs = pl.col(f"{lag} Stat Sum Abs Err")
        df_bias = pl.col(f"{lag} DF Sum Bias")
        stat_bias = pl.col(f"{lag} Stat Sum Bias")

        # Safe division: return null when actuals = 0
        safe_act = pl.when(act == 0.0).then(None).otherwise(act)

        df_acc = (
            (1 - df_abs / safe_act)
            .clip(-1.0, 1.0)
            .alias(f"{lag} DF Accuracy")
        )
        stat_acc = (
            (1 - stat_abs / safe_act)
            .clip(-1.0, 1.0)
            .alias(f"{lag} Stat Accuracy")
        )
        fva = (
            ((1 - df_abs / safe_act) - (1 - stat_abs / safe_act))
            .clip(-2.0, 2.0)
            .alias(f"{lag} FVA")
        )
        df_bias_pct   = (df_bias / safe_act).alias(f"{lag} DF Bias %")
        stat_bias_pct = (stat_bias / safe_act).alias(f"{lag} Stat Bias %")
        df_bias_units = df_bias.alias(f"{lag} DF Bias Units")
        stat_bias_units = stat_bias.alias(f"{lag} Stat Bias Units")

        exprs += [df_acc, stat_acc, fva, df_bias_pct, stat_bias_pct,
                  df_bias_units, stat_bias_units]

    return df.with_columns(exprs)


# ── Public API ───────────────────────────────────────────────────────────────

def compute_metrics(
    df_raw: pl.DataFrame,  # already normalized by DataStore (Forecast Level always present)
    group_by_cols: list[str],
    filters: dict | None = None,
    window: str | None = None,
) -> pl.DataFrame:
    """
    Main entry point for metric computation.

    Parameters
    ----------
    df_raw : normalized DataFrame from DataStore
    group_by_cols : list of column names to group by
    filters : optional dict of {col: value} filters
    window : optional time window string ('last_month', 'last_3_months', etc.)

    Returns
    -------
    DataFrame with one row per group containing all lag metrics.
    """
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    if window:
        df = filter_date_window(df, window)

    if df.is_empty():
        return pl.DataFrame()

    df = _abs_error_cols(df)
    agg = _aggregate_metrics(df, group_by_cols)
    result = _compute_derived_metrics(agg)
    return result


def compute_accuracy_trend(
    df_raw: pl.DataFrame,
    group_by_cols: list[str],
    filters: dict | None = None,
    window: str = "last_12_months",
    lag: str = "L2",
) -> pl.DataFrame:
    """
    Monthly accuracy trend. Returns one row per (group, month).
    """
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    df = filter_date_window(df, window)

    if df.is_empty():
        return pl.DataFrame()

    df = _abs_error_cols(df)
    time_groups = group_by_cols + ["SALES_DATE"]
    agg = _aggregate_metrics(df, time_groups)
    result = _compute_derived_metrics(agg)
    return result.sort(time_groups)


def compute_yoy_growth(
    df_raw: pl.DataFrame,
    group_by_cols: list[str],
    filters: dict | None = None,
) -> pl.DataFrame:
    """
    Year-over-year growth comparison.
    Returns one row per (group, year) with:
      - Sum Act Vol, Sum DF Fcst (Fcst lag), Sum Stat Fcst (Fcst lag)
      - YoY growth % for each
    """
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    if df.is_empty():
        return pl.DataFrame()

    df = df.with_columns(pl.col("SALES_DATE").dt.year().alias("Year"))
    year_groups = group_by_cols + ["Year"]

    from data import FCST_DF_VOL, FCST_STAT_VOL
    agg = df.group_by(year_groups).agg([
        pl.col(ACT_VOL).sum().alias("Sum Act Vol"),
        pl.col(FCST_DF_VOL).sum().alias("Sum DF Fcst Vol"),
        pl.col(FCST_STAT_VOL).sum().alias("Sum Stat Fcst Vol"),
    ]).sort(year_groups)

    # Compute YoY growth % using shift within each group
    # We do this in Python for clarity
    results = []
    group_keys = group_by_cols if group_by_cols else []

    if group_keys:
        for group_vals, grp_df in agg.group_by(group_keys):
            grp_df = grp_df.sort("Year")
            grp_df = _add_yoy_cols(grp_df)
            results.append(grp_df)
        return pl.concat(results).sort(year_groups)
    else:
        agg = agg.sort("Year")
        return _add_yoy_cols(agg)


def _add_yoy_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Add YoY % change columns using shift."""
    for col, alias in [
        ("Sum Act Vol", "Act YoY %"),
        ("Sum DF Fcst Vol", "DF Fcst YoY %"),
        ("Sum Stat Fcst Vol", "Stat Fcst YoY %"),
    ]:
        prev = df[col].shift(1)
        curr = df[col]
        yoy = ((curr - prev) / prev * 100).alias(alias)
        df = df.with_columns(yoy)
    return df


def compute_forecast_evolution(
    df_raw: pl.DataFrame,
    filters: dict | None = None,
    window: str | None = None,
) -> pl.DataFrame:
    """
    Show how DF forecast changed across lags (L2 → L1 → L0 → Fcst) vs Actuals.
    Returns one row per SALES_DATE with all lag sums + actuals.
    """
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    if window:
        df = filter_date_window(df, window)

    if df.is_empty():
        return pl.DataFrame()

    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS

    agg_exprs = [pl.col(ACT_VOL).sum().alias("Sum Act Vol")]
    for lag in ALL_LAGS:
        _, df_vol = LAG_DF_COLS[lag]
        _, stat_vol = LAG_STAT_COLS[lag]
        agg_exprs += [
            pl.col(df_vol).sum().alias(f"{lag} DF Sum Fcst"),
            pl.col(stat_vol).sum().alias(f"{lag} Stat Sum Fcst"),
        ]

    return df.group_by("SALES_DATE").agg(agg_exprs).sort("SALES_DATE")
    
def compute_forecast_evolution_accuracy(
    df_raw: pl.DataFrame,
    filters: dict | None = None,
    window: str | None = None,
) -> pl.DataFrame:
    """
    Show how accuracy changed across lags (L2 → L1 → L0) per month.
    Follows the portfolio accuracy formula: 1 - Sum(Abs Err) / Sum(Act Vol).
    """
    df = filter_to_actuals_period(df_raw)
    if filters:
        df = apply_filters(df, filters)
    if window:
        df = filter_date_window(df, window)
    if df.is_empty():
        return pl.DataFrame()

    from data import LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS, FORECAST_LEVEL_COL

    # 1. Pre-aggregate at SKU-month grain
    group_cols = [FORECAST_LEVEL_COL, "CatalogNumber", "SALES_DATE"]
    group_cols = [c for c in group_cols if c in df.columns]
    vol_cols = [ACT_VOL]
    for lag in ALL_LAGS:
        vol_cols.append(LAG_DF_COLS[lag][1])
        vol_cols.append(LAG_STAT_COLS[lag][1])

    df_pre = df.group_by(group_cols).agg([pl.col(c).sum() for c in vol_cols])

    # 2. Compute absolute errors for all lags
    for lag in ALL_LAGS:
        act = pl.col(ACT_VOL)
        df_fcst = pl.col(LAG_DF_COLS[lag][1])
        stat_fcst = pl.col(LAG_STAT_COLS[lag][1])

        df_pre = df_pre.with_columns([
            (act - df_fcst).abs().alias(f"{lag} DF Abs Err"),
            (act - stat_fcst).abs().alias(f"{lag} Stat Abs Err"),
        ])

    # 3. Sum everything by SALES_DATE
    agg_exprs = [pl.col(ACT_VOL).sum().alias("Sum Act Vol")]
    for lag in ALL_LAGS:
        agg_exprs += [
            pl.col(f"{lag} DF Abs Err").sum().alias(f"{lag} DF Sum Abs Err"),
            pl.col(f"{lag} Stat Abs Err").sum().alias(f"{lag} Stat Sum Abs Err"),
        ]

    res = df_pre.group_by("SALES_DATE").agg(agg_exprs).sort("SALES_DATE")

    # 4. Compute Accuracy Ratios
    for lag in ALL_LAGS:
        act = pl.col("Sum Act Vol")
        # Safe division: null if act is 0
        safe_act = pl.when(act == 0.0).then(None).otherwise(act)

        res = res.with_columns([
            (1 - pl.col(f"{lag} DF Sum Abs Err") / safe_act).clip(-1.0, 1.0).alias(f"{lag} DF Acc"),
            (1 - pl.col(f"{lag} Stat Sum Abs Err") / safe_act).clip(-1.0, 1.0).alias(f"{lag} Stat Acc"),
        ])

    # Return only the useful columns
    final_cols = ["SALES_DATE", "Sum Act Vol"]
    for lag in ["L2", "L1", "L0"]:  # Focus on the historical lags
        if f"{lag} DF Acc" in res.columns:
            final_cols.extend([f"{lag} DF Acc", f"{lag} Stat Acc"])

    return res.select(final_cols)


def get_top_offenders(
    df_raw: pl.DataFrame,
    rank_by_col: str,  # hierarchy column to rank by
    metric: Literal["accuracy", "bias_df", "bias_stat", "abs_err_df", "abs_err_stat", "fva"] = "abs_err_df",
    lag: str = "L2",
    n: int = 10,
    filters: dict | None = None,
    window: str = "last_3_months",
    ascending: bool = True,  # True = worst first (high abs err, low accuracy)
) -> pl.DataFrame:
    """
    Rank entities at rank_by_col level by the chosen metric.

    Returns top n rows sorted worst-first by default.
    """
    df = compute_metrics(
        df_raw=df_raw,
        group_by_cols=[rank_by_col],
        filters=filters,
        window=window,
    )

    if df.is_empty():
        return pl.DataFrame()

    metric_col_map = {
        "accuracy": f"{lag} DF Accuracy",
        "bias_df": f"{lag} DF Bias %",
        "bias_stat": f"{lag} Stat Bias %",
        "abs_err_df": f"{lag} DF Sum Abs Err",
        "abs_err_stat": f"{lag} Stat Sum Abs Err",
        "fva": f"{lag} FVA",
    }

    sort_col = metric_col_map.get(metric)
    if sort_col is None or sort_col not in df.columns:
        raise ValueError(f"Metric '{metric}' not available. Choose from: {list(metric_col_map)}")

    return df.sort(sort_col, descending=not ascending).head(n)
