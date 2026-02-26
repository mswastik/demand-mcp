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
    Year-over-year growth comparison with corrected current year calculation.
    
    Current year volume = Actuals (completed months) + DF/Stat Forecast (remaining months)
    This ensures fair YoY comparison since current year is incomplete.
    
    Returns one row per (group, year) with:
      - Sum Act Vol (historical years), Sum Act+Fcst Vol (current year)
      - Sum DF Fcst Vol, Sum Stat Fcst Vol
      - YoY growth % for each
    """
    from data import FCST_DF_VOL, FCST_STAT_VOL
    from datetime import date
    
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    if df.is_empty():
        return pl.DataFrame()

    # Determine current year and last completed month
    today = date.today()
    current_year = today.year
    last_completed_month = today.month - 1 if today.month > 1 else 12
    
    # Split data into historical (complete years) and current year
    df_with_year = df.with_columns([
        pl.col("SALES_DATE").dt.year().alias("Year"),
        pl.col("SALES_DATE").dt.month().alias("Month"),
    ])
    
    # For current year: use actuals for completed months, forecast for remaining
    df_current = df_with_year.filter(pl.col("Year") == current_year)
    df_historical = df_with_year.filter(pl.col("Year") < current_year)
    
    # Current year actuals (completed months only)
    df_current_actuals = df_current.filter(pl.col("Month") <= last_completed_month)
    
    # Current year forecast (remaining months) - use Fcst lag
    df_current_forecast = df_current.filter(pl.col("Month") > last_completed_month)
    
    year_groups = group_by_cols + ["Year"]
    
    # Historical years: sum actuals
    historical_agg = df_historical.group_by(year_groups).agg([
        pl.col(ACT_VOL).sum().alias("Sum Act Vol"),
        pl.col(FCST_DF_VOL).sum().alias("Sum DF Fcst Vol"),
        pl.col(FCST_STAT_VOL).sum().alias("Sum Stat Fcst Vol"),
    ])
    
    # Current year: actuals (completed months) + forecast (remaining months)
    if not df_current_actuals.is_empty():
        current_actuals_agg = df_current_actuals.group_by(year_groups).agg([
            pl.col(ACT_VOL).sum().alias("Sum Act Vol"),
            pl.col(FCST_DF_VOL).sum().alias("Sum DF Fcst Vol"),
            pl.col(FCST_STAT_VOL).sum().alias("Sum Stat Fcst Vol"),
        ])
    else:
        current_actuals_agg = pl.DataFrame(schema=historical_agg.schema)
    
    if not df_current_forecast.is_empty():
        # For remaining months, use forecast volumes as the "actual" equivalent
        current_forecast_agg = df_current_forecast.group_by(year_groups).agg([
            pl.col(FCST_DF_VOL).sum().alias("Sum DF Fcst Vol (Rem)"),
            pl.col(FCST_STAT_VOL).sum().alias("Sum Stat Fcst Vol (Rem)"),
        ])
        
        # Add forecast for remaining months to get total current year volume
        current_agg = current_actuals_agg.join(
            current_forecast_agg,
            on=group_by_cols if group_by_cols else ["Year"],
            how="left"
        ).with_columns([
            (pl.col("Sum Act Vol") + pl.col("Sum DF Fcst Vol (Rem)")).alias("Sum Act Vol (Incl Fcst)"),
            (pl.col("Sum DF Fcst Vol") + pl.col("Sum DF Fcst Vol (Rem)")).alias("Sum DF Fcst Vol (Total)"),
            (pl.col("Sum Stat Fcst Vol") + pl.col("Sum Stat Fcst Vol (Rem)")).alias("Sum Stat Fcst Vol (Total)"),
        ])
    else:
        current_agg = current_actuals_agg.with_columns([
            pl.col("Sum Act Vol").alias("Sum Act Vol (Incl Fcst)"),
            pl.col("Sum DF Fcst Vol").alias("Sum DF Fcst Vol (Total)"),
            pl.col("Sum Stat Fcst Vol").alias("Sum Stat Fcst Vol (Total)"),
        ])
    
    # Combine historical and current
    if not historical_agg.is_empty() and not current_agg.is_empty():
        # Rename columns to match
        current_final = current_agg.select([
            pl.col(c).alias(c) for c in historical_agg.columns
        ] + [
            pl.col("Sum Act Vol (Incl Fcst)").alias("Sum Act Vol"),
            pl.col("Sum DF Fcst Vol (Total)").alias("Sum DF Fcst Vol"),
            pl.col("Sum Stat Fcst Vol (Total)").alias("Sum Stat Fcst Vol"),
        ])
        combined = pl.concat([historical_agg, current_final])
    elif not current_agg.is_empty():
        combined = current_agg
    else:
        combined = historical_agg
    
    combined = combined.sort(year_groups)

    # Compute YoY growth % using shift within each group
    results = []
    group_keys = group_by_cols if group_by_cols else []

    if group_keys:
        for group_vals, grp_df in combined.group_by(group_keys):
            grp_df = grp_df.sort("Year")
            grp_df = _add_yoy_cols(grp_df)
            results.append(grp_df)
        return pl.concat(results).sort(year_groups)
    else:
        combined = combined.sort("Year")
        return _add_yoy_cols(combined)


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


def detect_anomalies_in_trend(
    df_raw: pl.DataFrame,
    group_by_cols: list[str],
    metric: Literal["accuracy", "bias", "volume"] = "accuracy",
    lag: str = "L2",
    window: str = "last_12_months",
    filters: dict | None = None,
    threshold_std: float = 1,
    return_all: bool = False,
    top_n: int | None = None,
) -> pl.DataFrame:
    """
    Detect anomalies in a metric trend over time using statistical methods.

    An anomaly is defined as a data point that deviates more than `threshold_std`
    standard deviations from the rolling mean (window=12 months).

    Parameters
    ----------
    df_raw : normalized DataFrame from DataStore
    group_by_cols : list of hierarchy columns to group by (SALES_DATE is always added)
    metric : 'accuracy' | 'bias' | 'volume' - which metric to analyze
    lag : 'L2' | 'L1' | 'L0' | 'Fcst' - which lag to use (for accuracy/bias)
    window : time window for the analysis
    filters : optional dict of {col: value} filters
    threshold_std : number of standard deviations for anomaly threshold (default: 1)
                    Lower = more sensitive (more anomalies), Higher = fewer anomalies
    return_all : if True, return all data points; if False (default), return only anomalies
    top_n : if set, return only top N anomalies sorted by deviation magnitude

    Returns
    -------
    DataFrame with one row per (group, month) including:
      - metric value
      - rolling_mean, rolling_std
      - lower_bound, upper_bound
      - is_anomaly (boolean)
      - anomaly_direction ('high' | 'low' | None)
      - deviation_magnitude (absolute z-score for ranking)
    
    Notes
    -----
    - If no anomalies are found and top_n is not specified, returns top 10 deviations
    - This ensures the function always returns useful data instead of empty results
    """
    df = filter_to_actuals_period(df_raw)

    if filters:
        df = apply_filters(df, filters)

    df = filter_date_window(df, window)

    if df.is_empty():
        return pl.DataFrame()

    # Compute the base metrics
    df_metrics = compute_accuracy_trend(
        df_raw=df_raw,
        group_by_cols=group_by_cols,
        filters=filters,
        window=window,
        lag=lag,
    )

    if df_metrics.is_empty():
        return pl.DataFrame()

    # Select the appropriate column based on metric type
    if metric == "accuracy":
        value_col = f"{lag} DF Accuracy"
    elif metric == "bias":
        value_col = f"{lag} DF Bias %"
    elif metric == "volume":
        value_col = "Sum Act Vol"
    else:
        raise ValueError(f"Unknown metric type: {metric}")

    if value_col not in df_metrics.columns:
        return pl.DataFrame()

    # Compute rolling statistics and detect anomalies per group
    time_col = "SALES_DATE"
    result_dfs = []

    group_cols = [c for c in group_by_cols if c in df_metrics.columns]
    
    # Debug info
    print(f"[DEBUG] detect_anomalies_in_trend: df_metrics shape={df_metrics.shape}")
    print(f"[DEBUG] group_cols={group_cols}, value_col={value_col}")
    print(f"[DEBUG] df_metrics null count in {value_col}: {df_metrics[value_col].null_count()}")

    if group_cols:
        group_count = 0
        for group_vals, grp_df in df_metrics.group_by(group_cols):
            grp_df = grp_df.sort(time_col)
            grp_df = _add_anomaly_columns(grp_df, value_col, threshold_std, debug=True)
            result_dfs.append(grp_df)
            group_count += 1
            if group_count <= 2:  # Only print first 2 groups to avoid spam
                print(f"[DEBUG] Processed group: {group_vals}, rows={grp_df.height}")
        print(f"[DEBUG] Total groups processed: {group_count}")
    else:
        grp_df = df_metrics.sort(time_col)
        grp_df = _add_anomaly_columns(grp_df, value_col, threshold_std, debug=True)
        result_dfs.append(grp_df)

    result = pl.concat(result_dfs).sort(group_cols + [time_col])

    # Filter out rows with null deviation_magnitude (zero volume items, no valid stats)
    print(f"[DEBUG] Before filtering null deviation_magnitude: {result.height} rows")
    result = result.filter(pl.col("deviation_magnitude").is_not_null())
    print(f"[DEBUG] After filtering null deviation_magnitude: {result.height} rows")

    # Filter to only anomalies if requested
    if not return_all:
        print(f"[DEBUG] Filtering to anomalies only (return_all=False)")
        result = result.filter(pl.col("is_anomaly"))
        print(f"[DEBUG] After anomaly filter: {result.height} rows")

    # Return top N anomalies if requested (even if no strict anomalies, return highest deviations)
    if top_n is not None and not result.is_empty():
        result = result.sort("deviation_magnitude", descending=True).head(top_n)
        print(f"[DEBUG] Returning top {top_n}: {result.height} rows")
    elif top_n is not None and result.is_empty():
        # No anomalies found, but user wants top N - return highest deviations from all data
        print(f"[DEBUG] No anomalies found, fetching top {top_n} from all data")
        all_result = pl.concat(result_dfs).sort(group_cols + [time_col])
        all_result = all_result.filter(pl.col("deviation_magnitude").is_not_null())
        result = all_result.sort("deviation_magnitude", descending=True).head(top_n)
        print(f"[DEBUG] Returning top {top_n} from all: {result.height} rows")
    elif top_n is None and result.is_empty() and not return_all:
        # No anomalies found and user didn't specify top_n - return top 10 deviations
        # This ensures the tool always returns useful data instead of empty results
        print(f"[DEBUG] No anomalies + no top_n, returning top 10 deviations")
        all_result = pl.concat(result_dfs).sort(group_cols + [time_col])
        all_result = all_result.filter(pl.col("deviation_magnitude").is_not_null())
        result = all_result.sort("deviation_magnitude", descending=True).head(10)
        print(f"[DEBUG] Returning top 10: {result.height} rows")

    print(f"[DEBUG] Final result: {result.height} rows")
    if not result.is_empty():
        print(f"[DEBUG] Sample output:")
        print(result.select(["SALES_DATE", value_col, "rolling_mean", "rolling_std", "deviation_magnitude", "is_anomaly"]).head(3))

    return result


def _add_anomaly_columns(df: pl.DataFrame, value_col: str, threshold_std: float, debug: bool = False) -> pl.DataFrame:
    """Add rolling statistics and anomaly detection columns."""
    min_periods = 3  # Minimum periods for rolling window
    
    if debug:
        print(f"[DEBUG] _add_anomaly_columns: input rows={df.height}, value_col={value_col}")
        print(f"[DEBUG] {value_col} null count: {df[value_col].null_count()}")
        print(f"[DEBUG] {value_col} non-null count: {df.height - df[value_col].null_count()}")

    # Compute rolling mean and std
    df = df.with_columns([
        pl.col(value_col).rolling_mean(window_size=3, min_samples=min_periods).alias("rolling_mean"),
        pl.col(value_col).rolling_std(window_size=3, min_samples=min_periods).alias("rolling_std"),
    ])
    
    if debug:
        print(f"[DEBUG] rolling_mean null count: {df['rolling_mean'].null_count()}")
        print(f"[DEBUG] rolling_std null count: {df['rolling_std'].null_count()}")

    # Compute bounds
    df = df.with_columns([
        (pl.col("rolling_mean") - threshold_std * pl.col("rolling_std")).alias("lower_bound"),
        (pl.col("rolling_mean") + threshold_std * pl.col("rolling_std")).alias("upper_bound"),
    ])

    # Detect anomalies (two steps: first create anomaly_direction, then is_anomaly)
    df = df.with_columns([
        pl.when(pl.col(value_col) < pl.col("lower_bound"))
        .then(pl.lit("low"))
        .when(pl.col(value_col) > pl.col("upper_bound"))
        .then(pl.lit("high"))
        .otherwise(None)
        .alias("anomaly_direction"),
    ])

    df = df.with_columns([
        pl.col("anomaly_direction").is_not_null().alias("is_anomaly"),
    ])

    # Calculate deviation magnitude (z-score) for ranking anomalies
    # Only calculate when both value and rolling_std are non-null and rolling_std > 0
    df = df.with_columns([
        pl.when(
            (pl.col(value_col).is_not_null()) & 
            (pl.col("rolling_std").is_not_null()) & 
            (pl.col("rolling_std") > 0)
        )
        .then(((pl.col(value_col) - pl.col("rolling_mean")) / pl.col("rolling_std")).abs())
        .otherwise(None)
        .alias("deviation_magnitude"),
    ])
    
    if debug:
        print(f"[DEBUG] deviation_magnitude null count: {df['deviation_magnitude'].null_count()}")
        print(f"[DEBUG] deviation_magnitude non-null count: {df.height - df['deviation_magnitude'].null_count()}")
        non_null_df = df.filter(pl.col("deviation_magnitude").is_not_null())
        if not non_null_df.is_empty():
            print(f"[DEBUG] Sample non-null rows:")
            print(non_null_df.select(["SALES_DATE", value_col, "rolling_mean", "rolling_std", "deviation_magnitude"]).head(3))

    return df
