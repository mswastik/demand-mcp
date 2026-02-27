"""
briefing.py — Pre-computation layer for the standard demand planning report.

generate_standard_report(ds, window, filters) does two things:
  1. Computes all standard metrics at multiple hierarchy levels and returns a
     compact ~700-token briefing JSON for the LLM to write commentary from.
  2. Builds all standard slides in the supplied PresentationBuilder so the
     presentation is ~80% complete before the LLM writes a single word.

Hierarchy conventions
---------------------
Product drill path :  Franchise → Product Line → IBP Level 5 → IBP Level 6
                      → IBP Level 7 → CatalogNumber
Location drill path:  Forecast Level → Region / Area → Country

The root_cause_hints in the briefing are the worst 5 IBP Level 7 /
CatalogNumber rows so the LLM can name specific SKUs in commentary without
needing to call any additional tool.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import polars as pl

from data import (
    ACT_VOL,
    FORECAST_LEVEL_COL,
    ALL_LAGS,
    LAG_DF_COLS,
    LAG_STAT_COLS,
    apply_filters,
    filter_date_window,
    filter_to_actuals_period,
)
from metrics import (
    compute_metrics,
    compute_accuracy_trend,
    compute_yoy_growth,
)

logger = logging.getLogger(__name__)

# ── Hierarchy navigation ─────────────────────────────────────────────────────

# Ordered product hierarchy coarse → fine
PRODUCT_HIERARCHY = [
    "Franchise",
    "Product Line",
    "IBP Level 5",
    "IBP Level 6",
    "IBP Level 7",
    "CatalogNumber",
]

# Ordered location hierarchy coarse → fine
LOCATION_HIERARCHY = [
    FORECAST_LEVEL_COL,  # "Forecast Level"
    "Area",
    "Region",
    "Country",
]

def next_drill_level(hierarchy_col: str) -> str | None:
    """
    Given a hierarchy column, return the next (finer) level to drill into.
    Returns None if already at the finest level.
    """
    for hier in (PRODUCT_HIERARCHY, LOCATION_HIERARCHY):
        if hierarchy_col in hier:
            idx = hier.index(hierarchy_col)
            return hier[idx + 1] if idx + 1 < len(hier) else None
    return None


# ── Low-level helpers ────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Round floats for compact JSON; pass through None and strings."""
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v


def _metric_row(row: dict, lag: str = "L2") -> dict:
    """Extract the key scalar metrics from a metrics computation row dict."""
    return {
        "df_acc": _safe(row.get(f"{lag} DF Accuracy")),
        "stat_acc": _safe(row.get(f"{lag} Stat Accuracy")),
        "bias_pct": _safe(row.get(f"{lag} DF Bias %")),
        "fva": _safe(row.get(f"{lag} FVA")),
        "act_vol": _safe(row.get("Sum Act Vol")),
    }


def _df_to_compact(
    df: pl.DataFrame,
    group_col: str,
    lag: str = "L2",
    max_rows: int = 15,
) -> list[dict]:
    """
    Convert a metrics DataFrame to a compact list of dicts.
    Sorted by abs error descending (worst first) so the LLM sees the most
    important rows first and can stop reading early.
    """
    if df.is_empty():
        return []

    sort_col = f"{lag} DF Sum Abs Err"
    if sort_col in df.columns:
        df = df.sort(sort_col, descending=True)

    rows = []
    for i, row in enumerate(df.head(max_rows).iter_rows(named=True)):
        entry = {group_col: row.get(group_col)}
        entry.update(_metric_row(row, lag))
        entry["rank"] = i + 1
        rows.append(entry)
    return rows


# ── Standard report entry point ──────────────────────────────────────────────

def generate_standard_report(
    ds: Any,  # DataStore instance
    window: str = "last_3_months",
    filters: Optional[dict] = None,
) -> dict:
    """
    Main entry point called by the MCP tool.

    Returns
    -------
    briefing : dict
        Compact structured JSON (~600-900 tokens) containing all pre-digested
        numbers the LLM needs to write commentary.

    Side-effects
    ------------
    None — the caller (server.py tool) is responsible for passing the briefing
    to build_standard_slides() to create the presentation slides.
    """
    df_base = filter_to_actuals_period(ds.df)
    if filters:
        df_base = apply_filters(df_base, filters)

    # Date range in filtered data
    dates = df_base["SALES_DATE"].sort().unique().to_list()
    date_min = str(dates[0]) if dates else None
    date_max = str(dates[-1]) if dates else None

    briefing: dict = {
        "window": window,
        "filters_applied": filters or {},
        "date_range": {"min": date_min, "max": date_max},
    }

    try:
        briefing["by_forecast_level"] = _by_forecast_level(df_base, window)
    except Exception as e:
        logger.warning(f"by_forecast_level failed: {e}")
        briefing["by_forecast_level"] = []

    try:
        briefing["by_product_line"] = _by_level(df_base, window, "Product Line")
    except Exception as e:
        logger.warning(f"by_product_line failed: {e}")
        briefing["by_product_line"] = []

    try:
        briefing["by_ibp5"] = _by_level(df_base, window, "IBP Level 5")
    except Exception as e:
        logger.warning(f"by_ibp5 failed: {e}")
        briefing["by_ibp5"] = []

    try:
        briefing["root_cause_hints"] = _root_cause_hints(df_base, window)
    except Exception as e:
        logger.warning(f"root_cause_hints failed: {e}")
        briefing["root_cause_hints"] = []

    try:
        briefing["trend"] = _trend(df_base)
    except Exception as e:
        logger.warning(f"trend failed: {e}")
        briefing["trend"] = {"months": [], "df_acc": [], "stat_acc": []}

    try:
        briefing["yoy"] = _yoy(apply_filters(ds.df, filters)) if filters else _yoy(ds.df)
    except Exception as e:
        logger.warning(f"yoy failed: {e}")
        briefing["yoy"] = []

    try:
        briefing["evolution"] = _evolution(df_base, window)
    except Exception as e:
        logger.warning(f"evolution failed: {e}")
        briefing["evolution"] = {"volume": [], "accuracy": []}

    return briefing


# ── Layer computation helpers ────────────────────────────────────────────────

def _by_forecast_level(df_base: pl.DataFrame, window: str) -> list[dict]:
    """Layer 1 — Forecast Level / Region rollup.

    Pre-aggregates at (Forecast Level, CatalogNumber, SALES_DATE) grain first
    so abs errors are computed on correctly-summed actuals/forecasts.
    """
    df_w = filter_date_window(df_base, window) if window else df_base
    if df_w.is_empty():
        return []

    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS

    pre_group_cols = [FORECAST_LEVEL_COL, "CatalogNumber", "SALES_DATE"]
    pre_group_cols = [c for c in pre_group_cols if c in df_w.columns]
    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df_w.columns]
    df_pre = df_w.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    df_pre = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_pre, [FORECAST_LEVEL_COL])
    result = _compute_derived_metrics(agg)
    return _df_to_compact(result, FORECAST_LEVEL_COL, "L2", max_rows=20)


def _by_level(df_base: pl.DataFrame, window: str, col: str) -> list[dict]:
    """Generic: aggregate metrics at any single hierarchy column.

    Pre-aggregates at (Forecast Level, CatalogNumber, SALES_DATE) grain first
    so abs errors are computed on correctly-summed actuals/forecasts.
    """
    if col not in df_base.columns:
        return []

    df_w = filter_date_window(df_base, window) if window else df_base
    if df_w.is_empty():
        return []

    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS, FORECAST_LEVEL_COL

    pre_group_cols = [FORECAST_LEVEL_COL, "CatalogNumber", "SALES_DATE"]
    pre_group_cols = [c for c in pre_group_cols if c in df_w.columns]
    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df_w.columns]
    df_pre = df_w.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    df_pre = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_pre, [col])
    result = _compute_derived_metrics(agg)
    return _df_to_compact(result, col, "L2", max_rows=15)


def _root_cause_hints(df_base: pl.DataFrame, window: str) -> list[dict]:
    """
    Top 5 worst rows at the finest available grain (CatalogNumber, falling
    back to IBP Level 7, IBP Level 6), annotated with their parent Product Line.

    Pre-aggregates at (Forecast Level, CatalogNumber, SALES_DATE) — or the
    finest grain equivalent — before computing abs errors.
    """
    df_w = filter_date_window(df_base, window) if window else df_base
    if df_w.is_empty():
        return []

    # Find finest grain column available
    for col in ["CatalogNumber", "IBP Level 7", "IBP Level 6"]:
        if col in df_w.columns:
            finest = col
            break
    else:
        return []

    # Determine which parent context cols are available
    parent_cols = [c for c in ["Product Line", "IBP Level 5", "Franchise"] if c in df_w.columns]
    # group_cols for the final output  (no SALES_DATE — sum across all months in window)
    output_group_cols = parent_cols + [finest]
    # pre-aggregation grain also includes SALES_DATE and Forecast Level
    pre_group_cols = [c for c in [FORECAST_LEVEL_COL, "SALES_DATE"] + output_group_cols if c in df_w.columns]
    # deduplicate preserving order
    seen = set()
    pre_group_cols = [c for c in pre_group_cols if not (c in seen or seen.add(c))]

    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS

    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df_w.columns]
    df_pre = df_w.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    df_pre = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_pre, output_group_cols)
    result = _compute_derived_metrics(agg)

    sort_col = "L2 DF Sum Abs Err"
    if sort_col in result.columns:
        result = result.sort(sort_col, descending=True)

    hints = []
    for row in result.head(5).iter_rows(named=True):
        entry = {c: row.get(c) for c in output_group_cols}
        entry.update(_metric_row(row, "L2"))
        hints.append(entry)
    return hints


def _trend(df_base: pl.DataFrame) -> dict:
    """
    12-month monthly L2 DF & Stat accuracy as parallel arrays.
    Compact: returns {months, df_acc, stat_acc} not a list of dicts.

    Pre-aggregates at (Forecast Level, CatalogNumber, SALES_DATE) grain before
    computing abs errors so the accuracy formula matches the reference:
      Accuracy = 1 - Sum(Abs Err per SKU-month) / Sum(Act Vol per SKU-month)
    """
    df_w = filter_date_window(df_base, "last_12_months")
    if df_w.is_empty():
        return {"months": [], "df_acc": [], "stat_acc": []}

    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS, FORECAST_LEVEL_COL

    # Step 1: sum all volume columns at the (Forecast Level, CatalogNumber, SALES_DATE) grain
    pre_group_cols = [FORECAST_LEVEL_COL, "CatalogNumber", "SALES_DATE"]
    pre_group_cols = [c for c in pre_group_cols if c in df_w.columns]
    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df_w.columns]
    df_pre = df_w.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    # Step 2: compute abs errors then aggregate by SALES_DATE only
    df_pre = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_pre, ["SALES_DATE"])
    result = _compute_derived_metrics(agg).sort("SALES_DATE")

    months = [str(d)[:7] for d in result["SALES_DATE"].to_list()]
    df_acc = [_safe(v) for v in result["L2 DF Accuracy"].to_list()]
    stat_acc = [_safe(v) for v in result["L2 Stat Accuracy"].to_list()]
    return {"months": months, "df_acc": df_acc, "stat_acc": stat_acc}


def _yoy(df_base: pl.DataFrame) -> list[dict]:
    """
    Year-over-year volume growth using act_with_forecast columns.
    
    Uses compute_yoy_growth which creates:
      - act_with_forecast_df: actuals for completed months + DF forecast for remaining
      - act_with_forecast_stat: actuals for completed months + Stat forecast for remaining
    
    Returns detailed data for table including volumes and YoY growth percentages
    for both DF and Stat forecast scenarios.
    Returns all available years (not limited to last 3).
    """
    from metrics import compute_yoy_growth

    if df_base.is_empty():
        return []

    # Use the compute_yoy_growth function which handles act_with_forecast logic
    result = compute_yoy_growth(df_base, group_by_cols=[])

    if result.is_empty():
        return []

    # Sort by year - return all available years
    result = result.sort("Year")

    rows = result.iter_rows(named=True)
    output = []

    for row in rows:
        entry = {
            "year": row["Year"],
            # Volumes
            "act_vol": _safe(row.get("Sum Act Vol")),
            "act_with_forecast_df_vol": _safe(row.get("Sum act_with_forecast_df Vol")),
            "act_with_forecast_stat_vol": _safe(row.get("Sum act_with_forecast_stat Vol")),
            "df_vol": _safe(row.get("Sum DF Fcst Vol")),
            "stat_vol": _safe(row.get("Sum Stat Fcst Vol")),
            # YoY growth percentages - these are the key metrics for the presentation
            "df_yoy_pct": _safe(row.get("DF Fcst YoY %")),
            "stat_yoy_pct": _safe(row.get("Stat Fcst YoY %")),
            "act_yoy_pct": _safe(row.get("Act YoY %")),
        }
        output.append(entry)

    # Mark current year
    from datetime import date
    current_year = date.today().year
    for entry in output:
        entry["is_current"] = entry["year"] == current_year

    return output


def _evolution(df_base: pl.DataFrame, window: str) -> dict:
    """
    Forecast evolution across lags (L2 → L1 → L0 → Fcst) vs Actuals.
    Returns both volume evolution and accuracy evolution.
    """
    from data import ALL_LAGS
    from metrics import compute_forecast_evolution, compute_forecast_evolution_accuracy
    
    if df_base.is_empty():
        return {"volume": [], "accuracy": []}
    
    # Volume evolution
    vol_evo = compute_forecast_evolution(df_raw=df_base, filters=None, window=window)
    volume_data = []
    if not vol_evo.is_empty():
        for row in vol_evo.sort("SALES_DATE").iter_rows(named=True):
            volume_data.append({
                "month": str(row["SALES_DATE"])[:7],
                "actual": _safe(row["Sum Act Vol"]),
                "l2_df": _safe(row.get("L2 DF Sum Fcst")),
                "l1_df": _safe(row.get("L1 DF Sum Fcst")),
                "l0_df": _safe(row.get("L0 DF Sum Fcst")),
                "fcst_df": _safe(row.get("Fcst DF Sum Fcst")),
            })
    
    # Accuracy evolution
    acc_evo = compute_forecast_evolution_accuracy(df_raw=df_base, filters=None, window=window)
    accuracy_data = []
    if not acc_evo.is_empty():
        for row in acc_evo.sort("SALES_DATE").iter_rows(named=True):
            accuracy_data.append({
                "month": str(row["SALES_DATE"])[:7],
                "actual_vol": _safe(row["Sum Act Vol"]),
                "l2_df_acc": _safe(row.get("L2 DF Acc")),
                "l1_df_acc": _safe(row.get("L1 DF Acc")),
                "l0_df_acc": _safe(row.get("L0 DF Acc")),
            })
    
    return {"volume": volume_data, "accuracy": accuracy_data}


# ── drill_down_slide computation ─────────────────────────────────────────────

def compute_drill_down(
    ds: Any,  # DataStore instance
    hierarchy_col: str,
    hierarchy_value: str,
    metric: str,
    window: str,
    filters: Optional[dict] = None,
) -> dict:
    """
    Compute a focused drill-down for one hierarchy node and return a compact
    briefing summary for the LLM plus the data needed to build the slide.

    Returns
    -------
    {
        "drill_level": str,          # column drilled into (next level down)
        "parent_col": str,
        "parent_value": str,
        "metric": str,
        "rows": [...],               # compact list of dicts
        "slide_data": {...}          # chart/table spec dicts for PresentationBuilder
    }
    """
    # Resolve next level down
    drill_level = next_drill_level(hierarchy_col)
    if drill_level is None:
        raise ValueError(
            f"'{hierarchy_col}' is already the finest hierarchy level. "
            "No further drill-down possible."
        )

    # Build combined filter
    combined_filters = dict(filters or {})
    combined_filters[hierarchy_col] = hierarchy_value

    df_base = filter_to_actuals_period(ds.df)
    df_base = apply_filters(df_base, combined_filters)
    df_base = filter_date_window(df_base, window)

    if df_base.is_empty():
        return {
            "error": f"No data found for {hierarchy_col}='{hierarchy_value}' "
                     f"in window '{window}'",
            "drill_level": drill_level,
            "parent_col": hierarchy_col,
            "parent_value": hierarchy_value,
            "rows": [],
        }

    if drill_level not in df_base.columns:
        raise ValueError(
            f"Column '{drill_level}' not found in data. "
            f"Available columns: {df_base.columns}"
        )

    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS, FORECAST_LEVEL_COL as _FL

    # Pre-aggregate at (Forecast Level, CatalogNumber, SALES_DATE) grain so
    # abs errors are computed on correctly-summed actuals/forecasts.
    pre_group_cols = [_FL, "CatalogNumber", "SALES_DATE", drill_level]
    pre_group_cols = [c for c in pre_group_cols if c in df_base.columns]
    # deduplicate preserving order
    seen: set = set()
    pre_group_cols = [c for c in pre_group_cols if not (c in seen or seen.add(c))]
    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df_base.columns]
    df_pre = df_base.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    df_err = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_err, [drill_level])
    result = _compute_derived_metrics(agg)

    rows = _df_to_compact(result, drill_level, "L2", max_rows=15)

    # Build slide_data depending on metric type
    slide_data = _drill_slide_data(rows, drill_level, metric, result)

    return {
        "drill_level": drill_level,
        "parent_col": hierarchy_col,
        "parent_value": hierarchy_value,
        "metric": metric,
        "window": window,
        "filters_applied": combined_filters,
        "rows": rows,
        "slide_data": slide_data,
    }


def _drill_slide_data(rows: list[dict], drill_level: str, metric: str, df: pl.DataFrame) -> dict:
    """
    Build chart and table spec dicts from drill-down rows.
    Returns {chart_spec, table_spec, layout}.
    """
    labels = [r[drill_level] or "N/A" for r in rows]

    if metric == "accuracy":
        df_vals = [round(r["df_acc"] * 100, 1) if r["df_acc"] is not None else None for r in rows]
        stat_vals = [round(r["stat_acc"] * 100, 1) if r["stat_acc"] is not None else None for r in rows]
        chart_spec = {
            "chart_type": "grouped_bar",
            "title": f"DF vs Stat Accuracy (L2) by {drill_level}",
            "x_data": labels,
            "y_data": {"DF Accuracy %": df_vals, "Stat Accuracy %": stat_vals},
            "x_label": drill_level,
            "y_label": "Accuracy %",
            "show_legend": True,
            "height": 320,
        }
        table_spec = {
            "headers": [drill_level, "DF Acc %", "Stat Acc %", "FVA %", "Bias %", "Act Vol"],
            "rows": [
                [
                    r[drill_level],
                    f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                    f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                    f"{r['fva']*100:.1f}%" if r["fva"] is not None else "N/A",
                    f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                    f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
                ]
                for r in rows
            ],
            "highlight_col": 1,
            "highlight_thresholds": [60, 80],
        }
        return {"chart_spec": chart_spec, "table_spec": table_spec, "layout": "chart_table"}

    elif metric == "bias":
        bias_vals = [round(r["bias_pct"] * 100, 1) if r["bias_pct"] is not None else None for r in rows]
        chart_spec = {
            "chart_type": "bar",
            "title": f"DF Bias % (L2) by {drill_level}",
            "x_data": labels,
            "y_data": bias_vals,
            "x_label": drill_level,
            "y_label": "Bias %",
            "show_legend": False,
            "height": 320,
        }
        table_spec = {
            "headers": [drill_level, "DF Acc %", "Bias %", "Act Vol"],
            "rows": [
                [
                    r[drill_level],
                    f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                    f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                    f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
                ]
                for r in rows
            ],
            "highlight_col": 2,
            "highlight_thresholds": [-10, 10],
        }
        return {"chart_spec": chart_spec, "table_spec": table_spec, "layout": "chart_table"}

    elif metric == "fva":
        fva_vals = [round(r["fva"] * 100, 1) if r["fva"] is not None else None for r in rows]
        chart_spec = {
            "chart_type": "bar",
            "title": f"Forecast Value Add (L2) by {drill_level}",
            "x_data": labels,
            "y_data": fva_vals,
            "x_label": drill_level,
            "y_label": "FVA % pts",
            "show_legend": False,
            "height": 320,
        }
        table_spec = {
            "headers": [drill_level, "DF Acc %", "Stat Acc %", "FVA %", "Act Vol"],
            "rows": [
                [
                    r[drill_level],
                    f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                    f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                    f"{r['fva']*100:.1f}%" if r["fva"] is not None else "N/A",
                    f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
                ]
                for r in rows
            ],
            "highlight_col": 3,
            "highlight_thresholds": [-5, 5],
        }
        return {"chart_spec": chart_spec, "table_spec": table_spec, "layout": "chart_table"}

    elif metric == "trend":
        # Trend for this parent node across time
        from data import FORECAST_LEVEL_COL
        trend_data = _trend_for_drill(df)
        chart_spec = {
            "chart_type": "line",
            "title": f"Monthly L2 Accuracy Trend",
            "x_data": trend_data["months"],
            "y_data": {"DF Accuracy %": trend_data["df_acc_pct"], "Stat Accuracy %": trend_data["stat_acc_pct"]},
            "x_label": "Month",
            "y_label": "Accuracy %",
            "show_legend": True,
            "height": 340,
        }
        return {"chart_spec": chart_spec, "table_spec": None, "layout": "chart"}

    else:  # top_offenders or default
        # Table of worst performers at drill_level sorted by abs error
        table_spec = {
            "headers": [drill_level, "DF Acc %", "Stat Acc %", "Bias %", "Act Vol", "Rank"],
            "rows": [
                [
                    r[drill_level],
                    f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                    f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                    f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                    f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
                    r["rank"],
                ]
                for r in rows
            ],
            "highlight_col": 1,
            "highlight_thresholds": [60, 80],
        }
        return {"chart_spec": None, "table_spec": table_spec, "layout": "table"}


def _trend_for_drill(df: pl.DataFrame) -> dict:
    """Helper: monthly accuracy trend for an already-filtered DataFrame.

    Pre-aggregates at (Forecast Level, CatalogNumber, SALES_DATE) grain before
    computing abs errors so the accuracy formula matches the reference.
    """
    from metrics import _abs_error_cols, _aggregate_metrics, _compute_derived_metrics
    from data import ACT_VOL, LAG_DF_COLS, LAG_STAT_COLS, ALL_LAGS, FORECAST_LEVEL_COL
    if df.is_empty():
        return {"months": [], "df_acc_pct": [], "stat_acc_pct": []}

    # Pre-aggregate at the correct grain before abs error computation
    pre_group_cols = [FORECAST_LEVEL_COL, "CatalogNumber", "SALES_DATE"]
    pre_group_cols = [c for c in pre_group_cols if c in df.columns]
    vol_cols = [ACT_VOL] + [v for lag in ALL_LAGS for _, v in [LAG_DF_COLS[lag], LAG_STAT_COLS[lag]]]
    vol_cols = [c for c in vol_cols if c in df.columns]
    df_pre = df.group_by(pre_group_cols).agg([pl.col(c).sum() for c in vol_cols])

    df_err = _abs_error_cols(df_pre)
    agg = _aggregate_metrics(df_err, ["SALES_DATE"])
    result = _compute_derived_metrics(agg).sort("SALES_DATE")
    months = [str(d)[:7] for d in result["SALES_DATE"].to_list()]
    df_acc = [round(v * 100, 1) if v is not None else None for v in result["L2 DF Accuracy"].to_list()]
    stat_acc = [round(v * 100, 1) if v is not None else None for v in result["L2 Stat Accuracy"].to_list()]
    return {"months": months, "df_acc_pct": df_acc, "stat_acc_pct": stat_acc}


# ── Slide-building from briefing ─────────────────────────────────────────────

def build_standard_slides(briefing: dict, builder: Any) -> None:
    """
    Create all standard slides in the PresentationBuilder from briefing data.
    Called by generate_standard_report MCP tool after computing briefing.

    Slides added (in order):
      1. Cover
      2. KPI Summary (metric cards)
      3. Accuracy Trend (line chart)
      4. Forecast Level Accuracy & Bias (chart_table)
      5. Top Product Lines (table)
      6. IBP Level 5 (table)
      7. Bias by Forecast Level (chart)
      8. FVA by Forecast Level (chart_table)
      9. YoY Volume Growth (chart)
    """
    from presentation import Slide, ChartSpec, TableSpec

    filters = briefing.get("filters_applied", {})
    window = briefing.get("window", "")
    scope = ", ".join(v for k, v in filters.items()) if filters else "All"
    dr = briefing.get("date_range", {})
    period_label = f"""
                    <div style="padding-top:60px; padding-bottom:160px; font-family:Arial Black;>{scope}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top:auto;">{dr.get('min','')} to {dr.get('max','')} | Window: {window}</div>
                    """

    # ── Slide 1: Cover ────────────────────────────────────────────────────────
    builder.add_slide(Slide(
        slide_id="cover",
        layout="title",
        title=builder.title,
        commentary=period_label,
    ))

    # ── Slide 2: KPI Summary ─────────────────────────────────────────────────
    fl_rows = briefing.get("by_forecast_level", [])
    if fl_rows:
        # Weighted average across all Forecast Levels
        total_vol = sum(r["act_vol"] or 0 for r in fl_rows)
        wa_df_acc = (
            sum((r["df_acc"] or 0) * (r["act_vol"] or 0) for r in fl_rows) / total_vol
            if total_vol else None
        )
        wa_stat_acc = (
            sum((r["stat_acc"] or 0) * (r["act_vol"] or 0) for r in fl_rows) / total_vol
            if total_vol else None
        )
        wa_bias = (
            sum((r["bias_pct"] or 0) * (r["act_vol"] or 0) for r in fl_rows) / total_vol
            if total_vol else None
        )
        wa_fva = (
            sum((r["fva"] or 0) * (r["act_vol"] or 0) for r in fl_rows) / total_vol
            if total_vol else None
        )

        cards = [
            {
                "label": "DF Accuracy (L2)",
                "value": f"{wa_df_acc*100:.1f}%" if wa_df_acc is not None else "N/A",
                "delta": f"Stat: {wa_stat_acc*100:.1f}%" if wa_stat_acc is not None else "",
                "direction": "up" if (wa_fva or 0) >= 0 else "down",
            },
            {
                "label": "Forecast Value Add",
                "value": f"{wa_fva*100:+.1f}%" if wa_fva is not None else "N/A",
                "delta": "DF vs Stat accuracy gap",
                "direction": "up" if (wa_fva or 0) >= 0 else "down",
            },
            {
                "label": "DF Bias % (L2)",
                "value": f"{wa_bias*100:+.1f}%" if wa_bias is not None else "N/A",
                "delta": ">0 under-fcst, <0 over-fcst",
                "direction": "neutral",
            },
            {
                "label": "Total Actual Volume",
                "value": f"{total_vol:,.0f}",
                "delta": window,
                "direction": "neutral",
            },
        ]
        builder.add_slide(Slide(
            slide_id="kpi_summary",
            layout="metrics",
            title="Executive Performance Summary",
            cards=cards,  # Store cards separately so commentary can be added later
        ))

    # ── Slide 3: Accuracy Trend ───────────────────────────────────────────────
    trend = briefing.get("trend", {})
    if trend.get("months"):
        df_acc_pct = [round(v * 100, 1) if v is not None else None for v in trend["df_acc"]]
        stat_acc_pct = [round(v * 100, 1) if v is not None else None for v in trend["stat_acc"]]
        
        # Add target line at 70% for all months
        target_line = [70.0] * len(trend["months"])
        
        builder.add_slide(Slide(
            slide_id="accuracy_trend",
            layout="chart",
            title="12-Month L2 Accuracy Trend",
            subtitle="Track accuracy performance against 70% target",
            chart=ChartSpec(
                chart_type="line",
                title="DF vs Stat Accuracy (L2) — Monthly Performance vs Target",
                x_data=trend["months"],
                y_data={
                    "DF Accuracy %": df_acc_pct,
                    "Stat Accuracy %": stat_acc_pct,
                    "Target (70%)": target_line,
                },
                x_label="Month",
                y_label="Accuracy % (Target: 70%)",
                #colors=["#1f77b4", "#ff7f0e", "#2ca02c"],  # Blue, Orange, Green
                show_legend=True,
                height=380,
            ),
        ))

    # ── Slide 4: Forecast Level Performance (Combined View) ──────────────────
    if fl_rows:
        # Build comprehensive table with all metrics
        headers = ["Forecast Level", "DF Acc %", "Stat Acc %", "FVA %", "Bias %", "Act Vol"]
        table_rows = [
            [
                r["Forecast Level"],
                f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                f"{r['fva']*100:.1f}%" if r["fva"] is not None else "N/A",
                f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
            ]
            for r in fl_rows
        ]
        
        # Build charts for two-col layout
        fl_labels = [r["Forecast Level"] for r in fl_rows]
        fl_df_acc = [round(r["df_acc"] * 100, 1) if r["df_acc"] is not None else None for r in fl_rows]
        fl_stat_acc = [round(r["stat_acc"] * 100, 1) if r["stat_acc"] is not None else None for r in fl_rows]
        fl_bias = [round(r["bias_pct"] * 100, 1) if r["bias_pct"] is not None else None for r in fl_rows]
        fl_fva = [round(r["fva"] * 100, 1) if r["fva"] is not None else None for r in fl_rows]
        
        builder.add_slide(Slide(
            slide_id="by_forecast_level",
            layout="two_col_chart_table",
            title="Forecast Level Performance - Multi-Metric View",
            subtitle="Left: Accuracy comparison | Right: Bias & FVA",
            chart=ChartSpec(
                chart_type="grouped_bar",
                title="DF vs Stat Accuracy % by Region",
                x_data=fl_labels,
                y_data={"DF Accuracy %": fl_df_acc, "Stat Accuracy %": fl_stat_acc},
                x_label="Forecast Level",
                y_label="Accuracy %",
                show_legend=True,
                height=240,
            ),
            chart2=ChartSpec(
                chart_type="grouped_bar",
                title="Bias % and FVA % by Region",
                x_data=fl_labels,
                y_data={"Bias %": fl_bias, "FVA %": fl_fva},
                x_label="Forecast Level",
                y_label="Percentage",
                show_legend=True,
                height=240,
            ),
            table=TableSpec(
                headers=headers,
                rows=table_rows,
                highlight_col=1,
                highlight_thresholds=(60, 80),
            ),
        ))

    # ── Slide 5: Top Product Lines ────────────────────────────────────────────
    pl_rows = briefing.get("by_product_line", [])
    if pl_rows:
        headers = ["Product Line", "DF Acc %", "Stat Acc %", "FVA %", "Bias %", "Act Vol"]
        table_rows = [
            [
                r["Product Line"],
                f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                f"{r['fva']*100:.1f}%" if r["fva"] is not None else "N/A",
                f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
            ]
            for r in pl_rows
        ]
        
        # Build chart showing worst performers (accuracy < 70%)
        pl_labels = [r["Product Line"][:20] + "..." if len(r["Product Line"]) > 20 else r["Product Line"] 
                     for r in pl_rows[:10]]  # Top 10 worst
        pl_df_acc = [round(r["df_acc"] * 100, 1) if r["df_acc"] is not None else None for r in pl_rows[:10]]
        pl_stat_acc = [round(r["stat_acc"] * 100, 1) if r["stat_acc"] is not None else None for r in pl_rows[:10]]
        
        builder.add_slide(Slide(
            slide_id="by_product_line",
            layout="chart_table",
            title="Product Line Performance - Worst First",
            subtitle="Focus on products with accuracy < 60% (critical) or 60-70% (concern)",
            chart=ChartSpec(
                chart_type="bar",
                title="Top 10 Worst Product Lines - DF vs Stat Accuracy",
                x_data=pl_labels,
                y_data={"DF Accuracy %": pl_df_acc, "Stat Accuracy %": pl_stat_acc},
                x_label="Product Line",
                y_label="Accuracy %",
                show_legend=True,
                height=280,
            ),
            table=TableSpec(
                headers=headers,
                rows=table_rows,
                highlight_col=1,
                highlight_thresholds=(60, 80),
            ),
        ))

    # ── Slide 6: IBP Level 5 ─────────────────────────────────────────────────
    ibp5_rows = briefing.get("by_ibp5", [])
    if ibp5_rows:
        headers = ["IBP Level 5", "DF Acc %", "Stat Acc %", "FVA %", "Bias %", "Act Vol"]
        table_rows = [
            [
                r["IBP Level 5"],
                f"{r['df_acc']*100:.1f}%" if r["df_acc"] is not None else "N/A",
                f"{r['stat_acc']*100:.1f}%" if r["stat_acc"] is not None else "N/A",
                f"{r['fva']*100:.1f}%" if r["fva"] is not None else "N/A",
                f"{r['bias_pct']*100:.1f}%" if r["bias_pct"] is not None else "N/A",
                f"{r['act_vol']:,.0f}" if r["act_vol"] is not None else "N/A",
            ]
            for r in ibp5_rows
        ]
        
        # Build chart for worst performers
        ibp5_labels = [r["IBP Level 5"][:25] + "..." if len(r["IBP Level 5"]) > 25 else r["IBP Level 5"]
                       for r in ibp5_rows[:10]]
        ibp5_df_acc = [round(r["df_acc"] * 100, 1) if r["df_acc"] is not None else None for r in ibp5_rows[:10]]
        ibp5_fva = [round(r["fva"] * 100, 1) if r["fva"] is not None else None for r in ibp5_rows[:10]]
        
        builder.add_slide(Slide(
            slide_id="by_ibp5",
            layout="chart_table",
            title="IBP Level 5 Performance - Worst First",
            subtitle="Granular product view for root cause analysis",
            chart=ChartSpec(
                chart_type="bar",
                title="Top 10 Worst IBP Level 5 - DF Accuracy & FVA",
                x_data=ibp5_labels,
                y_data={"DF Accuracy %": ibp5_df_acc, "FVA %": ibp5_fva},
                x_label="IBP Level 5",
                y_label="Percentage",
                show_legend=True,
                height=280,
            ),
            table=TableSpec(
                headers=headers,
                rows=table_rows,
                highlight_col=1,
                highlight_thresholds=(60, 80),
            ),
        ))

    # ── Slide 7: Forecast Evolution - Volume ─────────────────────────────────
    evolution = briefing.get("evolution", {})
    vol_data = evolution.get("volume", [])
    if len(vol_data) >= 3:
        months = [d["month"] for d in vol_data[-6:]]  # Last 6 months
        builder.add_slide(Slide(
            slide_id="forecast_evolution_volume",
            layout="chart",
            title="Forecast Evolution - Volume Across Lags",
            subtitle="How forecast volume changed from L2 to Final forecast",
            chart=ChartSpec(
                chart_type="line",
                title="Volume Evolution: L2 → L1 → L0 → Final vs Actual",
                x_data=months,
                y_data={
                    "Actual": [d["actual"] for d in vol_data[-6:]],
                    "L2 Forecast": [d["l2_df"] for d in vol_data[-6:]],
                    "L1 Forecast": [d["l1_df"] for d in vol_data[-6:]],
                    "L0 Forecast": [d["l0_df"] for d in vol_data[-6:]],
                    "Final Fcst": [d["fcst_df"] for d in vol_data[-6:]],
                },
                x_label="Month",
                y_label="Volume (Units)",
                show_legend=True,
                height=380,
            ),
        ))

    # ── Slide 8: Forecast Evolution - Accuracy ───────────────────────────────
    acc_data = evolution.get("accuracy", [])
    if len(acc_data) >= 3:
        months = [d["month"] for d in acc_data[-6:]]
        builder.add_slide(Slide(
            slide_id="forecast_evolution_accuracy",
            layout="chart",
            title="Forecast Evolution - Accuracy Development",
            subtitle="How accuracy improved as we got closer to the period",
            chart=ChartSpec(
                chart_type="line",
                title="Accuracy Evolution: L2 → L1 → L0 (Last 6 Months)",
                x_data=months,
                y_data={
                    "L2 Accuracy": [d["l2_df_acc"] * 100 if d["l2_df_acc"] else None for d in acc_data[-6:]],
                    "L1 Accuracy": [d["l1_df_acc"] * 100 if d["l1_df_acc"] else None for d in acc_data[-6:]],
                    "L0 Accuracy": [d["l0_df_acc"] * 100 if d["l0_df_acc"] else None for d in acc_data[-6:]],
                },
                x_label="Month",
                y_label="Accuracy %",
                show_legend=True,
                height=380,
            ),
        ))

    # ── Slide 9: YoY Volume Growth ────────────────────────────────────────────
    yoy_rows = briefing.get("yoy", [])
    if len(yoy_rows) >= 2:
        #years_e = [str(r["year"]) + (" (E)" if r.get("is_current") else "") for r in yoy_rows]
        years = [str(r["year"]) for r in yoy_rows]
        # Use act_with_forecast_df_vol for the chart (shows actuals + DF forecast for current year)
        vols_df = [r["act_with_forecast_df_vol"] for r in yoy_rows]
        vols_stat = [r["act_with_forecast_stat_vol"] for r in yoy_rows]
        # YoY growth percentages
        df_yoy = [r["df_yoy_pct"] for r in yoy_rows]
        stat_yoy = [r["stat_yoy_pct"] for r in yoy_rows]

        # Build detailed table with volumes and growth rates for both DF and Stat scenarios
        table_headers = [
            "Year", 
            "Actual Volume", 
            "Act+DF Fcst Vol", 
            "Act+Stat Fcst Vol",
            "DF YoY %",
            "Stat YoY %"
        ]
        table_rows = []
        for r in yoy_rows:
            year_label = str(r["year"]) + (" (E)" if r.get("is_current") else "")
            df_yoy_str = f"{r['df_yoy_pct']:+.1f}%" if r.get("df_yoy_pct") is not None else "N/A"
            stat_yoy_str = f"{r['stat_yoy_pct']:+.1f}%" if r.get("stat_yoy_pct") is not None else "N/A"
            table_rows.append([
                year_label,
                f"{r['act_vol']:,.0f}" if r["act_vol"] else "N/A",
                f"{r['act_with_forecast_df_vol']:,.0f}" if r.get("act_with_forecast_df_vol") else "N/A",
                f"{r['act_with_forecast_stat_vol']:,.0f}" if r.get("act_with_forecast_stat_vol") else "N/A",
                df_yoy_str,
                stat_yoy_str,
            ])

        builder.add_slide(Slide(
            slide_id="yoy_growth",
            layout="chart_table",
            title="Year-over-Year Volume Growth",
            subtitle="Current year includes actuals (completed months) + forecast (remaining months)",
            chart=ChartSpec(
                chart_type="combo_bar_line",
                title="Annual Volume & YoY Growth (DF vs Stat)",
                x_data=years,
                y_data={
                    "Act+DF Fcst Vol": vols_df,
                    "Act+Stat Fcst Vol": vols_stat,
                    "DF YoY %": df_yoy,
                    "Stat YoY %": stat_yoy,
                },
                x_label="Year",
                y_label="Volume (Units)",
                show_legend=True,
                height=280,
            ),
            table=TableSpec(
                headers=table_headers,
                rows=table_rows,
                highlight_col=4,  # Highlight DF YoY % column
                highlight_thresholds=(-5, 5),  # Red if < -5%, Green if > +5%
            ),
        ))
