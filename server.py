"""
server.py — Demand Planning MCP Server

Start with:
    python server.py                        # uses config.yaml in current dir
    python server.py --config /path/to/config.yaml

Tools exposed:
    Data exploration:
        1.  get_data_info
        2.  get_hierarchy_members
        3.  get_date_range

    Metrics:
        4.  compute_accuracy_summary
        5.  compute_bias_summary
        6.  compute_fva_summary
        7.  get_accuracy_trend
        8.  get_yoy_growth
        9.  get_forecast_evolution
        10. get_forecast_evolution_accuracy — accuracy % across lags (L2→L1→L0)
        11. get_top_offenders
        12. detect_anomalies_in_trend — identify statistical anomalies in metric trends

    Presentation:
        13. generate_standard_report  — auto-initialises a new presentation, builds
                                        standard slides, returns compact briefing JSON.
                                        HTML is saved automatically after each mutation.
        14. add_slide                  — append a custom slide; auto-saves HTML.
        15. add_commentary            — write narrative for a named slide; auto-saves HTML.
        16. get_presentation_status   — list slide IDs + commentary status (resume support).
        17. drill_down_slide          — append a drill-down slide; auto-saves HTML.

    Editing existing presentations:
        18. list_presentations        — list saved HTML files; flags which are editable.
        19. load_presentation         — reload a saved presentation for editing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl
import yaml
from mcp.server.fastmcp import FastMCP

# ── Bootstrap ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demand_mcp")


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    logger.info(f"Reading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in config relative to the config file's directory
    base = config_path.parent
    for key in ("parquet_path", "brand_config", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((base / cfg[key]).resolve())

    return cfg


def _load_brand(brand_path: str) -> dict:
    with open(brand_path) as f:
        return json.load(f)


def _resolve_config_path() -> str:
    """
    Config path resolution order:
      1. DEMAND_MCP_CONFIG environment variable
      2. config.yaml next to server.py
      3. config.yaml in current working directory
    """
    if "DEMAND_MCP_CONFIG" in os.environ:
        return os.environ["DEMAND_MCP_CONFIG"]

    server_dir_config = Path(__file__).parent / "config.yaml"
    if server_dir_config.exists():
        return str(server_dir_config)

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return str(cwd_config)

    raise FileNotFoundError(
        "Could not find config.yaml. Set the DEMAND_MCP_CONFIG environment variable "
        "to the full path of your config file, e.g.:\n"
        "  export DEMAND_MCP_CONFIG=/path/to/demand_mcp/config.yaml"
    )


# ── Server singleton state ───────────────────────────────────────────────────

class ServerState:
    config: dict
    brand: dict
    presentation: Any  # PresentationBuilder | None

    def __init__(self, config: dict, brand: dict):
        self.config = config
        self.brand = brand
        self.presentation = None


_state: ServerState | None = None


def _bootstrap(config_path: str | None = None) -> None:
    """
    Load config, brand, and parquet data into memory.
    Safe to call multiple times — skips if already initialized.
    """
    global _state
    if _state is not None:
        return  # already initialized

    path = config_path or _resolve_config_path()
    config = _load_config(path)
    brand = _load_brand(config["brand_config"])

    from data import DataStore
    DataStore.load(config["parquet_path"])

    _state = ServerState(config=config, brand=brand)
    logger.info("Server state initialized successfully.")


# ── FastMCP with lifespan hook ───────────────────────────────────────────────

# Bootstrap is lazy — triggered on first tool call.
# Works correctly with both `python server.py` and `fastmcp run server.py`.
mcp = FastMCP("demand-mcp")


def _get_state() -> ServerState:
    """Return ServerState, bootstrapping on first tool call."""
    if _state is None:
        _bootstrap()
    if _state is None:
        raise RuntimeError(
            "Server not initialized. Ensure DEMAND_MCP_CONFIG points to your config.yaml "
            "or that config.yaml exists next to server.py."
        )
    return _state


def _get_ds():
    """Shortcut to DataStore singleton — also triggers bootstrap if needed."""
    _get_state()  # ensures DataStore.load() has been called
    from data import DataStore
    return DataStore.get()


# ── Helper: DataFrame → JSON-serialisable dict ───────────────────────────────

def _df_to_dict(df: pl.DataFrame, max_rows: int = 50) -> dict:
    """Convert polars DataFrame to a JSON-safe dict for MCP responses."""
    if df.is_empty():
        return {"columns": [], "rows": [], "row_count": 0}

    truncated = df.head(max_rows)
    return {
        "columns": truncated.columns,
        "rows": truncated.to_dicts(),
        "row_count": df.shape[0],
        "truncated": df.shape[0] > max_rows,
    }


def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return str(v)


def _fmt_num(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 1 — get_data_info
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_data_info() -> dict:
    """
    Return metadata about the loaded dataset: row count, date range,
    available columns, and unique counts at each hierarchy level.
    Call this first to orient yourself before any analysis.
    """
    ds = _get_ds()
    from data import LOC_HIERARCHY, PROD_HIERARCHY

    date_min, date_max = ds.date_range()

    hierarchy_uniques = {}
    for col in LOC_HIERARCHY + PROD_HIERARCHY:
        try:
            hierarchy_uniques[col] = ds.df[col].drop_nulls().n_unique()
        except Exception:
            hierarchy_uniques[col] = None

    return {
        "row_count": ds.df.shape[0],
        "date_range": {
            "min": str(date_min),
            "max": str(date_max),
        },
        "columns": ds.df.columns,
        "location_hierarchy": LOC_HIERARCHY,
        "product_hierarchy": PROD_HIERARCHY,
        "hierarchy_unique_counts": hierarchy_uniques,
        "loaded_at": str(ds.loaded_at),
        "parquet_path": str(ds.parquet_path),
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 2 — get_hierarchy_members
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_hierarchy_members(
    level: str,
    filters: Optional[dict] = None,
) -> dict:
    """
    Return unique values at a given hierarchy level, optionally filtered.

    Parameters
    ----------
    level : hierarchy column name, e.g. 'Franchise', 'Region', 'Business Unit'
    filters : optional dict of {column: value} to narrow the search,
              e.g. {"Business Sector": "MedSurg", "Region": "EMEA"}

    Returns
    -------
    List of unique values sorted alphabetically.
    """
    ds = _get_ds()
    members = ds.hierarchy_members(level, filters)
    return {"level": level, "members": members, "count": len(members)}


# ════════════════════════════════════════════════════════════════════════════
# TOOL 3 — get_date_range
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_date_range(filters: Optional[dict] = None) -> dict:
    """
    Return the min and max SALES_DATE in the dataset, optionally filtered.

    Parameters
    ----------
    filters : optional dict of {column: value} filters

    Returns
    -------
    min_date, max_date as ISO strings, plus list of all available months.
    """
    ds = _get_ds()
    from data import apply_filters

    df = ds.df
    if filters:
        df = apply_filters(df, filters)

    dates = df["SALES_DATE"].cast(pl.Date).sort().unique().to_list()
    return {
        "min_date": str(dates[0]) if dates else None,
        "max_date": str(dates[-1]) if dates else None,
        "available_months": [str(d) for d in dates],
        "month_count": len(dates),
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 4 — compute_accuracy_summary
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_accuracy_summary(
    group_by: list[str],
    filters: Optional[dict] = None,
    window: Optional[str] = None,
) -> dict:
    """
    Compute forecast accuracy (all four lags: L2, L1, L0, Fcst) for DF and Stat.

    Formula: Accuracy = 1 - Sum(Abs Error) / Sum(Act Orders Vol)

    The Forecast Level logic (East Asia Area → use Area, else Region) is
    applied automatically when grouping.

    Parameters
    ----------
    group_by : list of hierarchy columns to group by,
               e.g. ["Forecast Level", "Franchise"]
               Always include "Forecast Level" for accuracy reporting.
    filters  : optional {col: value} filters
    window   : time window — 'last_month', 'last_3_months', 'last_12_months',
               'ytd', or 'YYYY-MM:YYYY-MM'. If None, uses all actuals history.

    Returns
    -------
    Table with one row per group and accuracy columns for each lag.
    """
    ds = _get_ds()
    from metrics import compute_metrics

    result = compute_metrics(
        df_raw=ds.df,
        group_by_cols=group_by,
        filters=filters,
        window=window
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 5 — compute_bias_summary
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_bias_summary(
    group_by: list[str],
    filters: Optional[dict] = None,
    window: Optional[str] = None,
) -> dict:
    """
    Compute forecast bias (DF and Stat, all four lags).

    Bias % = Sum(Act Vol - Fcst Vol) / Sum(Act Vol)
    Positive bias → under-forecast; Negative bias → over-forecast.

    Parameters
    ----------
    group_by : hierarchy columns to group by
    filters  : optional {col: value} filters
    window   : time window string

    Returns
    -------
    Table with Bias % and Bias Units columns per lag.
    """
    ds = _get_ds()
    from metrics import compute_metrics

    result = compute_metrics(
        df_raw=ds.df,
        group_by_cols=group_by,
        filters=filters,
        window=window
    )

    # Keep only bias-related columns + group + actuals
    if not result.is_empty():
        keep = group_by + ["Sum Act Vol"] + [
            c for c in result.columns
            if "Bias" in c
        ]
        keep = [c for c in keep if c in result.columns]
        result = result.select(keep)

    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 6 — compute_fva_summary
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_fva_summary(
    group_by: list[str],
    filters: Optional[dict] = None,
    window: Optional[str] = None,
) -> dict:
    """
    Compute Forecast Value Add (FVA = DF Accuracy - Stat Accuracy).

    Positive FVA → DF process adds value over statistical baseline.
    Negative FVA → Stat forecast beats DF (demand planner is hurting accuracy).

    Parameters
    ----------
    group_by : hierarchy columns to group by
    filters  : optional {col: value} filters
    window   : time window string

    Returns
    -------
    Table with FVA columns per lag plus both accuracy series.
    """
    ds = _get_ds()
    from metrics import compute_metrics

    result = compute_metrics(
        df_raw=ds.df,
        group_by_cols=group_by,
        filters=filters,
        window=window
    )

    if not result.is_empty():
        keep = group_by + ["Sum Act Vol"] + [
            c for c in result.columns
            if "Accuracy" in c or "FVA" in c
        ]
        keep = [c for c in keep if c in result.columns]
        result = result.select(keep)

    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 7 — get_accuracy_trend
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_accuracy_trend(
    group_by: list[str],
    filters: Optional[dict] = None,
    window: str = "last_12_months",
    lag: str = "L2",
) -> dict:
    """
    Monthly accuracy trend for both DF and Stat at the specified lag.

    Parameters
    ----------
    group_by : hierarchy columns to group by (SALES_DATE is always added)
    filters  : optional {col: value} filters
    window   : time window (default last_12_months)
    lag      : 'L2' | 'L1' | 'L0' | 'Fcst' (default L2)

    Returns
    -------
    Table with one row per (group, month) with accuracy and bias metrics.
    Use this data to build a line chart showing accuracy trend over time.
    """
    ds = _get_ds()
    from metrics import compute_accuracy_trend

    result = compute_accuracy_trend(
        df_raw=ds.df,
        group_by_cols=group_by,
        filters=filters,
        window=window,
        lag=lag,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 8 — get_yoy_growth
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_yoy_growth(
    group_by: list[str],
    filters: Optional[dict] = None,
) -> dict:
    """
    Year-over-year volume growth for Actuals, DF Forecast, and Stat Forecast.

    Uses the Fcst lag (current month forecast) for forecast YoY.
    Only completed months are included (actuals period).

    Parameters
    ----------
    group_by : hierarchy columns to group by
    filters  : optional {col: value} filters

    Returns
    -------
    Table with Sum Act Vol, Sum DF Fcst Vol, Sum Stat Fcst Vol, and YoY %
    for each year per group.
    """
    ds = _get_ds()
    from metrics import compute_yoy_growth

    result = compute_yoy_growth(
        df_raw=ds.df,
        group_by_cols=group_by,
        filters=filters,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 9 — get_forecast_evolution
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_forecast_evolution(
    filters: Optional[dict] = None,
    window: Optional[str] = None,
) -> dict:
    """
    Show how the DF and Stat forecasts evolved across lags (L2 → L1 → L0 → Fcst)
    versus actuals, aggregated by month.

    Use this to answer: "How did our forecast change as we got closer to the period?"
    and "Which lag introduced the most error?"

    Parameters
    ----------
    filters : optional {col: value} filters to scope to a product/location
    window  : time window string

    Returns
    -------
    Table with one row per SALES_DATE showing Sum Act Vol and all lag forecasts.
    """
    ds = _get_ds()
    from metrics import compute_forecast_evolution

    result = compute_forecast_evolution(
        df_raw=ds.df,
        filters=filters,
        window=window,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 10 — get_forecast_evolution_accuracy
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_forecast_evolution_accuracy(
    filters: Optional[dict] = None,
    window: Optional[str] = None,
) -> dict:
    """
    Show how forecast accuracy developed across lags (L2 → L1 → L0) per month.

    Use this to see if our forecast accuracy improved as we got closer to
    the period (i.e., is L0 more accurate than L2?).

    Parameters
    ----------
    filters : optional {col: value} filters to scope to a product/location
    window  : time window string

    Returns
    -------
    Table with one row per SALES_DATE showing Sum Act Vol and lag accuracies.
    """
    ds = _get_ds()
    from metrics import compute_forecast_evolution_accuracy

    result = compute_forecast_evolution_accuracy(
        df_raw=ds.df,
        filters=filters,
        window=window,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 11 — get_top_offenders
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_top_offenders(
    rank_by_col: str,
    metric: str = "abs_err_df",
    lag: str = "L2",
    n: int = 10,
    filters: Optional[dict] = None,
    window: str = "last_3_months",
    ascending: bool = True,
) -> dict:
    """
    Rank entities at a given hierarchy level by a chosen metric.

    Use iteratively to drill down the hierarchy:
    - Start at Franchise level to find worst franchises
    - Filter to the worst franchise, then rank by Product Line
    - Continue down to CatalogNumber for root cause

    Parameters
    ----------
    rank_by_col : hierarchy column to rank, e.g. 'Franchise', 'Product Line',
                  'CatalogNumber', 'Region', 'Country'
    metric      : 'abs_err_df' | 'abs_err_stat' | 'accuracy' | 'bias_df' |
                  'bias_stat' | 'fva'
                  (default: abs_err_df — highest absolute error = worst)
    lag         : 'L2' | 'L1' | 'L0' | 'Fcst' (default L2)
    n           : number of top offenders to return (default 10)
    filters     : optional {col: value} filters to scope the analysis,
                  e.g. {"Franchise": "Trauma"} to find top offenders within Trauma
    window      : 'last_month' | 'last_3_months' (default last_3_months)
    ascending   : True = worst first (high abs err, low accuracy);
                  set False to get best performers instead

    Returns
    -------
    Ranked table with metric columns for all lags.
    """
    ds = _get_ds()
    from metrics import get_top_offenders as _top

    result = _top(
        df_raw=ds.df,
        rank_by_col=rank_by_col,
        metric=metric,
        lag=lag,
        n=n,
        filters=filters,
        window=window,
        ascending=ascending,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 12 — detect_anomalies_in_trend
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def detect_anomalies_in_trend(
    group_by: list[str],
    metric: Literal["accuracy", "bias", "volume"] = "accuracy",
    lag: str = "L2",
    window: str = "last_12_months",
    filters: Optional[dict] = None,
    threshold_std: float = 1.1,
    return_all: bool = False,
    top_n: Optional[int] = 15,
) -> dict:
    """
    Detect statistical anomalies in metric trends over time.

    Uses rolling statistics (3-month rolling mean ± N standard deviations) to
    identify months where the metric deviated significantly from the trend.

    Use this to find:
    - Sudden drops in forecast accuracy that need investigation
    - Unusual bias spikes indicating process issues
    - Volume anomalies that may explain accuracy problems

    Parameters
    ----------
    group_by : hierarchy columns to group by (SALES_DATE is always added),
               e.g. ["Forecast Level"], ["Franchise"], ["Product Line"]
    metric : 'accuracy' | 'bias' | 'volume' — which metric to analyze
    lag : 'L2' | 'L1' | 'L0' | 'Fcst' — forecast lag (for accuracy/bias)
    window : time window for analysis (default: last_12_months)
    filters : optional {col: value} filters to scope the analysis
    threshold_std : number of standard deviations for anomaly detection (default: 1.1)
                    Lower = more sensitive (more anomalies), Higher = fewer anomalies
                    Try 0.6 for sensitive, 1.0+ for strict
    return_all : if True, return all data points; if False (default), return only anomalies
    top_n : if set, return only top N anomalies sorted by deviation magnitude
            (if no anomalies found, returns top N deviations anyway)

    Returns
    -------
    Table with one row per anomaly (or per month if return_all=True) including:
      - metric value (accuracy/bias/volume)
      - rolling_mean, rolling_std (3-month window)
      - lower_bound, upper_bound (anomaly thresholds)
      - is_anomaly (boolean)
      - anomaly_direction ('high' | 'low' | None)
      - deviation_magnitude (z-score for ranking)

    Usage pattern
    -------------
    1. Call with group_by=["Forecast Level"] to find anomalous regions
    2. Filter to anomalous region, call with group_by=["Franchise"]
    3. Continue drilling down to root cause level
    4. Use findings to add drill-down slides via drill_down_slide()

    Examples
    --------
    # Get only anomalies (default, compact output):
    detect_anomalies_in_trend(group_by=["Forecast Level"])

    # Get top 5 worst anomalies (recommended for LLM context):
    detect_anomalies_in_trend(group_by=["Franchise"], top_n=5)

    # Use higher threshold for fewer anomalies:
    detect_anomalies_in_trend(group_by=["Product Line"], threshold_std=1.0)

    # Get all data for custom analysis:
    detect_anomalies_in_trend(group_by=["Forecast Level"], return_all=True)

    # Sensitive detection (more anomalies):
    detect_anomalies_in_trend(group_by=["Forecast Level"], threshold_std=0.6)
    """
    ds = _get_ds()
    from metrics import detect_anomalies_in_trend as _detect

    result = _detect(
        df_raw=ds.df,
        group_by_cols=group_by,
        metric=metric,
        lag=lag,
        window=window,
        filters=filters,
        threshold_std=threshold_std,
        return_all=return_all,
        top_n=top_n,
    )
    return _df_to_dict(result)


# ════════════════════════════════════════════════════════════════════════════
# TOOL 13 — add_slide
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def add_slide(
    layout: str,
    title: str,
    subtitle: str = "",
    commentary: str = "",
    chart: Optional[dict] = None,
    table: Optional[dict] = None,
    chart2: Optional[dict] = None,
) -> dict:
    """
    Add a slide to the current presentation and auto-save the HTML file.

    Call generate_standard_report() first (it auto-initialises the presentation).
    Use this tool for custom slides beyond the standard set.

    Layouts
    -------
    - "title"       : Cover or section divider. Uses title + subtitle.
    - "metrics"     : KPI cards. Pass metric card data as JSON string in commentary:
                      '[{"label":"DF Accuracy","value":"72.3%","delta":"vs 68% LY","direction":"up"}, ...]'
    - "chart"       : Full-width chart + optional commentary.
    - "table"       : Full-width table + optional commentary.
    - "chart_table" : Side-by-side chart (left) and table (right).
    - "two_col"     : Two charts side by side.
    - "commentary"  : Text-only slide for narrative/observations.

    Chart spec (pass as dict in `chart` or `chart2`)
    -------------------------------------------------
    {
      "chart_type": "line" | "bar" | "grouped_bar" | "waterfall",
      "title": "Chart Title",
      "x_data": [...],
      "y_data": [...] or {"Series A": [...], "Series B": [...]},
      "x_label": "Month",
      "y_label": "Volume",
      "colors": ["#hex1", "#hex2"],
      "show_legend": true,
      "height": 380
    }

    Table spec (pass as dict in `table`)
    -------------------------------------
    {
      "headers": ["Col1", "Col2", ...],
      "rows": [[val1, val2, ...], ...],
      "highlight_col": 2,
      "highlight_thresholds": [60, 80]
    }

    Returns
    -------
    Slide index, confirmation, and output_path of the auto-saved HTML file.
    """
    state = _get_state()
    if state.presentation is None:
        return {
            "error": (
                "No presentation active. Call generate_standard_report() first "
                "to auto-initialise a presentation, or load_presentation() to "
                "resume an existing one."
            )
        }

    from presentation import Slide, ChartSpec, TableSpec

    def _make_chart(c: dict | None) -> ChartSpec | None:
        if not c:
            return None
        return ChartSpec(
            chart_type=c.get("chart_type", "bar"),
            title=c.get("title", ""),
            x_data=c.get("x_data", []),
            y_data=c.get("y_data", []),
            x_label=c.get("x_label", ""),
            y_label=c.get("y_label", ""),
            colors=c.get("colors"),
            show_legend=c.get("show_legend", True),
            height=c.get("height", 380),
        )

    def _make_table(t: dict | None) -> TableSpec | None:
        if not t:
            return None
        return TableSpec(
            headers=t.get("headers", []),
            rows=t.get("rows", []),
            highlight_col=t.get("highlight_col"),
            highlight_thresholds=tuple(t["highlight_thresholds"]) if t.get("highlight_thresholds") else None,
        )

    slide = Slide(
        layout=layout,
        title=title,
        subtitle=subtitle,
        commentary=commentary,
        chart=_make_chart(chart),
        table=_make_table(table),
        chart2=_make_chart(chart2),
    )
    state.presentation.add_slide(slide)
    out_path = state.presentation.auto_save()

    return {
        "status": "slide_added",
        "slide_index": len(state.presentation.slides) - 1,
        "layout": layout,
        "title": title,
        "output_path": str(out_path) if out_path else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 12 — generate_standard_report
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def generate_standard_report(
    title: str = "Demand Planning Review",
    subtitle: str = "",
    window: str = "last_3_months",
    filters: Optional[dict] = None,
) -> dict:
    """
    *** START HERE for every new presentation. ***

    Auto-initialises a fresh presentation (no need to call initialize_presentation),
    pre-computes all standard metrics at multiple hierarchy levels, builds standard
    slides automatically (cover, KPI summary, accuracy trend, Forecast Level table,
    Product Line table, IBP Level 5 table, bias chart, FVA chart, YoY growth chart),
    and returns a compact briefing JSON (~700 tokens) for the LLM to write commentary.

    The HTML file is written automatically and updated after every subsequent
    add_commentary() or drill_down_slide() call — no finalize step needed.

    Workflow
    --------
      1. Call generate_standard_report()    — slides built, briefing returned.
      2. Call add_commentary(slide_id, ...) — write narrative for each slide.
      3. Optionally call drill_down_slide() — for anomalies worth investigating.
      (HTML is saved automatically after steps 2 and 3.)

    Parameters
    ----------
    title    : Presentation title shown on the cover slide.
               Default: 'Demand Planning Review'.
    subtitle : Optional subtitle (e.g. period or team name).
    window   : Time window — 'last_month', 'last_3_months', 'last_12_months',
               'ytd', or 'YYYY-MM:YYYY-MM'. Default: last_3_months.
    filters  : Optional {col: value} to scope to a Franchise, Product Line, etc.
               e.g. {"Franchise": "Trauma"}. Omit for a full-portfolio report.

    Returns
    -------
    briefing : dict with keys:
        window, filters_applied, date_range,
        by_forecast_level  — list of {Forecast Level, df_acc, stat_acc, bias_pct, fva, act_vol}
        by_product_line    — list of same shape, ranked worst first
        by_ibp5            — list of same shape at IBP Level 5 grain
        root_cause_hints   — top 5 worst CatalogNumber/IBP Level 7 rows for commentary
        trend              — {months, df_acc, stat_acc} arrays for 12-month chart
        yoy                — [{year, act_vol, yoy_pct}, ...] last 3 years
        slides_created     — list of {slide_id, title} for slides just built
        output_path        — path of the auto-saved HTML file
    """
    state = _get_state()
    from presentation import PresentationBuilder
    from briefing import generate_standard_report as _briefing, build_standard_slides

    # Auto-initialise (or reset) the presentation — no separate tool call needed.
    state.presentation = PresentationBuilder(
        brand=state.brand,
        output_dir=state.config.get("output_dir", "output"),
    )
    state.presentation.initialize(title=title, subtitle=subtitle)

    ds = _get_ds()
    briefing = _briefing(ds, window=window, filters=filters)
    build_standard_slides(briefing, state.presentation)

    # Auto-save HTML + state JSON after building standard slides.
    out_path = state.presentation.auto_save()

    briefing["slides_created"] = [
        {"slide_id": s["slide_id"], "title": s["title"]}
        for s in state.presentation.status()
    ]
    briefing["output_path"] = str(out_path) if out_path else None

    return briefing


# ════════════════════════════════════════════════════════════════════════════
# TOOL 13 — add_commentary
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def add_commentary(slide_id: str, commentary: str) -> dict:
    """
    Add or update the commentary text on a specific slide, then auto-save the HTML.

    Call this after generate_standard_report() to write narrative for each slide.
    Write 2-4 bullet points or sentences per slide based on the briefing data.

    Parameters
    ----------
    slide_id   : Named or 8-char slide ID from get_presentation_status() or
                 slides_created in generate_standard_report() response.
                 Standard IDs: 'cover', 'kpi_summary', 'accuracy_trend',
                 'by_forecast_level', 'by_product_line', 'by_ibp5',
                 'bias_summary', 'fva_summary', 'yoy_growth'.
    commentary : Text for the commentary box. Supports plain text and HTML:
                 '<ul><li>Point 1</li><li>Point 2</li></ul>'
                 '<strong>key term</strong>'

    Returns
    -------
    {status, slide_id, title, output_path} or {error} if slide_id not found.
    """
    state = _get_state()
    if state.presentation is None:
        return {
            "error": (
                "No presentation active. Call generate_standard_report() first "
                "or load_presentation() to resume an existing one."
            )
        }

    slide = state.presentation.add_commentary_by_id(slide_id, commentary)
    if slide is None:
        available = [s["slide_id"] for s in state.presentation.status()]
        return {
            "error": f"slide_id '{slide_id}' not found.",
            "available_slide_ids": available,
        }

    out_path = state.presentation.auto_save()
    return {
        "status": "commentary_added",
        "slide_id": slide.slide_id,
        "title": slide.title,
        "output_path": str(out_path) if out_path else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 14 — get_presentation_status
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_presentation_status() -> dict:
    """
    Return the current state of all slides in the active presentation.

    Use this to:
    - See which slides already have commentary (has_commentary: true) vs. need it.
    - Resume after a context reset — call this one tool to know exactly where
      you left off before continuing with add_commentary() or drill_down_slide().
    - Verify that generate_standard_report() created the expected slides.

    Returns
    -------
    {
        "active": bool,              # false if no presentation is loaded
        "slide_count": int,
        "output_path": str | null,   # current auto-save path
        "slides": [
            {
                "slide_index": int,
                "slide_id": str,
                "title": str,
                "layout": str,
                "has_commentary": bool
            },
            ...
        ],
        "commentary_complete": bool
    }
    """
    state = _get_state()
    if state.presentation is None:
        return {
            "active": False,
            "slide_count": 0,
            "output_path": None,
            "slides": [],
            "commentary_complete": False,
            "hint": "Call generate_standard_report() to start a new presentation.",
        }

    slides = state.presentation.status()
    out_path = None
    if state.presentation._auto_filename:
        candidate = state.presentation.output_dir / state.presentation._auto_filename
        out_path = str(candidate) if candidate.exists() else None

    return {
        "active": True,
        "slide_count": len(slides),
        "output_path": out_path,
        "slides": slides,
        "commentary_complete": all(s["has_commentary"] for s in slides) if slides else False,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 15 — drill_down_slide
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def drill_down_slide(
    hierarchy_col: str,
    hierarchy_value: str,
    metric: str = "accuracy",
    window: str = "last_3_months",
    filters: Optional[dict] = None,
) -> dict:
    """
    Append a focused drill-down slide, then auto-save the HTML.

    Automatically resolves the NEXT level down in the hierarchy and
    computes a metric table at that level, scoped to the parent node.

    Hierarchy drill paths
    ---------------------
    Product : Franchise → Product Line → IBP Level 5 → IBP Level 6
                        → IBP Level 7 → CatalogNumber
    Location: Forecast Level → Area / Region → Country

    Parameters
    ----------
    hierarchy_col   : The PARENT column to drill from.
                      e.g. 'Franchise', 'Product Line', 'Forecast Level'
    hierarchy_value : The value to scope to, e.g. 'Trauma', 'APAC'
    metric          : 'accuracy' | 'bias' | 'fva' | 'trend' | 'top_offenders'
    window          : time window string (default: last_3_months)
    filters         : optional additional {col: value} filters

    Returns
    -------
    {
        "slide_id"        : str    — use with add_commentary()
        "title"           : str,
        "drill_level"     : str,
        "parent_col"      : str,
        "parent_value"    : str,
        "briefing_summary": list,   — compact rows for commentary
        "output_path"     : str     — path of the auto-saved HTML
    }
    """
    state = _get_state()
    if state.presentation is None:
        return {
            "error": (
                "No presentation active. Call generate_standard_report() first "
                "or load_presentation() to resume an existing one."
            )
        }

    from briefing import compute_drill_down
    from presentation import Slide, ChartSpec, TableSpec

    ds = _get_ds()

    try:
        result = compute_drill_down(
            ds=ds,
            hierarchy_col=hierarchy_col,
            hierarchy_value=hierarchy_value,
            metric=metric,
            window=window,
            filters=filters,
        )
    except ValueError as e:
        return {"error": str(e)}

    if "error" in result:
        return result

    drill_level = result["drill_level"]
    slide_data = result["slide_data"]
    layout = slide_data["layout"]
    title = f"{metric.title()}: {hierarchy_value} → {drill_level}"

    def _cs(c):
        if not c:
            return None
        return ChartSpec(
            chart_type=c["chart_type"],
            title=c["title"],
            x_data=c["x_data"],
            y_data=c["y_data"],
            x_label=c.get("x_label", ""),
            y_label=c.get("y_label", ""),
            show_legend=c.get("show_legend", True),
            height=c.get("height", 320),
        )

    def _ts(t):
        if not t:
            return None
        return TableSpec(
            headers=t["headers"],
            rows=t["rows"],
            highlight_col=t.get("highlight_col"),
            highlight_thresholds=tuple(t["highlight_thresholds"]) if t.get("highlight_thresholds") else None,
        )

    slide = Slide(
        layout=layout,
        title=title,
        chart=_cs(slide_data.get("chart_spec")),
        table=_ts(slide_data.get("table_spec")),
    )
    state.presentation.add_slide(slide)
    out_path = state.presentation.auto_save()

    return {
        "status": "slide_added",
        "slide_id": slide.slide_id,
        "title": title,
        "drill_level": drill_level,
        "parent_col": hierarchy_col,
        "parent_value": hierarchy_value,
        "briefing_summary": result["rows"],
        "output_path": str(out_path) if out_path else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 16 — list_presentations
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_presentations() -> dict:
    """
    List all saved presentation HTML files in the output directory.

    Use this to discover which presentations exist and which ones can
    be loaded for editing.

    Returns
    -------
    {
        "output_dir": str,
        "presentations": [
            {
                "filename"  : str,   # e.g. "demand_review_20260225_1200.html"
                "path"      : str,   # full absolute path
                "has_state" : bool,  # True → editable via load_presentation()
                "size_kb"   : float,
                "modified"  : str,   # ISO datetime
            },
            ...
        ]
    }
    """
    from datetime import datetime as _dt

    state = _get_state()
    output_dir = Path(state.config.get("output_dir", "output"))

    if not output_dir.exists():
        return {"output_dir": str(output_dir), "presentations": []}

    entries = []
    for html_path in sorted(output_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        state_path = html_path.with_suffix(html_path.suffix + ".state.json")
        stat = html_path.stat()
        entries.append({
            "filename" : html_path.name,
            "path"     : str(html_path.resolve()),
            "has_state": state_path.exists(),
            "size_kb"  : round(stat.st_size / 1024, 1),
            "modified" : _dt.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })

    return {
        "output_dir"   : str(output_dir.resolve()),
        "presentations": entries,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 17 — load_presentation
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def load_presentation(filename: str) -> dict:
    """
    Load an existing presentation for editing.

    Reads the sidecar .state.json written alongside the HTML file and restores
    the full in-memory session (all slides with their original slide_ids).
    After calling this tool you can:
      • add_commentary(slide_id, ...)   — update/add commentary; auto-saves HTML
      • add_slide(...)                  — append new slides; auto-saves HTML
      • drill_down_slide(...)           — append drill-down slides; auto-saves HTML
      • get_presentation_status()       — see the full slide list

    Parameters
    ----------
    filename : HTML filename to load (just the name, e.g. "demand_review_20260225.html").
               Must exist in the configured output directory and have a matching
               .state.json sidecar file created by finalize_presentation().

    Returns
    -------
    On success:
        {
            "status"        : "loaded",
            "filename"      : str,
            "title"         : str,
            "subtitle"      : str,
            "slide_count"   : int,
            "slides"        : [...],  # same format as get_presentation_status()
            "commentary_complete": bool,
        }
    On error:
        {"error": str, "available": [list of editable filenames]}
    """
    from presentation import PresentationBuilder

    state = _get_state()
    output_dir = Path(state.config.get("output_dir", "output"))

    html_path  = output_dir / filename
    state_path = html_path.with_suffix(html_path.suffix + ".state.json")

    # Build list of editable files for helpful error messages.
    def _editable():
        if not output_dir.exists():
            return []
        return [
            p.name for p in sorted(output_dir.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.with_suffix(p.suffix + ".state.json").exists()
        ]

    if not html_path.exists():
        return {
            "error"    : f"File not found: {html_path}",
            "available": _editable(),
        }

    if not state_path.exists():
        return {
            "error": (
                f"No editable state found for '{filename}'. "
                "Only presentations created by this server (finalize_presentation) "
                "can be reloaded. The sidecar file would be: "
                f"{state_path.name}"
            ),
            "available": _editable(),
        }

    try:
        state.presentation = PresentationBuilder.from_state_file(
            state_path=state_path,
            brand=state.brand,
            output_dir=output_dir,
        )
    except Exception as exc:
        return {"error": f"Failed to load state: {exc}"}

    slides = state.presentation.status()
    return {
        "status"             : "loaded",
        "filename"           : filename,
        "title"              : state.presentation.title,
        "subtitle"           : state.presentation.subtitle,
        "slide_count"        : len(slides),
        "slides"             : slides,
        "commentary_complete": all(s["has_commentary"] for s in slides) if slides else False,
    }


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Demand Planning MCP Server")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config.yaml. If omitted, checks DEMAND_MCP_CONFIG env var, "
            "then config.yaml next to server.py, then config.yaml in cwd."
        ),
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help=(
            "Transport mode:\n"
            "  stdio — used when an LLM client (e.g. Claude Desktop, ollama-mcp) "
            "          spawns this process directly. Do NOT run manually in a terminal.\n"
            "  sse   — HTTP+SSE server at http://host:port/sse. "
            "          Use when your LLM client connects over HTTP."
        ),
    )
    parser.add_argument("--host", default=None, help="Host for SSE transport (overrides config)")
    parser.add_argument("--port", default=None, type=int, help="Port for SSE transport (overrides config)")
    args = parser.parse_args()

    # Bootstrap here so errors surface immediately in the terminal,
    # rather than silently failing when the first tool is called.
    _bootstrap(config_path=args.config)
    state = _get_state()

    if args.transport == "sse":
        host = args.host or state.config.get("server", {}).get("host", "127.0.0.1")
        port = args.port or state.config.get("server", {}).get("port", 8000)
        logger.info(f"Starting MCP server (SSE) at http://{host}:{port}/sse")
        logger.info("Point your LLM client to: http://%s:%s/sse", host, port)
        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("Starting MCP server (stdio) — waiting for LLM client to connect via stdin/stdout.")
        logger.info("Do NOT type into this terminal. Launch via your LLM client's MCP config instead.")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
