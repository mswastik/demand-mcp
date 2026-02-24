"""
data.py — Data loading and normalization layer.

Loads the parquet file once at startup. All tools consume the DataStore singleton.
PackContent normalization is applied here so no tool ever sees raw counts.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

# ── Column name constants ────────────────────────────────────────────────────

# Location hierarchy (coarse → fine)
LOC_HIERARCHY = [
    "Stryker Group Region",
    "Area",
    "Region",
    "Country",
]

# Product hierarchy (coarse → fine)
PROD_HIERARCHY = [
    "Business Sector",
    "Business Unit",
    "Franchise",
    "Product Line",
    "IBP Level 5",
    "IBP Level 6",
    "IBP Level 7",
    "CatalogNumber",
]

# All hierarchy columns together
ALL_HIERARCHY = LOC_HIERARCHY + PROD_HIERARCHY

# Volume series (after dividing by PackContent)
ACT_VOL = "Act Orders Vol"
FCST_DF_VOL = "Fcst DF Final Rev Vol"
FCST_STAT_VOL = "Fcst Stat Final Rev Vol"

L0_DF_VOL = "L0 DF Final Rev Vol"
L1_DF_VOL = "L1 DF Final Rev Vol"
L2_DF_VOL = "L2 DF Final Rev Vol"

L0_STAT_VOL = "L0 Stat Final Rev Vol"
L1_STAT_VOL = "L1 Stat Final Rev Vol"
L2_STAT_VOL = "L2 Stat Final Rev Vol"

# Revenue series (raw, no PackContent division)
ACT_REV = "Act Orders Rev Val"
FCST_DF_REV = "Fcst DF Final Rev Val"
FCST_STAT_REV = "Fcst Stat Final Rev Val"

# Raw source column names in parquet
_RAW_ACT = "`Act Orders Rev"
_RAW_FCST_DF = "`Fcst DF Final Rev"
_RAW_FCST_STAT = "`Fcst Stat Final Rev"
_RAW_L0_DF = "L0 DF Final Rev"
_RAW_L1_DF = "L1 DF Final Rev"
_RAW_L2_DF = "L2 DF Final Rev"
_RAW_L0_STAT = "L0 Stat Final Rev"
_RAW_L1_STAT = "L1 Stat Final Rev"
_RAW_L2_STAT = "L2 Stat Final Rev"

LAG_DF_COLS = {
    "L2": (_RAW_L2_DF, L2_DF_VOL),
    "L1": (_RAW_L1_DF, L1_DF_VOL),
    "L0": (_RAW_L0_DF, L0_DF_VOL),
    "Fcst": (_RAW_FCST_DF, FCST_DF_VOL),
}

LAG_STAT_COLS = {
    "L2": (_RAW_L2_STAT, L2_STAT_VOL),
    "L1": (_RAW_L1_STAT, L1_STAT_VOL),
    "L0": (_RAW_L0_STAT, L0_STAT_VOL),
    "Fcst": (_RAW_FCST_STAT, FCST_STAT_VOL),
}

ALL_LAGS = ["L2", "L1", "L0", "Fcst"]

# ── Forecast level logic ─────────────────────────────────────────────────────

FORECAST_LEVEL_COL = "Forecast Level"


# ── DataStore singleton ──────────────────────────────────────────────────────

class DataStore:
    """
    Holds the fully normalized DataFrame in memory.
    Call DataStore.load(path) once at server startup.
    """

    _instance: Optional["DataStore"] = None
    df: pl.DataFrame
    loaded_at: datetime
    parquet_path: Path

    @classmethod
    def load(cls, parquet_path: str | Path) -> "DataStore":
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        logger.info(f"Loading parquet from {path} …")
        raw = pl.read_parquet(path)

        ds = cls.__new__(cls)
        ds.parquet_path = path
        ds.loaded_at = datetime.now()
        ds.df = _normalize(raw)  # Forecast Level column included inside _normalize

        cls._instance = ds
        logger.info(
            f"DataStore loaded: {ds.df.shape[0]:,} rows, "
            f"{ds.df.shape[1]} columns. Loaded at {ds.loaded_at:%Y-%m-%d %H:%M:%S}"
        )
        return ds

    @classmethod
    def get(cls) -> "DataStore":
        if cls._instance is None:
            raise RuntimeError("DataStore not initialized. Call DataStore.load() first.")
        return cls._instance

    def date_range(self) -> tuple[date, date]:
        dates = self.df["SALES_DATE"].cast(pl.Date)
        return dates.min(), dates.max()

    def hierarchy_members(self, level: str, filters: Optional[dict] = None) -> list[str]:
        """Return unique values at a given hierarchy level, optionally filtered."""
        df = self.df
        if filters:
            df = apply_filters(df, filters)
        col = _resolve_hierarchy_col(level)
        return sorted(df[col].drop_nulls().unique().to_list())


# ── Internal helpers ─────────────────────────────────────────────────────────

def _normalize(raw: pl.DataFrame) -> pl.DataFrame:
    """
    Apply PackContent division to all volume series.
    Revenue series are kept as-is.
    SALES_DATE is cast to Date.

    All numeric columns are cast to Float64 first to handle:
      - Decimal types (polars arithmetic on Decimal raises errors)
      - Integer types (division would truncate without cast)
    PackContent zero/null values are replaced with 1.0 to avoid division-by-zero
    (a PackContent of 0 is a data quality issue; treating as 1 preserves raw value).
    """
    # ── 1. Cast all relevant numeric columns from Decimal/Int → Float64 ──────
    numeric_raw_cols = [
        _RAW_ACT, _RAW_FCST_DF, _RAW_FCST_STAT,
        _RAW_L0_DF, _RAW_L1_DF, _RAW_L2_DF,
        _RAW_L0_STAT, _RAW_L1_STAT, _RAW_L2_STAT,
        "PackContent",
        "Act Orders Rev Val", "Fcst DF Final Rev Val", "Fcst Stat Final Rev Val",
    ]
    cast_exprs = []
    for col in numeric_raw_cols:
        if col in raw.columns:
            cast_exprs.append(pl.col(col).cast(pl.Float64))
    if cast_exprs:
        raw = raw.with_columns(cast_exprs)

    # ── 2. Safe PackContent: replace 0 and null with 1.0 ────────────────────
    raw = raw.with_columns(
        pl.when(pl.col("PackContent").is_null() | (pl.col("PackContent") == 0.0))
        .then(1.0)
        .otherwise(pl.col("PackContent"))
        .alias("PackContent")
    )

    pack = pl.col("PackContent")

    # ── 3. Divide all volume series by PackContent ───────────────────────────
    exprs = [
        (pl.col(_RAW_ACT) / pack).alias(ACT_VOL),
        (pl.col(_RAW_FCST_DF) / pack).alias(FCST_DF_VOL),
        (pl.col(_RAW_FCST_STAT) / pack).alias(FCST_STAT_VOL),
        (pl.col(_RAW_L0_DF) / pack).alias(L0_DF_VOL),
        (pl.col(_RAW_L1_DF) / pack).alias(L1_DF_VOL),
        (pl.col(_RAW_L2_DF) / pack).alias(L2_DF_VOL),
        (pl.col(_RAW_L0_STAT) / pack).alias(L0_STAT_VOL),
        (pl.col(_RAW_L1_STAT) / pack).alias(L1_STAT_VOL),
        (pl.col(_RAW_L2_STAT) / pack).alias(L2_STAT_VOL),
    ]
    df = raw.with_columns(exprs)

    # ── 4. Cast date column ──────────────────────────────────────────────────
    if df["SALES_DATE"].dtype != pl.Date:
        df = df.with_columns(pl.col("SALES_DATE").cast(pl.Date))

    # ── 5. Add Forecast Level column (persistent, available to all tools) ────
    df = df.with_columns(
        pl.when(pl.col("Area") == "East Asia")
        .then(pl.col("Area"))
        .otherwise(pl.col("Region"))
        .alias(FORECAST_LEVEL_COL)
    )

    return df


def _resolve_hierarchy_col(level: str) -> str:
    """Accept level names case-insensitively or by shorthand."""
    mapping = {l.lower(): l for l in ALL_HIERARCHY + [FORECAST_LEVEL_COL]}
    key = level.lower().strip()
    if key in mapping:
        return mapping[key]
    # allow shorthand like 'franchise', 'bu', 'catalognum'
    for col in ALL_HIERARCHY:
        if col.lower().startswith(key):
            return col
    raise ValueError(f"Unknown hierarchy level: '{level}'. Valid: {ALL_HIERARCHY}")


def apply_filters(df: pl.DataFrame, filters: dict) -> pl.DataFrame:
    """
    Apply a dict of {column_name: value_or_list_of_values} filters.
    Column names are resolved case-insensitively.
    """
    for col_name, value in filters.items():
        resolved = _resolve_filter_col(df, col_name)
        if isinstance(value, list):
            df = df.filter(pl.col(resolved).is_in(value))
        else:
            df = df.filter(pl.col(resolved) == value)
    return df


def _resolve_filter_col(df: pl.DataFrame, name: str) -> str:
    """Find the actual column name in df, case-insensitively."""
    lower_map = {c.lower(): c for c in df.columns}
    key = name.lower().strip()
    if key in lower_map:
        return lower_map[key]
    for c in df.columns:
        if c.lower().startswith(key):
            return c
    raise ValueError(f"Filter column not found: '{name}'. Available: {df.columns}")


def filter_to_actuals_period(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows where SALES_DATE < current month start (i.e. actuals exist)."""
    today = date.today()
    cutoff = date(today.year, today.month, 1)
    return df.filter(pl.col("SALES_DATE") < cutoff)


def filter_date_window(df: pl.DataFrame, window: str) -> pl.DataFrame:
    """
    window: 'last_month' | 'last_3_months' | 'last_12_months' |
            'ytd' | 'YYYY-MM:YYYY-MM' (explicit range)
    """
    today = date.today()
    current_month_start = date(today.year, today.month, 1)

    if window == "last_month":
        # The single most recent completed month
        if today.month == 1:
            start = date(today.year - 1, 12, 1)
            end = date(today.year, 1, 1)
        else:
            start = date(today.year, today.month - 1, 1)
            end = current_month_start
        return df.filter(
            (pl.col("SALES_DATE") >= start) & (pl.col("SALES_DATE") < end)
        )

    elif window == "last_3_months":
        # Three most recent completed months
        year, month = today.year, today.month
        months_back = []
        for _ in range(3):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
            months_back.append(date(year, month, 1))
        start = min(months_back)
        return df.filter(
            (pl.col("SALES_DATE") >= start) & (pl.col("SALES_DATE") < current_month_start)
        )

    elif window == "last_12_months":
        year, month = today.year, today.month - 1
        if month == 0:
            month = 12
            year -= 1
        end = date(year, month, 1)
        # 12 months back from end
        s_month = end.month - 11
        s_year = end.year
        if s_month <= 0:
            s_month += 12
            s_year -= 1
        start = date(s_year, s_month, 1)
        return df.filter(
            (pl.col("SALES_DATE") >= start) & (pl.col("SALES_DATE") <= end)
        )

    elif window == "ytd":
        start = date(today.year, 1, 1)
        return df.filter(
            (pl.col("SALES_DATE") >= start) & (pl.col("SALES_DATE") < current_month_start)
        )

    elif ":" in window:
        # explicit "YYYY-MM:YYYY-MM"
        parts = window.split(":")
        start = date(int(parts[0][:4]), int(parts[0][5:7]), 1)
        end_m = int(parts[1][5:7])
        end_y = int(parts[1][:4])
        end_m_next = end_m + 1 if end_m < 12 else 1
        end_y_next = end_y if end_m < 12 else end_y + 1
        end = date(end_y_next, end_m_next, 1)
        return df.filter(
            (pl.col("SALES_DATE") >= start) & (pl.col("SALES_DATE") < end)
        )

    else:
        raise ValueError(
            f"Unknown window: '{window}'. Use: last_month, last_3_months, "
            "last_12_months, ytd, or YYYY-MM:YYYY-MM"
        )
