# Demand Planning MCP Server

An MCP server exposing **19 tools** for automated demand planning analysis.
The LLM orchestrates these tools to navigate product/location hierarchies,
compute accuracy/bias/FVA metrics, identify root causes, and produce
a self-contained Reveal.js HTML presentation — and to load and edit
an existing presentation without rebuilding it from scratch.

---

## Project Structure

```
demand_mcp/
├── server.py          # FastMCP server entry point — 19 tools
├── data.py            # Data loading, PackContent normalization, filters
├── metrics.py         # Accuracy, bias, FVA, YoY, evolution computations
├── presentation.py    # Reveal.js + Plotly HTML builder
├── config.yaml        # Server configuration (paths, fiscal year, defaults)
├── brand.json         # Brand colors, fonts — edit to match your org
├── requirements.txt
├── prompts/
│   └── system.md      # System prompt for your LLM orchestrator
└── output/            # Generated HTML presentations go here
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Configuration

### 1. Edit `config.yaml`

```yaml
parquet_path: "path/to/your/demand.parquet"
output_dir:   "output"
brand_config: "brand.json"
fiscal_year_start_month: 1   # 1=Jan, 4=Apr, 7=Jul, 10=Oct
```

### 2. Edit `brand.json`

Prefilled with Stryker defaults. Update `colors`, `fonts`, and `logo_path`
to match your organization's branding. Key fields:

```json
{
  "company_name": "Your Company",
  "colors": {
    "primary":   "#F4A11D",   // main accent (buttons, bars, highlights)
    "secondary": "#003087",   // headings, slide headers
    "df_color":  "#F4A11D",   // DF forecast line/bars in charts
    "stat_color":"#003087",   // Stat forecast line/bars in charts
    "actual_color":"#2E7D32"  // Actuals line/bars in charts
  },
  "fonts": {
    "heading": "Calibri, Arial, sans-serif",
    "body":    "Calibri, Arial, sans-serif"
  }
}
```

---

## Running the Server

```bash
cd demand_mcp
python server.py
# or with a custom config:
python server.py --config /path/to/config.yaml
```

The server starts on `127.0.0.1:8000` (configurable in `config.yaml`).
The parquet file is loaded **once at startup** — restart the server to pick up new data.

---

## Connecting your LLM

Configure your local LLM client to connect to the MCP server.
Copy the contents of `prompts/system.md` as the system prompt.

Example using any OpenAI-compatible client with MCP support:

```python
client = YourLLMClient(
    mcp_server_url="http://127.0.0.1:8000",
    system_prompt=open("prompts/system.md").read(),
)
response = client.chat("Generate a demand planning review for last month.")
```

---

## The 19 Tools

### Data Exploration
| # | Tool | Purpose |
|---|------|---------|
| 1 | `get_data_info` | Row count, date range, hierarchy unique counts |
| 2 | `get_hierarchy_members` | List values at any hierarchy level (with optional filters) |
| 3 | `get_date_range` | Available months (with optional filters) |

### Metrics
| # | Tool | Purpose |
|---|------|---------|
| 4 | `compute_accuracy_summary` | Accuracy (all 4 lags) grouped by any hierarchy combo |
| 5 | `compute_bias_summary` | Bias % and units (all 4 lags) |
| 6 | `compute_fva_summary` | FVA = DF Accuracy − Stat Accuracy |
| 7 | `get_accuracy_trend` | Monthly accuracy trend for a given lag |
| 8 | `get_yoy_growth` | Year-over-year volume growth vs DF and Stat forecasts |
| 9 | `get_forecast_evolution` | L2 → L1 → L0 → Fcst vs Actuals by month |
| 10 | `get_top_offenders` | Rank hierarchy members by any metric (for drill-down) |

### Presentation (build new)
| # | Tool | Purpose |
|---|------|---------|
| 11 | `initialize_presentation` | Start a new blank HTML presentation session |
| 12 | `add_slide` | Append a slide (title/metrics/chart/table/chart_table/two_col/commentary) |
| 13 | `finalize_presentation` | Write HTML + sidecar `.state.json`; returns output path |

### Efficient Report Generation
| # | Tool | Purpose |
|---|------|---------|
| 14 | `generate_standard_report` | Pre-compute all 9 standard slides + return briefing JSON in one call |
| 15 | `add_commentary` | Write narrative text to a named slide (by `slide_id`) |
| 16 | `get_presentation_status` | List all slides, IDs, and commentary state — use to resume after reset |
| 17 | `drill_down_slide` | Append a focused drill-down slide at the next hierarchy level |

### Editing Existing Presentations
| # | Tool | Purpose |
|---|------|---------|
| 18 | `list_presentations` | List saved HTML files in the output dir; flags which are editable |
| 19 | `load_presentation` | Reload a saved presentation into memory for editing |

---

## Time Windows

All metric tools accept a `window` parameter:

| Value | Description |
|-------|-------------|
| `last_month` | Single most recent completed month |
| `last_3_months` | Three most recent completed months |
| `last_12_months` | Rolling 12 months |
| `ytd` | January 1 to last completed month |
| `YYYY-MM:YYYY-MM` | Explicit range, e.g. `2024-01:2024-06` |
| `None` | All available actuals history |

---

## Hierarchy Reference

### Location (coarse → fine)
`Stryker Group Region` → `Area` → `Region` → `Country`

### Product (coarse → fine)
`Business Sector` → `Business Unit` → `Franchise` → `Product Line`
→ `IBP Level 5` → `IBP Level 6` → `IBP Level 7` → `CatalogNumber`

**Forecast Level** (used for accuracy reporting):
- East Asia Area → uses **Area** as the forecast level
- All other → uses **Region** as the forecast level

---

## Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Accuracy** | `1 - ΣAbsErr / ΣAct` | Higher = better. Target ≥ 70% |
| **Bias %** | `Σ(Act−Fcst) / ΣAct` | +ve = under-forecast, −ve = over-forecast |
| **Bias Units** | `Σ(Act−Fcst)` | Signed volume gap |
| **FVA** | `DF Acc − Stat Acc` | +ve = DF adds value; −ve = Stat beats DF |

All volume series are **PackContent-normalized** (divided by PackContent at load time).

---

## Output

Generated presentations are saved to the `output_dir` specified in `config.yaml`.
Open the `.html` file in any browser — no server needed, all dependencies are CDN-linked.

Use arrow keys or click to navigate slides. Speaker notes can be toggled with `S`.

### Sidecar state file

`finalize_presentation()` writes two files every time:

```
output/
  demand_review_20260225_1200.html           ← the browser presentation
  demand_review_20260225_1200.html.state.json ← structured slide data (editable)
```

The `.state.json` stores every slide's layout, chart data, table data, commentary,
and `slide_id`. `load_presentation(filename)` reads this file to restore the full
editing session — all subsequent tools (`add_slide`, `add_commentary`,
`drill_down_slide`, `finalize_presentation`) then work exactly as if the session
had never ended.
