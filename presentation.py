"""
presentation.py — Reveal.js HTML presentation builder.

Manages a presentation session in memory. The LLM calls:
  1. initialize_presentation(title, subtitle)
  2. add_slide(...) one or more times
  3. finalize_presentation() → writes HTML file and returns path

Charts are rendered as Plotly JSON embedded inline (no server needed).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import polars as pl

SlideLayout = Literal["title", "metrics", "chart", "table", "chart_table", "commentary", "two_col"]


@dataclass
class ChartSpec:
    """Plotly chart specification."""
    chart_type: Literal["line", "bar", "grouped_bar", "waterfall", "heatmap", "scatter"]
    title: str
    x_data: list
    y_data: list | dict  # dict for multi-series: {"Series Name": [values]}
    x_label: str = ""
    y_label: str = ""
    colors: list[str] | None = None
    show_legend: bool = True
    height: int = 380


@dataclass
class TableSpec:
    """Table specification."""
    headers: list[str]
    rows: list[list]
    highlight_col: int | None = None  # column index to color-code
    highlight_thresholds: tuple | None = None  # (low, high) for red/green


@dataclass
class Slide:
    layout: SlideLayout
    title: str
    subtitle: str = ""
    commentary: str = ""
    chart: ChartSpec | None = None
    table: TableSpec | None = None
    chart2: ChartSpec | None = None  # for two_col layout
    slide_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        # Allow briefing.py to pass a fixed human-readable slide_id.
        # If the field is still an auto-generated UUID fragment, leave it.
        pass


class PresentationBuilder:
    """Stateful presentation builder. One instance per presentation session."""

    def __init__(self, brand: dict, output_dir: str | Path):
        self.brand = brand
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slides: list[Slide] = []
        self.title = "Demand Planning Review"
        self.subtitle = ""
        self.created_at = datetime.now()

    def initialize(self, title: str, subtitle: str = "") -> None:
        self.title = title
        self.subtitle = subtitle
        self.slides = []

    def add_slide(self, slide: Slide) -> None:
        self.slides.append(slide)

    def add_commentary_by_id(self, slide_id: str, commentary: str) -> Slide | None:
        """Find a slide by slide_id and update its commentary. Returns the slide or None."""
        for slide in self.slides:
            if slide.slide_id == slide_id:
                slide.commentary = commentary
                return slide
        return None

    def status(self) -> list[dict]:
        """Return a compact status list for get_presentation_status()."""
        return [
            {
                "slide_index": i,
                "slide_id": s.slide_id,
                "title": s.title,
                "layout": s.layout,
                "has_commentary": bool(s.commentary and s.commentary.strip()),
            }
            for i, s in enumerate(self.slides)
        ]

    def finalize(self, filename: str | None = None) -> Path:
        if filename is None:
            ts = self.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"demand_review_{ts}.html"

        out_path = self.output_dir / filename
        html = self._render_html()
        out_path.write_text(html, encoding="utf-8")
        return out_path

    # ── HTML rendering ──────────────────────────────────────────────────────

    def _render_html(self) -> str:
        c = self.brand["colors"]
        f = self.brand["fonts"]

        slides_html = "\n".join(self._render_slide(s) for s in self.slides)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title}</title>

  <!-- Reveal.js -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/reveal.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/theme/white.min.css">

  <!-- Plotly -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>

  <style>
    :root {{
      --primary:      {c['primary']};
      --primary-dark: {c['primary_dark']};
      --secondary:    {c['secondary']};
      --bg:           {c['background']};
      --surface:      {c['surface']};
      --text:         {c['text_primary']};
      --text-muted:   {c['text_secondary']};
      --positive:     {c['positive']};
      --negative:     {c['negative']};
      --neutral:      {c['neutral']};
      --font-heading: {f['heading']};
      --font-body:    {f['body']};
    }}

    .reveal {{ font-family: var(--font-body); background: var(--bg); }}
    .reveal h1, .reveal h2, .reveal h3 {{
      font-family: var(--font-heading);
      color: var(--secondary);
      text-transform: none;
      letter-spacing: -0.02em;
    }}
    .reveal .slides section {{ text-align: left; padding: 0.5rem 1.5rem; }}

    /* Title slide */
    .slide-title-content {{
      display: flex; flex-direction: column; justify-content: center;
      height: 100%; padding: 2rem;
    }}
    .slide-title-content h1 {{ font-size: 2rem; color: var(--secondary); margin-bottom: 0.5rem; }}
    .slide-title-content .subtitle {{ font-size: 1rem; color: var(--text-muted); }}
    .title-bar {{
      width: 80px; height: 5px; background: var(--primary);
      margin: 1rem 0;
    }}

    /* Section header bar */
    .slide-header {{
      background: var(--secondary);
      color: white;
      margin: -0.5rem -1.5rem 1rem -1.5rem;
      padding: 0.5rem 1.5rem;
      font-family: var(--font-heading);
      font-size: 1.6rem;
      font-weight: 600;
      display: flex; align-items: center; gap: 0.75rem;
    }}
    .slide-header .accent-bar {{ width: 4px; height: 1.5rem; background: var(--primary); }}

    /* Metric cards */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
      margin: 0.5rem 0;
    }}
    .metric-card {{
      background: var(--surface);
      border-left: 4px solid var(--primary);
      border-radius: 4px;
      padding: 0.75rem 1rem;
    }}
    .metric-card.positive {{ border-left-color: var(--positive); }}
    .metric-card.negative {{ border-left-color: var(--negative); }}
    .metric-card .label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: var(--secondary); line-height: 1.1; }}
    .metric-card .delta {{ font-size: 0.75rem; margin-top: 0.2rem; }}
    .delta.up {{ color: var(--positive); }}
    .delta.down {{ color: var(--negative); }}

    /* Tables */
    .data-table {{
      width: 100%; border-collapse: collapse;
      font-size: 0.78rem;
    }}
    .data-table th {{
      background: var(--secondary); color: white;
      padding: 0.4rem 0.6rem; text-align: left;
      font-weight: 600; font-size: 0.72rem;
      text-transform: uppercase; letter-spacing: 0.03em;
    }}
    .data-table td {{
      padding: 0.35rem 0.6rem;
      border-bottom: 1px solid #e8e8e8;
    }}
    .data-table tr:nth-child(even) td {{ background: var(--surface); }}
    .data-table tr:hover td {{ background: #fff3e0; }}
    .cell-positive {{ color: var(--positive); font-weight: 600; }}
    .cell-negative {{ color: var(--negative); font-weight: 600; }}
    .cell-neutral  {{ color: var(--neutral); }}

    /* Commentary */
    .commentary-box {{
      background: var(--surface);
      border-left: 3px solid var(--primary);
      border-radius: 0 4px 4px 0;
      padding: 0.75rem 1rem;
      font-size: 0.82rem;
      color: var(--text);
      line-height: 1.6;
      margin-top: 0.75rem;
    }}
    .commentary-box ul {{ margin: 0.3rem 0 0 1.2rem; padding: 0; }}
    .commentary-box li {{ margin-bottom: 0.2rem; }}

    /* Two-col layout */
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}

    /* Chart wrapper */
    .chart-wrap {{ width: 100%; }}

    /* Footer */
    .slide-footer {{
      position: absolute; bottom: 0.4rem; right: 1.5rem;
      font-size: 0.6rem; color: var(--text-muted);
    }}
  </style>
</head>
<body>
<div class="reveal">
  <div class="slides">
    {slides_html}
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/reveal.min.js"></script>
<script>
  Reveal.initialize({{
    hash: true,
    transition: '{self.brand["presentation"]["slide_transition"]}',
    slideNumber: 'c/t',
    controls: true,
    progress: true,
    center: false,
  }});

  // Render all Plotly charts after reveal initialises
  Reveal.on('ready', () => renderAllCharts());
  Reveal.on('slidechanged', () => renderAllCharts());

  function renderAllCharts() {{
    document.querySelectorAll('[data-plotly]').forEach(el => {{
      if (el.dataset.rendered) return;
      try {{
        const spec = JSON.parse(el.dataset.plotly);
        Plotly.newPlot(el, spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
        el.dataset.rendered = '1';
      }} catch(e) {{ console.error('Plotly error', e); }}
    }});
  }}
</script>
</body>
</html>"""

    def _render_slide(self, slide: Slide) -> str:
        if slide.layout == "title":
            return self._render_title_slide(slide)
        elif slide.layout == "metrics":
            return self._render_metrics_slide(slide)
        elif slide.layout == "chart":
            return self._render_chart_slide(slide)
        elif slide.layout == "table":
            return self._render_table_slide(slide)
        elif slide.layout == "chart_table":
            return self._render_chart_table_slide(slide)
        elif slide.layout == "commentary":
            return self._render_commentary_slide(slide)
        elif slide.layout == "two_col":
            return self._render_two_col_slide(slide)
        return f"<section><h2>{slide.title}</h2></section>"

    def _render_title_slide(self, slide: Slide) -> str:
        company = self.brand.get("company_name", "")
        return f"""<section>
  <div class="slide-title-content">
    <div style="color: var(--text-muted); font-size: 0.85rem;">{company}</div>
    <h1>{slide.title}</h1>
    <div class="title-bar"></div>
    <div class="subtitle">{slide.subtitle or self.subtitle}</div>
    <div style="margin-top: 2rem; font-size: 0.75rem; color: var(--text-muted);">
      {self.created_at.strftime('%B %Y')}
    </div>
  </div>
</section>"""

    def _render_metrics_slide(self, slide: Slide) -> str:
        # commentary may carry structured metric data as JSON string
        # Format: [{"label":"..","value":"..","delta":"..","direction":"up|down|neutral"}]
        cards_html = ""
        try:
            cards = json.loads(slide.commentary) if slide.commentary.startswith("[") else []
            for card in cards:
                direction = card.get("direction", "neutral")
                css_class = "positive" if direction == "up" else ("negative" if direction == "down" else "")
                delta_html = ""
                if card.get("delta"):
                    delta_class = "up" if direction == "up" else ("down" if direction == "down" else "")
                    arrow = "▲" if direction == "up" else ("▼" if direction == "down" else "–")
                    delta_html = f'<div class="delta {delta_class}">{arrow} {card["delta"]}</div>'
                cards_html += f"""<div class="metric-card {css_class}">
  <div class="label">{card.get("label","")}</div>
  <div class="value">{card.get("value","")}</div>
  {delta_html}
</div>"""
        except Exception:
            cards_html = f'<div class="commentary-box">{slide.commentary}</div>'

        return f"""<section>
  {self._header(slide.title)}
  <div class="metrics-grid">{cards_html}</div>
  {self._footer()}
</section>"""

    def _render_chart_slide(self, slide: Slide) -> str:
        chart_html = self._chart_html(slide.chart, slide.slide_id) if slide.chart else ""
        comm_html = self._commentary_html(slide.commentary)
        return f"""<section>
  {self._header(slide.title)}
  {chart_html}
  {comm_html}
  {self._footer()}
</section>"""

    def _render_table_slide(self, slide: Slide) -> str:
        table_html = self._table_html(slide.table) if slide.table else ""
        comm_html = self._commentary_html(slide.commentary)
        return f"""<section>
  {self._header(slide.title)}
  {table_html}
  {comm_html}
  {self._footer()}
</section>"""

    def _render_chart_table_slide(self, slide: Slide) -> str:
        chart_html = self._chart_html(slide.chart, slide.slide_id, height=280) if slide.chart else ""
        table_html = self._table_html(slide.table) if slide.table else ""
        comm_html = self._commentary_html(slide.commentary)
        return f"""<section>
  {self._header(slide.title)}
  <div class="two-col">
    <div>{chart_html}</div>
    <div>{table_html}</div>
  </div>
  {comm_html}
  {self._footer()}
</section>"""

    def _render_two_col_slide(self, slide: Slide) -> str:
        left_html = self._chart_html(slide.chart, slide.slide_id + "L", height=300) if slide.chart else ""
        right_html = self._chart_html(slide.chart2, slide.slide_id + "R", height=300) if slide.chart2 else ""
        comm_html = self._commentary_html(slide.commentary)
        return f"""<section>
  {self._header(slide.title)}
  <div class="two-col">
    <div>{left_html}</div>
    <div>{right_html}</div>
  </div>
  {comm_html}
  {self._footer()}
</section>"""

    def _render_commentary_slide(self, slide: Slide) -> str:
        return f"""<section>
  {self._header(slide.title)}
  <div class="commentary-box">{slide.commentary}</div>
  {self._footer()}
</section>"""

    def _header(self, title: str) -> str:
        return f"""<div class="slide-header">
  <div class="accent-bar"></div>
  {title}
  <img style="margin-left:auto;" src="{self.brand.get('logo_path', '')}" alt="Logo">
</div>"""

    def _footer(self) -> str:
        company = self.brand.get("company_name", "")
        return f'<div class="slide-footer">{company} Confidential · {self.created_at.strftime("%B %Y")}</div>'

    def _commentary_html(self, commentary: str) -> str:
        if not commentary:
            return ""
        return f'<div class="commentary-box">{commentary}</div>'

    def _table_html(self, spec: TableSpec) -> str:
        if spec is None:
            return ""

        header_cells = "".join(f"<th>{h}</th>" for h in spec.headers)

        rows_html = ""
        for row in spec.rows:
            cells = ""
            for i, cell in enumerate(row):
                css = ""
                if (spec.highlight_col is not None
                        and i == spec.highlight_col
                        and spec.highlight_thresholds):
                    low, high = spec.highlight_thresholds
                    try:
                        val = float(str(cell).replace("%", "").replace(",", ""))
                        if val >= high:
                            css = ' class="cell-positive"'
                        elif val <= low:
                            css = ' class="cell-negative"'
                        else:
                            css = ' class="cell-neutral"'
                    except Exception:
                        pass
                cells += f"<td{css}>{cell}</td>"
            rows_html += f"<tr>{cells}</tr>"

        return f"""<table class="data-table">
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

    def _chart_html(
        self,
        spec: ChartSpec,
        chart_id: str,
        height: int | None = None,
    ) -> str:
        if spec is None:
            return ""

        c = self.brand["colors"]
        palette = spec.colors or c["chart_palette"]
        h = height or spec.height or self.brand["presentation"]["chart_height_px"]

        if spec.chart_type == "line":
            plotly_spec = self._line_chart(spec, palette, h)
        elif spec.chart_type in ("bar", "grouped_bar"):
            plotly_spec = self._bar_chart(spec, palette, h)
        elif spec.chart_type == "waterfall":
            plotly_spec = self._waterfall_chart(spec, palette, h)
        else:
            plotly_spec = self._bar_chart(spec, palette, h)  # fallback

        spec_json = json.dumps(plotly_spec)
        div_id = f"chart_{chart_id}"
        return f'<div id="{div_id}" class="chart-wrap" style="height:{h}px" data-plotly=\'{spec_json}\'></div>'

    def _base_layout(self, spec: ChartSpec, height: int) -> dict:
        f = self.brand["fonts"]
        c = self.brand["colors"]
        return {
            "title": {"text": spec.title, "font": {"size": 13, "color": c["secondary"]}, "x": 0},
            "height": height,
            "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": f["body"], "size": 11, "color": c["text_primary"]},
            "legend": {"orientation": "h", "y": -0.2, "x": 0},
            "showlegend": spec.show_legend,
            "xaxis": {"title": spec.x_label, "gridcolor": "#e8e8e8"},
            "yaxis": {"title": spec.y_label, "gridcolor": "#e8e8e8"},
        }

    def _line_chart(self, spec: ChartSpec, palette: list, height: int) -> dict:
        traces = []
        if isinstance(spec.y_data, dict):
            for i, (name, values) in enumerate(spec.y_data.items()):
                traces.append({
                    "type": "scatter", "mode": "lines+markers",
                    "name": name, "x": spec.x_data, "y": values,
                    "line": {"color": palette[i % len(palette)], "width": 2},
                    "marker": {"size": 5},
                })
        else:
            traces.append({
                "type": "scatter", "mode": "lines+markers",
                "x": spec.x_data, "y": spec.y_data,
                "line": {"color": palette[0], "width": 2},
            })
        return {"data": traces, "layout": self._base_layout(spec, height)}

    def _bar_chart(self, spec: ChartSpec, palette: list, height: int) -> dict:
        barmode = "group" if spec.chart_type == "grouped_bar" else "relative"
        traces = []
        if isinstance(spec.y_data, dict):
            for i, (name, values) in enumerate(spec.y_data.items()):
                traces.append({
                    "type": "bar", "name": name,
                    "x": spec.x_data, "y": values,
                    "marker": {"color": palette[i % len(palette)]},
                })
        else:
            colors = [
                palette[0] if v >= 0 else self.brand["colors"]["negative"]
                for v in spec.y_data
            ]
            traces.append({
                "type": "bar",
                "x": spec.x_data, "y": spec.y_data,
                "marker": {"color": colors},
            })
        layout = self._base_layout(spec, height)
        layout["barmode"] = barmode
        return {"data": traces, "layout": layout}

    def _waterfall_chart(self, spec: ChartSpec, palette: list, height: int) -> dict:
        c = self.brand["colors"]
        traces = [{
            "type": "waterfall",
            "x": spec.x_data,
            "y": spec.y_data,
            "connector": {"line": {"color": "#cccccc"}},
            "increasing": {"marker": {"color": c["positive"]}},
            "decreasing": {"marker": {"color": c["negative"]}},
            "totals": {"marker": {"color": c["secondary"]}},
        }]
        return {"data": traces, "layout": self._base_layout(spec, height)}
