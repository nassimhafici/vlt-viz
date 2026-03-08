import pandas as pd

RET_COLS = ["r1d", "r1w", "r1m", "r3m", "rytd", "r1y"]

COL_LABELS = {
    "symbol": "Symbol", "name": "Name", "asset_class": "Class",
    "country_cd": "Country", "sector_cd": "Sector", "currency": "Ccy",
    "r1d": "1D", "r1w": "1W", "r1m": "1M", "r3m": "3M", "rytd": "YTD", "r1y": "1Y",
}

BUCKET_LABELS = {
    "asset_class": "Asset Class",
    "country_cd":  "Country",
    "sector_cd":   "Sector",
}

# ── Light palette ─────────────────────────────────────────────────
BG       = "#f9fafb"   # page background
BG2      = "#ffffff"   # cards, table rows
BG3      = "#f3f4f6"   # hover, subtle fill
BORDER   = "#e5e7eb"   # dividers
TEXT     = "#111827"   # primary text
TEXT_DIM = "#6b7280"   # labels, secondary
TEXT_MID = "#374151"   # mid-level
GREEN    = "#059669"   # positive
RED      = "#dc2626"   # negative
BLUE     = "#2563eb"   # accent / active
YELLOW   = "#d97706"   # warning / neutral-warm
GRAY     = "#9ca3af"   # N/A, disabled

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"


# ── Formatters — always 2 decimals ───────────────────────────────
def fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 100:.2f}%"


def fmt_bps(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}bp"


def fmt_return(val, is_diff=False) -> str:
    return fmt_bps(val) if is_diff else fmt_pct(val)


def fmt_price(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return f"{val:.2f}"


# ── Color helpers ────────────────────────────────────────────────
def color_return(val) -> str:
    if pd.isna(val): return f"color: {GRAY}"
    return f"color: {GREEN}" if val > 0 else f"color: {RED}"


def color_return_bg(val) -> str:
    if pd.isna(val): return f"color: {GRAY}"
    if val > 0.05:    return f"color: {GREEN}; font-weight: 500"
    elif val > 0:     return f"color: #16a34a"
    elif val > -0.05: return f"color: {RED}"
    else:             return f"color: {RED}; font-weight: 500"


# ── Shared table styles ──────────────────────────────────────────
def _table_styles():
    return [
        {"selector": "th", "props": [
            ("background", BG3),
            ("color", TEXT_DIM),
            ("font-family", FONT),
            ("font-size", "9px"),
            ("font-weight", "600"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.06em"),
            ("border-bottom", f"1px solid {BORDER}"),
            ("border-top", f"1px solid {BORDER}"),
            ("padding", "8px 12px"),
        ]},
        {"selector": "td", "props": [
            ("font-family", FONT),
            ("font-size", "12px"),
            ("color", TEXT),
            ("padding", "6px 12px"),
            ("border-bottom", f"1px solid {BORDER}"),
            ("background", BG2),
        ]},
        {"selector": "tr:hover td", "props": [
            ("background", BG3),
        ]},
    ]


def style_returns_table(df: pd.DataFrame, ret_cols: list):
    present = [c for c in ret_cols if c in df.columns]
    return (
        df.style
        .applymap(color_return, subset=present)
        .format({c: (lambda x: fmt_pct(x) if pd.notna(x) else "—") for c in present})
        .set_table_styles(_table_styles())
    )


def style_pivot_table(df: pd.DataFrame):
    return (
        df.style
        .applymap(color_return_bg)
        .format(lambda x: fmt_pct(x) if pd.notna(x) else "—")
        .set_table_styles(_table_styles())
    )


# ─────────────────────────────────────────────────────────────────
# Category filter helper — shared across pages
# ─────────────────────────────────────────────────────────────────
CATEGORY_LABELS = {
    "country":      "Countries",
    "sector-us":    "US Sectors",
    "sector-world": "World Sectors",
    "sector-ca":    "Canada Sectors",
    "style":        "Styles & Factors",
    "thematic":     "Thematics",
    "commodity":    "Commodities",
    "benchmark":    "Benchmarks",
    "vol-etf":      "Vol ETFs",
    "vix-index":    "VIX Indices",
    "rate":         "Rates",
    "cash":         "Cash",
}

def apply_category_filter(df: "pd.DataFrame", key: str = "cat_filter") -> "pd.DataFrame":
    """
    Render a Category selectbox in the current sidebar context.
    Returns the filtered DataFrame. Requires 'category' column in df.
    Pass a unique key per page to avoid session_state conflicts.
    """
    import streamlit as _st

    cats_in_data = (
        df["category"].dropna().unique().tolist()
        if "category" in df.columns else []
    )
    ordered  = [c for c in CATEGORY_LABELS if c in cats_in_data]
    ordered += [c for c in sorted(cats_in_data) if c not in ordered]
    options  = ["All"] + [CATEGORY_LABELS.get(c, c) for c in ordered]
    _cat_map = {CATEGORY_LABELS.get(c, c): c for c in ordered}

    sel = _st.selectbox("Category", options, key=key)
    if sel == "All":
        return df
    raw_cat = _cat_map.get(sel, sel)
    return df[df["category"] == raw_cat].copy()


# ─────────────────────────────────────────────────────────────────
# Unified KPI card — use this on every page
# ─────────────────────────────────────────────────────────────────
def kpi_card(col, label: str, value: str, sub: str = "", vc: str = "neutral"):
    """
    Render a styled KPI card into a Streamlit column.
    vc: "green" | "red" | "yellow" | "neutral"
    """
    import streamlit as _st
    themes = {
        "green":   ("#f0fdf4", "#16a34a", "#14532d"),
        "red":     ("#fef2f2", "#dc2626", "#7f1d1d"),
        "yellow":  ("#fffbeb", "#d97706", "#78350f"),
        "neutral": ("#f8fafc", "#64748b", "#1e293b"),
    }
    bg, accent, tc = themes.get(vc, themes["neutral"])
    col.markdown(
        f"<div style='"
        f"background:{bg};"
        f"border-radius:8px;"
        f"padding:12px 14px 10px 14px;"
        f"border-top:2px solid {accent};"
        f"min-height:72px;"
        f"box-sizing:border-box'>"
        f"<div style='"
        f"font-family:{FONT};"
        f"font-size:9px;"
        f"font-weight:700;"
        f"letter-spacing:0.12em;"
        f"text-transform:uppercase;"
        f"color:{accent};"
        f"margin-bottom:6px;"
        f"white-space:nowrap;"
        f"overflow:hidden;"
        f"text-overflow:ellipsis'>{label}</div>"
        f"<div style='"
        f"font-family:{FONT};"
        f"font-size:18px;"
        f"font-weight:300;"
        f"color:{tc};"
        f"line-height:1;"
        f"letter-spacing:-0.01em'>{value}</div>"
        f"<div style='"
        f"font-family:{FONT};"
        f"font-size:10px;"
        f"color:{accent};"
        f"opacity:0.65;"
        f"margin-top:4px;"
        f"white-space:nowrap;"
        f"overflow:hidden;"
        f"text-overflow:ellipsis'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )