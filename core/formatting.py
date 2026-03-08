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

# ── Palette ──────────────────────────────────────────────────────
BG       = "#f9fafb"
BG2      = "#ffffff"
BG3      = "#f3f4f6"
BORDER   = "#e5e7eb"

TEXT     = "#111827"
TEXT_DIM = "#6b7280"
TEXT_MID = "#374151"   # mid-level

GREEN    = "#059669"
RED      = "#dc2626"
BLUE     = "#2563eb"
YELLOW   = "#d97706"
GRAY     = "#9ca3af"

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"


# ── Formatters ───────────────────────────────────────────────────
def fmt_pct(val):
    if val is None or pd.isna(val):
        return "—"

    arrow = "▲" if val > 0 else "▼" if val < 0 else ""
    return f"{arrow} {val*100:.2f}%"


def fmt_bps(val):
    if val is None or pd.isna(val):
        return "—"

    arrow = "▲" if val > 0 else "▼" if val < 0 else ""
    return f"{arrow} {val:.2f}bp"


def fmt_return(val, is_diff=False):
    return fmt_bps(val) if is_diff else fmt_pct(val)


def fmt_price(val):
    if val is None or pd.isna(val):
        return "—"
    return f"{val:.2f}"


# ── Color helpers ─────────────────────────────────────────────────
def color_return(val):
    if pd.isna(val):
        return f"color:{GRAY}"

    return f"color:{GREEN}" if val > 0 else f"color:{RED}"


# ── Performance bar (visual) ─────────────────────────────────────
def bar_return(val):

    if pd.isna(val):
        return ""

    width = min(abs(val) * 400, 100)

    color = GREEN if val > 0 else RED

    return f"""
        background: linear-gradient(
            90deg,
            {color}22 {width}%,
            transparent {width}%
        );
    """


# ── Highlight best / worst ───────────────────────────────────────
def highlight_extreme(s):

    if s.dtype != float:
        return [''] * len(s)

    max_val = s.max()
    min_val = s.min()

    return [
        f"font-weight:600;color:{GREEN}" if v == max_val else
        f"font-weight:600;color:{RED}" if v == min_val else ""
        for v in s
    ]


# ── Sparkline (unicode) ──────────────────────────────────────────
SPARK = "▁▂▃▄▅▆▇"

def sparkline(data):

    if len(data) == 0:
        return ""

    mn, mx = min(data), max(data)

    if mx == mn:
        return SPARK[0] * len(data)

    return "".join(
        SPARK[int((x-mn)/(mx-mn)*(len(SPARK)-1))]
        for x in data
    )


# ── Shared table styles ──────────────────────────────────────────
def _table_styles():

    return [

        {
            "selector": "thead th",
            "props": [

                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "2"),

                ("background", BG3),
                ("color", TEXT_DIM),

                ("font-family", FONT),
                ("font-size", "8px"),
                ("font-weight", "600"),

                ("text-transform", "uppercase"),
                ("letter-spacing", "0.06em"),

                ("border-bottom", f"1px solid {BORDER}"),
                ("border-top", f"1px solid {BORDER}"),

                ("padding", "6px 8px"),
            ],
        },

        {
            "selector": "td",
            "props": [

                ("font-family", FONT),
                ("font-size", "11px"),

                ("color", TEXT),

                ("padding", "4px 8px"),

                ("border-bottom", f"1px solid {BORDER}"),
                ("background", BG2),

            ],
        },

        {
            "selector": "tr:hover td",
            "props": [
                ("background", BG3)
            ],
        },

    ]


# ── Returns table ─────────────────────────────────────────────────
def style_returns_table(df: pd.DataFrame, ret_cols: list):

    present = [c for c in ret_cols if c in df.columns]

    styler = (
        df.style
        .applymap(color_return, subset=present)
        .applymap(bar_return, subset=present)
        .apply(highlight_extreme, subset=present)
        .format({c: fmt_pct for c in present})
        .set_table_styles(_table_styles())
    )

    return styler


# ── Pivot table (heatmap) ─────────────────────────────────────────
def style_pivot_table(df: pd.DataFrame):

    return (
        df.style
        .background_gradient(
            cmap="RdYlGn",
            vmin=-0.05,
            vmax=0.05
        )
        .format(lambda x: fmt_pct(x) if pd.notna(x) else "—")
        .set_table_styles(_table_styles())
    )


# ─────────────────────────────────────────────────────────────────
# Category filter helper
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


def apply_category_filter(df: pd.DataFrame, key: str = "cat_filter"):

    import streamlit as st

    cats_in_data = (
        df["category"].dropna().unique().tolist()
        if "category" in df.columns else []
    )

    ordered  = [c for c in CATEGORY_LABELS if c in cats_in_data]
    ordered += [c for c in sorted(cats_in_data) if c not in ordered]

    options  = ["All"] + [CATEGORY_LABELS.get(c, c) for c in ordered]

    _cat_map = {CATEGORY_LABELS.get(c, c): c for c in ordered}

    sel = st.selectbox("Category", options, key=key)

    if sel == "All":
        return df

    raw_cat = _cat_map.get(sel, sel)

    return df[df["category"] == raw_cat].copy()


# ─────────────────────────────────────────────────────────────────
# KPI card
# ─────────────────────────────────────────────────────────────────
def kpi_card(col, label: str, value: str, sub: str = "", vc: str = "neutral"):

    import streamlit as st

    themes = {

        "green":   ("#f0fdf4", "#16a34a", "#14532d"),
        "red":     ("#fef2f2", "#dc2626", "#7f1d1d"),
        "yellow":  ("#fffbeb", "#d97706", "#78350f"),
        "neutral": ("#f8fafc", "#64748b", "#1e293b"),

    }

    bg, accent, tc = themes.get(vc, themes["neutral"])

    arrow = "▲" if vc == "green" else "▼" if vc == "red" else ""

    col.markdown(

        f"""
        <div style="
        background:{bg};
        border-radius:8px;
        padding:12px 14px;
        border-top:2px solid {accent};
        min-height:70px">

        <div style="
        font-family:{FONT};
        font-size:9px;
        font-weight:700;
        letter-spacing:0.12em;
        text-transform:uppercase;
        color:{accent};
        margin-bottom:6px">

        {label}

        </div>

        <div style="
        font-family:{FONT};
        font-size:18px;
        font-weight:300;
        color:{tc};
        line-height:1">

        {value}

        </div>

        <div style="
        font-family:{FONT};
        font-size:10px;
        color:{accent};
        opacity:0.7;
        margin-top:4px">

        {arrow} {sub}

        </div>

        </div>
        """,

        unsafe_allow_html=True,
    )