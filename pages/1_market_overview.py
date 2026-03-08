import pandas as pd
import streamlit as st
from core.db import load_returns_with_meta
from core.formatting import (
    FONT, COL_LABELS, RET_COLS, CATEGORY_LABELS, apply_category_filter,
    style_returns_table, fmt_pct, GREEN, RED, BORDER, BG2, BG3,
    TEXT, TEXT_DIM, GRAY, kpi_card
)
from core.charts import returns_bar

st.title("overview")

# ── Load ─────────────────────────────────────────────────────────
df = load_returns_with_meta()
if df.empty:
    st.warning("No data."); st.stop()

EXCLUDE_CLASSES = {"Fixed Income", "Volatility"}
df = df[~df["asset_class"].isin(EXCLUDE_CLASSES)].copy()

# ── Sidebar — Category filter, default = Benchmarks ──────────────
with st.sidebar:
    display_as = st.radio("Display", ["Name", "Ticker"], horizontal=True)
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)

    cats_in_data = df["category"].dropna().unique().tolist() if "category" in df.columns else []
    ordered  = [c for c in CATEGORY_LABELS if c in cats_in_data]
    ordered += [c for c in sorted(cats_in_data) if c not in ordered]
    options  = ["All"] + [CATEGORY_LABELS.get(c, c) for c in ordered]
    _cat_map = {CATEGORY_LABELS.get(c, c): c for c in ordered}

    default_idx = options.index("Benchmarks") if "Benchmarks" in options else 0
    sel_cat = st.selectbox("Category", options, index=default_idx, key="cat_overview")

filtered = df.copy()
if sel_cat != "All":
    filtered = filtered[filtered["category"] == _cat_map.get(sel_cat, sel_cat)]

# ── Period pills ──────────────────────────────────────────────────
WINDOWS = [("r1d","1D"),("r1w","1W"),("r1m","1M"),("r3m","3M"),("rytd","YTD"),("r1y","1Y")]
if "ret_window" not in st.session_state:
    st.session_state.ret_window = "r1m"

# Pills — all real st.button; active wrapped in div.pill-active-wrap for CSS targeting
_pill_cols = st.columns(len(WINDOWS), gap="small")
for _col, (_key, _label) in zip(_pill_cols, WINDOWS):
    _is_active = (_key == st.session_state.ret_window)
    with _col:
        if _is_active:
            st.markdown("<div class='pill-active-wrap'>", unsafe_allow_html=True)
        if st.button(_label, key=f"w_{_key}", use_container_width=True):
            st.session_state.ret_window = _key
            st.rerun()
        if _is_active:
            st.markdown("</div>", unsafe_allow_html=True)

ret_window = st.session_state.ret_window
wlabel     = dict(WINDOWS)[ret_window]

# ── KPI strip ─────────────────────────────────────────────────────
valid = filtered[ret_window].dropna()
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if not valid.empty:
    top    = filtered.nlargest(1,  ret_window).iloc[0]
    bottom = filtered.nsmallest(1, ret_window).iloc[0]
    n_pos  = int((valid > 0).sum())
    n_neg  = int((valid < 0).sum())
    n      = len(valid)
    avg    = valid.mean()

    def _name(row):
        raw = row["name"] if display_as == "Name" else row["symbol"]
        return (raw[:22] + "…") if len(str(raw)) > 24 else str(raw)

    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    kpi_card(k1, "Best",     fmt_pct(top[ret_window]),    _name(top),    "green")
    kpi_card(k2, "Worst",    fmt_pct(bottom[ret_window]), _name(bottom), "red")
    kpi_card(k3, "Average",  fmt_pct(avg),                f"{n} assets", "green" if avg>=0 else "red")
    kpi_card(k4, "Positive", f"{n_pos/n*100:.0f}%",       f"{n_pos} of {n}", "green")
    kpi_card(k5, "Negative", f"{n_neg/n*100:.0f}%",       f"{n_neg} of {n}", "red")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ── Bar chart ─────────────────────────────────────────────────────
st.markdown(
    f"<p style='font-family:{FONT};font-size:9px;font-weight:600;color:{TEXT_DIM};"
    f"text-transform:uppercase;letter-spacing:0.12em;margin:0 0 10px 0'>"
    f"TOP &amp; BOTTOM — {wlabel}</p>", unsafe_allow_html=True)

chart_df = filtered.copy()
chart_df["_label"] = chart_df["name"].str[:30] if display_as == "Name" else chart_df["symbol"]
st.plotly_chart(returns_bar(chart_df, ret_col=ret_window, top_n=12, label_col="_label"), width="stretch")

# ── Table ─────────────────────────────────────────────────────────
st.markdown(
    f"<p style='font-family:{FONT};font-size:9px;font-weight:600;color:{TEXT_DIM};"
    f"text-transform:uppercase;letter-spacing:0.12em;margin:36px 0 10px 0'>"
    f"ALL — {len(filtered)} assets</p>", unsafe_allow_html=True)

tbl = filtered.copy()
if display_as == "Name":
    tbl["symbol"] = tbl["name"]

display_cols = ["symbol", "category", "country_cd"] + RET_COLS
available    = [c for c in display_cols if c in tbl.columns]
col_rename   = {**COL_LABELS, "symbol": "NAME" if display_as == "Name" else "SYMBOL",
                "category": "CATEGORY", "country_cd": "COUNTRY"}
table        = tbl[available].copy().rename(columns=col_rename)
ret_display  = [COL_LABELS[c] for c in RET_COLS if c in tbl.columns]
st.dataframe(style_returns_table(table, ret_display),
             use_container_width=True, height=520, hide_index=True)