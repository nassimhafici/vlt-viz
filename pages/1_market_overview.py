import pandas as pd
import streamlit as st
from core.db import load_returns_with_meta
from core.formatting import (
    FONT, COL_LABELS, RET_COLS, CATEGORY_LABELS, apply_category_filter,
    style_returns_table, fmt_pct, GREEN, RED, BORDER, BG, BG2, BG3, TEXT, TEXT_DIM, TEXT_MID, GRAY
)
from core.charts import returns_bar

st.title("overview")

# ── Load ─────────────────────────────────────────────────────────
df = load_returns_with_meta()
if df.empty:
    st.warning("No data."); st.stop()

EXCLUDE_CLASSES = {"Fixed Income", "Volatility"}
df = df[~df["asset_class"].isin(EXCLUDE_CLASSES)].copy()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    display_as = st.radio("Display", ["Name", "Ticker"], horizontal=True)
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    # Category filter (replaces asset_class + country)
    filtered = apply_category_filter(df, key="cat_overview")

# ── Period pill selector ──────────────────────────────────────────
WINDOWS = [("r1d","1D"),("r1w","1W"),("r1m","1M"),("r3m","3M"),("rytd","YTD"),("r1y","1Y")]
if "ret_window" not in st.session_state:
    st.session_state.ret_window = "r1m"

st.markdown("""
<style>
[data-testid="stButton"] > button {
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    width: 100% !important;
    border-radius: 6px !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border: 1.5px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #9ca3af !important;
    box-shadow: none !important;
    transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #eff6ff !important;
}
[data-testid="stButton"] > button:focus {
    box-shadow: none !important; outline: none !important;
}
</style>
""", unsafe_allow_html=True)

pill_cols = st.columns(len(WINDOWS), gap="small")
for col, (key, label) in zip(pill_cols, WINDOWS):
    if key == st.session_state.ret_window:
        col.markdown(
            f"<div style='height:32px;display:flex;align-items:center;justify-content:center;"
            f"background:#2563eb;border-radius:6px;'>"
            f"<span style='font-family:{FONT};font-size:11px;font-weight:700;"
            f"letter-spacing:0.08em;color:#fff'>{label}</span></div>",
            unsafe_allow_html=True)
    else:
        if col.button(label, key=f"w_{key}"):
            st.session_state.ret_window = key
            st.rerun()

ret_window = st.session_state.ret_window
wlabel     = dict(WINDOWS)[ret_window]

# ── Category badge ────────────────────────────────────────────────
cat_val = st.session_state.get("__cat_filter__", "All")
cat_lbl = cat_val if cat_val == "All" else cat_val
st.markdown(
    f"<p style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
    f"text-transform:uppercase;letter-spacing:0.1em;margin:18px 0 0 0'>"
    f"{cat_lbl} &nbsp;·&nbsp; {len(filtered)} assets</p>",
    unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────
valid = filtered[ret_window].dropna()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

if not valid.empty:
    top    = filtered.nlargest(1,  ret_window).iloc[0]
    bottom = filtered.nsmallest(1, ret_window).iloc[0]
    n_pos  = int((valid > 0).sum())
    n_neg  = int((valid < 0).sum())
    n      = len(valid)
    avg    = valid.mean()

    def _name(row):
        raw = row["name"] if display_as == "Name" else row["symbol"]
        return (raw[:20] + "…") if len(raw) > 22 else raw

    def kpi_card(label, value, sub, color, bg_color):
        return (
            f"<div style='background:{bg_color};border-radius:10px;"
            f"padding:20px 24px;height:100%;'>"
            f"<div style='font-family:{FONT};font-size:9px;font-weight:700;"
            f"letter-spacing:0.14em;text-transform:uppercase;color:{color};"
            f"opacity:0.7;margin-bottom:8px'>{label}</div>"
            f"<div style='font-family:{FONT};font-size:28px;font-weight:300;"
            f"color:{color};line-height:1;letter-spacing:-0.02em'>{value}</div>"
            f"<div style='font-family:{FONT};font-size:11px;color:{color};"
            f"opacity:0.55;margin-top:8px;font-weight:400'>{sub}</div>"
            f"</div>"
        )

    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    k1.markdown(kpi_card("BEST",     fmt_pct(top[ret_window]),    _name(top),    "#065f46","#ecfdf5"), unsafe_allow_html=True)
    k2.markdown(kpi_card("WORST",    fmt_pct(bottom[ret_window]), _name(bottom), "#991b1b","#fef2f2"), unsafe_allow_html=True)
    k3.markdown(kpi_card("AVERAGE",  fmt_pct(avg), f"{n} assets",
        "#1e3a5f" if avg >= 0 else "#7f1d1d", "#f0f9ff" if avg >= 0 else "#fff7f7"), unsafe_allow_html=True)
    k4.markdown(kpi_card("% POSITIVE", f"{n_pos/n*100:.0f}%", f"{n_pos} of {n}", "#065f46","#f0fdf4"), unsafe_allow_html=True)
    k5.markdown(kpi_card("% NEGATIVE", f"{n_neg/n*100:.0f}%", f"{n_neg} of {n}", "#991b1b","#fff1f2"), unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

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

st.dataframe(
    style_returns_table(table, ret_display),
    use_container_width=True, height=520, hide_index=True)