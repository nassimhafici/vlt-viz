"""
Rates — Fixed Income / FRED data only
KPIs: SOFR, 3M, 1Y, 10Y, 30Y current levels + 1D change in bps
Yield curve: current / 1 month ago / 1 year ago
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.db import load_returns_with_meta, load_prices_by_symbol, load_assets
from core.formatting import (
    fmt_price, FONT,
    GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

st.title("rates")


# ── Integer bps formatter ─────────────────────────────────────────
def fmt_bps_int(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    rounded = int(round(val))
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded}bp"


# ── Data ─────────────────────────────────────────────────────────
assets  = load_assets()
fi      = assets[assets["asset_class"] == "Fixed Income"].copy()

if fi.empty:
    st.warning("No Fixed Income assets found.")
    st.stop()

returns = load_returns_with_meta()
returns = returns[returns["symbol"].isin(fi["symbol"])]

all_fi_syms = fi["symbol"].tolist()
prices_map  = load_prices_by_symbol(all_fi_syms)


# ── Helper: get close at date offset ─────────────────────────────
def get_close_at(symbol: str, offset_days: int = 0) -> float | None:
    df = prices_map.get(symbol, pd.DataFrame())
    if df.empty or "close" not in df.columns:
        return None

    closes = df["close"].dropna()

    if offset_days == 0:
        return float(closes.iloc[-1]) if len(closes) >= 1 else None

    idx = len(closes) + offset_days
    if idx < 0:
        return None

    return float(closes.iloc[idx]) if idx < len(closes) else None


# ── KPI strip ─────────────────────────────────────────────────────
KPI_SYMBOLS = {
    "SOFR":  "SOFR",
    "3M":    "DGS3MO",
    "1Y":    "DGS1",
    "10Y":   "DGS10",
    "30Y":   "DGS30",
}


def kpi_card(col, label, symbol):

    last = get_close_at(symbol, 0)
    prev = get_close_at(symbol, -1)

    if last is None:
        col.markdown(
            f"<div style='padding:0 0 0 14px;border-left:1px solid {BORDER}'>"
            f"<div style='font-family:{FONT};font-size:10px;color:{TEXT_DIM};"
            f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px'>{label}</div>"
            f"<div style='font-family:{FONT};font-size:22px;font-weight:300;color:{GRAY}'>—</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        return

    chg = (last - prev) * 100 if prev is not None else None

    sub_html = ""

    if chg is not None:
        rounded = int(round(chg))
        sign = "+" if rounded > 0 else ""
        sc = RED if rounded > 0 else GREEN if rounded < 0 else TEXT_DIM

        sub_html = (
            f"<div style='font-family:{FONT};font-size:11px;color:{sc};"
            f"margin-top:4px'>{sign}{rounded} bp</div>"
        )

    col.markdown(
        f"<div style='padding:0 0 0 14px;border-left:1px solid {BORDER}'>"
        f"<div style='font-family:{FONT};font-size:10px;color:{TEXT_DIM};"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px'>{label}</div>"
        f"<div style='font-family:{FONT};font-size:22px;font-weight:300;color:{TEXT}'>{last:.2f}%</div>"
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True
    )


cols = st.columns(len(KPI_SYMBOLS))

for col, (label, sym) in zip(cols, KPI_SYMBOLS.items()):
    kpi_card(col, label, sym)

st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)


# ── Yield curve ───────────────────────────────────────────────────
MATURITY_ORDER = {
    "DGS1MO": 1/12,
    "DGS3MO": 3/12,
    "DGS6MO": 6/12,
    "DGS1": 1,
    "DGS2": 2,
    "DGS5": 5,
    "DGS7": 7,
    "DGS10": 10,
    "DGS20": 20,
    "DGS30": 30,
}

curve_syms = [s for s in MATURITY_ORDER if s in fi["symbol"].values]

# helper pour l'affichage
def label_x(m):
    return f"{int(m*12)}M" if m < 1 else f"{int(m)}Y"

# compression long terme pour rapprocher 10Y et 30Y
def compress_maturity(m):
    if m < 10:
        return m
    return 10 + (m - 10) * 0.4  # 10Y→10, 20Y→14, 30Y→18 environ

if curve_syms:
    st.markdown(
        f"<p style='font-family:{FONT};font-size:12px;color:{TEXT_DIM};"
        f"text-transform:uppercase;letter-spacing:0.12em;margin:0 0 10px 0'>"
        f"Yield curve</p>",
        unsafe_allow_html=True
    )

    snapshots = {
        "Today":   (BLUE, "solid", 2.0, 0),
        "1M ago":  (YELLOW, "dot", 1.2, -21),
        "1Y ago":  (GRAY, "dot", 1.0, -252),
    }

    fig = go.Figure()

    for snap_label, (color, dash, width, offset) in snapshots.items():
        xs, ys, labels = [], [], []

        for sym in curve_syms:
            v = get_close_at(sym, offset)
            if v is None:
                continue
            xs.append(compress_maturity(MATURITY_ORDER[sym]))
            ys.append(v)
            labels.append(label_x(MATURITY_ORDER[sym]))

        if not xs:
            continue

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            text=labels,
            mode="lines+markers" if snap_label == "Today" else "lines",
            name=snap_label,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=5, color=color) if snap_label == "Today" else dict(size=0),
            hovertemplate=f"{snap_label} — %{{text}}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=BG2,
        plot_bgcolor=BG2,
        height=280,
        font=dict(family=FONT, color=TEXT_MID, size=10),
        margin=dict(l=52, r=16, t=16, b=44),
        xaxis=dict(
            showgrid=False,
            linecolor="#e5e7eb",
            tickvals=[compress_maturity(MATURITY_ORDER[s]) for s in curve_syms],
            ticktext=[label_x(MATURITY_ORDER[s]) for s in curve_syms],
            tickfont=dict(size=9, color=TEXT_DIM)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f3f4f6",
            gridwidth=0.5,
            ticksuffix="%",
            tickfont=dict(size=9, color=TEXT_DIM)
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Full table ───────────────────────────────────────────────────
st.markdown(
    f"<p style='font-family:{FONT};font-size:12px;color:{TEXT_DIM};"
    f"text-transform:uppercase;letter-spacing:0.12em;margin:32px 0 10px 0'>"
    f"Levels and changes</p>",
    unsafe_allow_html=True
)

if not returns.empty:
    disp = returns[["symbol","name","r1d","r1w","r1m","r3m","rytd","r1y"]].copy()
    disp["last"] = disp["symbol"].apply(lambda s: get_close_at(s, 0))
    disp = disp[["symbol","name","last","r1d","r1w","r1m","r3m","rytd","r1y"]]

    # ── force integer bps ──
    bps_cols = ["r1d","r1w","r1m","r3m","rytd","r1y"]
    disp[bps_cols] = disp[bps_cols].round().astype("Int64")

    def color_bps(v):
        if pd.isna(v):
            return f"color:{GRAY}"
        return f"color:{RED}" if v > 0 else f"color:{GREEN}"

    styled = (
        disp.rename(columns={
            "symbol":"SYMBOL",
            "name":"NAME",
            "last":"LAST (%)",
            "r1d":"1D",
            "r1w":"1W",
            "r1m":"1M",
            "r3m":"3M",
            "rytd":"YTD",
            "r1y":"1Y"
        }).style
        .applymap(color_bps, subset=["1D","1W","1M","3M","YTD","1Y"])
        .format({c: (lambda x: "—" if pd.isna(x) else f"{'+' if x>0 else ''}{x}bp")
                 for c in ["1D","1W","1M","3M","YTD","1Y"]})
        .format({"LAST (%)": lambda x: f"{x:.2f}%" if pd.notna(x) else "—"})
        .set_table_styles([
            {"selector":"th","props":[
                ("background",BG),
                ("color",TEXT_DIM),
                ("font-family",FONT),
                ("font-size","11px"),
                ("text-transform","uppercase"),
                ("border-bottom",f"1px solid {BORDER}")
            ]},
            {"selector":"td","props":[
                ("background",BG2),
                ("font-family",FONT),
                ("font-size","12px"),
                ("color",TEXT),
                ("border-bottom",f"1px solid {BORDER}")
            ]},
            {"selector":"tr:hover td","props":[("background",BG3)]},
        ])
    )

    st.dataframe(styled, use_container_width=True, height=480, hide_index=True)