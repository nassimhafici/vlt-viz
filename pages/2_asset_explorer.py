import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from core.db import load_assets, load_prices
from core.formatting import (
    kpi_card,
    FONT, CATEGORY_LABELS, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

# ── Palette multi-lignes ──────────────────────────────────────────
LINE_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#ec4899",  # pink
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
]

assets = load_assets()
if assets.empty:
    st.warning("No assets found."); st.stop()

all_symbols = sorted(assets["symbol"].tolist())

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    cats_in_data = assets["category"].dropna().unique().tolist() if "category" in assets.columns else []
    ordered_cats = [c for c in CATEGORY_LABELS if c in cats_in_data]
    ordered_cats += [c for c in sorted(cats_in_data) if c not in ordered_cats]
    cat_options  = ["All"] + [CATEGORY_LABELS.get(c, c) for c in ordered_cats]
    _cat_inv     = {CATEGORY_LABELS.get(c, c): c for c in ordered_cats}

    sel_cat = st.selectbox("Category", cat_options, key="cmp_cat")
    if sel_cat == "All":
        fa = assets.copy()
    else:
        fa = assets[assets["category"] == _cat_inv.get(sel_cat, sel_cat)]

    available = sorted(fa["symbol"].tolist())

    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)

    selected_syms = st.multiselect(
        "Symbols",
        available,
        default=["SVIX", "VEQT.TO", "SPY"],
        key="cmp_syms",
        max_selections=10,
    )

    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)

    period = st.select_slider(
        "Period", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
        value="1Y", key="cmp_per"
    )

    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)

    show_vol   = st.checkbox("Show volatility bands (±1σ)", value=False, key="cmp_vol")
    show_dd    = st.checkbox("Show drawdown", value=True, key="cmp_dd")

if not selected_syms:
    st.info("Select at least one symbol in the sidebar.")
    st.stop()

# ── Load & align prices ───────────────────────────────────────────
period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": 99999}
n_days = period_map[period]

price_dict = {}
for sym in selected_syms:
    p = load_prices(sym)
    if p.empty:
        continue
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p.sort_values("datetime").set_index("datetime")
    price_dict[sym] = p["close"].dropna()

if not price_dict:
    st.warning("No price data available for selected symbols.")
    st.stop()

# Align on common dates, trim to period
closes = pd.DataFrame(price_dict)
closes = closes.sort_index()

# Keep only last n_days rows
closes = closes.tail(n_days)

# Common start date = first date where ALL selected symbols have data
common_start = closes.dropna(how="any").index.min()
if pd.isna(common_start):
    # fallback: use pairwise available data
    common_start = closes.index.min()

closes_aligned = closes[closes.index >= common_start]

last_date = closes_aligned.index.max().strftime("%Y-%m-%d")
start_date = closes_aligned.index.min().strftime("%Y-%m-%d")

# ── Normalize to 100 ─────────────────────────────────────────────
base = closes_aligned.iloc[0]
normalized = (closes_aligned / base * 100).round(4)

# ── Title ─────────────────────────────────────────────────────────
st.title(f"compare assets · normalized to 100 · {start_date} → {last_date}")

# ── KPI strip — final normalized value per symbol ─────────────────
cols = st.columns(len(selected_syms))
for i, sym in enumerate(selected_syms):
    if sym not in normalized.columns:
        continue
    series = normalized[sym].dropna()
    if series.empty:
        continue
    final_val  = series.iloc[-1]
    total_ret  = final_val - 100
    vc = "green" if total_ret >= 0 else "red"
    kpi_card(cols[i], sym, f"{final_val:.1f}", sub=f"{total_ret:+.1f} pts", vc=vc)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Main normalized chart ─────────────────────────────────────────
fig = go.Figure()

for i, sym in enumerate(selected_syms):
    if sym not in normalized.columns:
        continue
    series = normalized[sym].dropna()
    color  = LINE_COLORS[i % len(LINE_COLORS)]

    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=sym,
        line=dict(color=color, width=1.8),
        hovertemplate=f"<b>{sym}</b> %{{x|%Y-%m-%d}}: %{{y:.1f}}<extra></extra>",
    ))

    # Optional: ±1σ rolling volatility band (21d window)
    if show_vol and len(series) > 21:
        roll_std = series.rolling(21).std()
        upper = series + roll_std
        lower = series - roll_std
        fig.add_trace(go.Scatter(
            x=pd.concat([series.index.to_series(), series.index.to_series()[::-1]]).tolist(),
            y=pd.concat([upper, lower[::-1]]).tolist(),
            fill="toself",

            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"{sym} ±1σ",
        ))

# Baseline at 100
fig.add_hline(y=100, line=dict(color=BORDER, width=1, dash="dot"),
              annotation_text="base 100", annotation_font_size=8,
              annotation_font_color=TEXT_DIM)

fig.update_layout(
    paper_bgcolor=BG2, plot_bgcolor=BG2,
    height=400,
    font=dict(family=FONT, color=TEXT_DIM, size=10),
    margin=dict(l=54, r=16, t=16, b=44),
    hovermode="x unified",
    xaxis=dict(
        showgrid=False, linecolor=BORDER, zeroline=False,
        tickfont=dict(size=9, color=TEXT_DIM),
        rangeslider=dict(visible=False),
        rangebreaks=[dict(bounds=["sat", "mon"])],
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
        zeroline=False, tickfont=dict(size=9, color=TEXT_DIM),
        ticksuffix="",
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(size=9), orientation="h",
        yanchor="bottom", y=1.01, xanchor="left", x=0,
    ),
)
st.plotly_chart(fig, use_container_width=True)

# ── Drawdown chart ────────────────────────────────────────────────
if show_dd:
    fig_dd = go.Figure()

    for i, sym in enumerate(selected_syms):
        if sym not in normalized.columns:
            continue
        series = normalized[sym].dropna()
        color  = LINE_COLORS[i % len(LINE_COLORS)]
        roll_max = series.cummax()
        dd = (series / roll_max - 1) * 100

        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=sym,
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            #fillcolor=color + "18",
            hovertemplate=f"<b>{sym}</b> %{{x|%Y-%m-%d}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig_dd.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2, height=160,
        font=dict(family=FONT, color=TEXT_DIM, size=10),
        margin=dict(l=54, r=16, t=8, b=36),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False, linecolor=BORDER,
            tickfont=dict(size=8, color=TEXT_DIM),
            rangeslider=dict(visible=False),
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f3f4f6",
            tickfont=dict(size=8, color=TEXT_DIM),
            ticksuffix="%",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
        title=dict(text="Drawdown from peak (%)", font=dict(size=10, color=TEXT_DIM), x=0),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Stats table ───────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

rows = []
for sym in selected_syms:
    if sym not in closes_aligned.columns:
        continue
    raw = closes_aligned[sym].dropna()
    if len(raw) < 2:
        continue
    rets   = raw.pct_change().dropna()
    norm_s = normalized[sym].dropna()
    total  = norm_s.iloc[-1] - 100 if len(norm_s) else np.nan
    vol_a  = rets.std() * (252 ** 0.5) * 100
    sharpe = (rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else np.nan
    roll_max = norm_s.cummax()
    max_dd = ((norm_s / roll_max - 1) * 100).min()
    pct_up = (rets > 0).mean() * 100

    name = assets.loc[assets["symbol"] == sym, "name"].values
    name = name[0][:24] if len(name) else sym

    rows.append({
        "Symbol":      sym,
        "Name":        name,
        f"Ret ({period})": f"{total:+.1f} pts",
        "Vol (ann.)":  f"{vol_a:.1f}%",
        "Sharpe":      f"{sharpe:.2f}" if not np.isnan(sharpe) else "—",
        "Max DD":      f"{max_dd:.1f}%",
        "% days ↑":    f"{pct_up:.1f}%",
    })

if rows:
    df_stats = pd.DataFrame(rows).set_index("Symbol")
    st.dataframe(df_stats, use_container_width=True)