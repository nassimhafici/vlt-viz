import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from core.db import load_assets, load_prices
from core.formatting import (
    FONT, CATEGORY_LABELS, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID, BG2, BG3, GRAY, BLUE
)

st.title("explorer")

assets = load_assets()
if assets.empty:
    st.warning("No assets found."); st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    # Category filter
    cats_in_data = assets["category"].dropna().unique().tolist() if "category" in assets.columns else []
    ordered_cats = [c for c in CATEGORY_LABELS if c in cats_in_data]
    ordered_cats += [c for c in sorted(cats_in_data) if c not in ordered_cats]
    cat_options  = ["All"] + [CATEGORY_LABELS.get(c, c) for c in ordered_cats]
    _cat_inv     = {CATEGORY_LABELS.get(c, c): c for c in ordered_cats}

    sel_cat = st.selectbox("Category", cat_options, key="expl_cat")
    if sel_cat == "All":
        fa = assets.copy()
    else:
        fa = assets[assets["category"] == _cat_inv.get(sel_cat, sel_cat)]

    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    selected = st.selectbox("Symbol", sorted(fa["symbol"].tolist()), key="expl_sym")
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    chart_type = st.selectbox("Chart type", ["Line","Candlestick","OHLC"],
                               label_visibility="collapsed", key="expl_ct")
    period = st.select_slider(
        "Period", ["1M","3M","6M","1Y","3Y","5Y","Max"], value="1Y", key="expl_per")

# ── Asset meta ────────────────────────────────────────────────────
a     = assets[assets["symbol"] == selected].iloc[0]
price = load_prices(selected)

if price.empty:
    st.warning(f"No price data for **{selected}**."); st.stop()

price["datetime"] = pd.to_datetime(price["datetime"])
price = price.sort_values("datetime")

period_map = {"1M":21,"3M":63,"6M":126,"1Y":252,"3Y":756,"5Y":1260,"Max":99999}
price = price.tail(period_map[period])

# ── Meta strip ───────────────────────────────────────────────────
last_close  = price["close"].dropna().iloc[-1]
prev_close  = price["close"].dropna().iloc[-2] if len(price) > 1 else last_close
chg_1d      = (last_close / prev_close - 1) * 100
chg_period  = (last_close / price["close"].dropna().iloc[0] - 1) * 100

cat_display = CATEGORY_LABELS.get(a.get("category",""), a.get("category","—")) if "category" in a else "—"

m_cols = st.columns(6)
for col, lbl, val, vc in [
    (m_cols[0], "Symbol",   selected,              TEXT),
    (m_cols[1], "Name",     a["name"][:28],        TEXT_MID),
    (m_cols[2], "Category", cat_display,           TEXT_MID),
    (m_cols[3], "Last",     f"{last_close:.2f}",  TEXT),
    (m_cols[4], "1D chg",   f"{chg_1d:+.2f}%",   GREEN if chg_1d >= 0 else RED),
    (m_cols[5], f"{period} chg", f"{chg_period:+.2f}%", GREEN if chg_period >= 0 else RED),
]:
    col.markdown(
        f"<div style='padding:0 0 0 14px;border-left:1px solid {BORDER}'>"
        f"<div style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px'>{lbl}</div>"
        f"<div style='font-family:{FONT};font-size:16px;font-weight:400;color:{vc};"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{val}</div>"
        f"</div>", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Price chart ───────────────────────────────────────────────────
COLORS = {"Line": BLUE, "Candlestick": None, "OHLC": None}

fig = go.Figure()
if chart_type == "Line":
    fig.add_trace(go.Scatter(
        x=price["datetime"], y=price["close"],
        mode="lines", name=selected,
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"))
elif chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=price["datetime"],
        open=price["open"], high=price["high"],
        low=price["low"],   close=price["close"],
        name=selected,
        increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED),   fillcolor=RED)))
else:
    fig.add_trace(go.Ohlc(
        x=price["datetime"],
        open=price["open"], high=price["high"],
        low=price["low"],   close=price["close"],
        name=selected,
        increasing=dict(line=dict(color=GREEN)),
        decreasing=dict(line=dict(color=RED))))

fig.update_layout(
    paper_bgcolor=BG2, plot_bgcolor=BG2, height=340,
    font=dict(family=FONT, color=TEXT_DIM, size=10),
    margin=dict(l=54, r=16, t=16, b=44),
    hovermode="x unified",
    xaxis=dict(showgrid=False, linecolor=BORDER, zeroline=False,
               tickfont=dict(size=9, color=TEXT_DIM),
               rangeslider=dict(visible=False)),
    yaxis=dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
               zeroline=False, tickfont=dict(size=9, color=TEXT_DIM)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
)
st.plotly_chart(fig, width="stretch")

# ── Volume ────────────────────────────────────────────────────────
if "volume" in price.columns and price["volume"].notna().sum() > 5:
    vol_data = price[["datetime","volume"]].dropna()
    fig_v = go.Figure(go.Bar(
        x=vol_data["datetime"], y=vol_data["volume"],
        marker=dict(color=BG3, line=dict(width=0)),
        hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>"))
    fig_v.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2, height=110,
        margin=dict(l=54,r=16,t=4,b=36),
        xaxis=dict(showgrid=False, linecolor=BORDER,
                   tickfont=dict(size=8, color=TEXT_DIM),
                   rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=False, tickfont=dict(size=8, color=TEXT_DIM)))
    st.plotly_chart(fig_v, width="stretch")

# ── Stats strip ───────────────────────────────────────────────────
closes = price["close"].dropna()
if len(closes) > 5:
    rets = closes.pct_change().dropna()
    hi = closes.max(); lo = closes.min()
    vol_a = rets.std() * (252**0.5) * 100
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    s_cols = st.columns(6)
    for col, lbl, val in [
        (s_cols[0], f"High ({period})", f"{hi:.2f}"),
        (s_cols[1], f"Low ({period})",  f"{lo:.2f}"),
        (s_cols[2], "Vol (ann.)",       f"{vol_a:.1f}%"),
        (s_cols[3], "Avg daily ret",    f"{rets.mean()*100:+.3f}%"),
        (s_cols[4], "% days up",        f"{(rets>0).mean()*100:.1f}%"),
        (s_cols[5], "Observations",     str(len(closes))),
    ]:
        col.markdown(
            f"<div style='padding:10px 0 0 14px;border-left:1px solid {BORDER}'>"
            f"<div style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px'>{lbl}</div>"
            f"<div style='font-family:{FONT};font-size:15px;color:{TEXT};font-weight:400'>{val}</div>"
            f"</div>", unsafe_allow_html=True)