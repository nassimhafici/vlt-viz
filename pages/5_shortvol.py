"""
Short Vol — SVIX strategy monitor
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.db import load_prices
from core.formatting import (
    FONT, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

st.title("short vol")

# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
SYMS = ["SVIX", "^VIX", "^VIX3M", "^VVIX", "SVXY", "VXX", "^VIX1D", "^VIX6M"]

@st.cache_data(ttl=3600)
def _load():
    out = {}
    for s in SYMS:
        df = load_prices(s)
        if not df.empty:
            df = df.set_index("datetime").sort_index()
            df.index = pd.to_datetime(df.index)
            out[s] = df["close"].dropna()
    return out

raw = _load()

def S(sym): return raw.get(sym, pd.Series(dtype=float))

svix  = S("SVIX")
vix   = S("^VIX")
vix3m = S("^VIX3M")
vvix  = S("^VVIX")
vix1d = S("^VIX1D")
vix6m = S("^VIX6M")

if svix.empty or vix.empty:
    st.warning("SVIX or VIX data not available. Run the pipeline first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    _PERIODS = {"3M":63, "6M":126, "1Y":252, "2Y":504, "3Y":756, "5Y":1260}
    sel_p = st.selectbox("Period", list(_PERIODS.keys()), index=2)
    N = _PERIODS[sel_p]
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    show_svxy = st.checkbox("Show SVXY", value=True)
    show_vxx  = st.checkbox("Show VXX",  value=False)

# ── Windowed series ───────────────────────────────────────────────
svix_w  = svix.tail(N)
vix_w   = vix.tail(N)
vix3m_w = vix3m.tail(N)  if not vix3m.empty else pd.Series(dtype=float)
vvix_w  = vvix.tail(N)   if not vvix.empty  else pd.Series(dtype=float)

# ── Returns ───────────────────────────────────────────────────────
ret_d   = svix_w.pct_change()
vol_21  = ret_d.rolling(21).std() * np.sqrt(252) * 100
vol_63  = ret_d.rolling(63).std() * np.sqrt(252) * 100
cum_ret = (svix_w.iloc[-1] / svix_w.iloc[0] - 1) * 100

def _rolling_ret(n):
    return (svix.iloc[-1] / svix.iloc[-n-1] - 1) * 100 if len(svix) > n else np.nan

r1d = _rolling_ret(1); r1w = _rolling_ret(5)
r1m = _rolling_ret(21); r3m = _rolling_ret(63)

# ── Drawdown ──────────────────────────────────────────────────────
def _dd(s):
    s = s.ffill()
    return (s / s.expanding().max() - 1) * 100

dd_svix  = _dd(svix_w)
max_dd   = dd_svix.min()
curr_dd  = dd_svix.iloc[-1]

# ── Sharpe ────────────────────────────────────────────────────────
sharpe = (ret_d.mean() / ret_d.std() * np.sqrt(252)) if ret_d.std() > 0 else np.nan

# ── VIX term structure ────────────────────────────────────────────
_common = svix_w.index.intersection(vix_w.index).intersection(vix3m_w.index) \
          if not vix3m_w.empty else svix_w.index.intersection(vix_w.index)

if not vix3m_w.empty and len(_common) > 5:
    ratio   = vix_w.reindex(_common) / vix3m_w.reindex(_common)
    in_ctg  = ratio < 1          # True = contango, favorable
    pct_ctg = in_ctg.mean() * 100
    ratio_now = float(ratio.iloc[-1])
else:
    ratio   = pd.Series(dtype=float)
    in_ctg  = pd.Series(dtype=bool)
    pct_ctg = np.nan
    ratio_now = np.nan

regime_ok  = (not np.isnan(ratio_now)) and ratio_now < 1
regime_lbl = "Contango" if regime_ok else ("Backwardation" if not np.isnan(ratio_now) else "—")
regime_col = GREEN if regime_ok else RED

vix_now   = float(vix_w.iloc[-1])
vvix_now  = float(vvix_w.iloc[-1]) if not vvix_w.empty else np.nan

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _layout(**kw):
    b = dict(paper_bgcolor=BG2, plot_bgcolor=BG2,
             font=dict(family=FONT, color=TEXT_DIM, size=10),
             margin=dict(l=54, r=16, t=28, b=44),
             hovermode="x unified")
    b.update(kw); return b

def _xax(**kw):
    b = dict(showgrid=False, linecolor=BORDER, zeroline=False,
             tickfont=dict(size=9, color=TEXT_DIM))
    b.update(kw); return b

def _yax(**kw):
    b = dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
             zeroline=False, tickfont=dict(size=9, color=TEXT_DIM))
    b.update(kw); return b

_LEG_TOP = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM),
                orientation="h", x=0, y=1.08)
_LEG_R   = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM),
                orientation="v", x=1.02, y=1, xanchor="left")

def _sec(t, top=28):
    st.markdown(
        f"<p style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
        f"text-transform:uppercase;letter-spacing:0.12em;"
        f"margin:{top}px 0 10px 0'>{t}</p>", unsafe_allow_html=True)

def _kpi(col, label, value, sub="", vc=TEXT):
    if vc == GREEN:    bg, ac, tc = "#f0fdf4","#16a34a","#14532d"
    elif vc == RED:    bg, ac, tc = "#fef2f2","#dc2626","#7f1d1d"
    elif vc == YELLOW: bg, ac, tc = "#fffbeb","#d97706","#78350f"
    else:              bg, ac, tc = "#f8fafc","#64748b","#1e293b"
    col.markdown(
        f"<div style='background:{bg};border-radius:10px;"
        f"padding:12px 14px 10px;border-top:3px solid {ac}'>"
        f"<div style='font-size:8px;font-weight:700;letter-spacing:0.14em;"
        f"text-transform:uppercase;color:{ac};font-family:{FONT};"
        f"margin-bottom:6px'>{label}</div>"
        f"<div style='font-size:19px;font-weight:300;color:{tc};"
        f"line-height:1;font-family:{FONT}'>{value}</div>"
        f"<div style='font-size:10px;color:{ac};opacity:.7;margin-top:5px;"
        f"font-family:{FONT};white-space:nowrap;overflow:hidden;"
        f"text-overflow:ellipsis'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)

def _fmt(v, sfx="", d=2, sign=True):
    if pd.isna(v): return "—"
    s = "+" if sign and v > 0 else ""
    return f"{s}{v:.{d}f}{sfx}"

_TS = [
    {"selector":"th","props":[("background",BG3),("color",TEXT_DIM),("font-family",FONT),
     ("font-size","10px"),("font-weight","600"),("text-transform","uppercase"),
     ("letter-spacing","0.06em"),("border-bottom",f"1px solid {BORDER}"),
     ("padding","7px 12px"),("white-space","nowrap")]},
    {"selector":"td","props":[("background",BG2),("font-family",FONT),("font-size","12px"),
     ("color",TEXT),("border-bottom",f"1px solid {BORDER}"),("padding","5px 12px")]},
    {"selector":"tr:hover td","props":[("background",BG3)]},
]

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
k = st.columns(8)
_kpi(k[0], f"return {sel_p}", _fmt(cum_ret, "%"),
     vc=GREEN if cum_ret > 0 else RED)
_kpi(k[1], "return 1D", _fmt(r1d, "%"),
     vc=GREEN if not pd.isna(r1d) and r1d > 0 else RED)
_kpi(k[2], "return 1M", _fmt(r1m, "%"),
     vc=GREEN if not pd.isna(r1m) and r1m > 0 else RED)
_kpi(k[3], "max drawdown", f"{max_dd:.2f}%",
     sub=f"now {curr_dd:.2f}%",
     vc=RED if max_dd < -20 else YELLOW if max_dd < -10 else TEXT)
_kpi(k[4], "sharpe", f"{sharpe:.2f}" if not pd.isna(sharpe) else "—",
     vc=GREEN if not pd.isna(sharpe) and sharpe > 1 else
        YELLOW if not pd.isna(sharpe) and sharpe > 0 else RED)
_kpi(k[5], "VIX / VIX3M",
     f"{ratio_now:.3f}" if not np.isnan(ratio_now) else "—",
     sub=regime_lbl, vc=regime_col)
_kpi(k[6], f"% contango ({sel_p})", f"{pct_ctg:.0f}%" if not np.isnan(pct_ctg) else "—",
     sub="VIX < VIX3M",
     vc=GREEN if not np.isnan(pct_ctg) and pct_ctg > 70 else YELLOW)
_kpi(k[7], "VVIX", f"{vvix_now:.1f}" if not np.isnan(vvix_now) else "—",
     sub="vol of vol",
     vc=RED if not np.isnan(vvix_now) and vvix_now > 120
        else YELLOW if not np.isnan(vvix_now) and vvix_now > 100 else TEXT)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "Performance", "Term Structure", "Drawdowns", "Volatility", "VIX & VVIX"
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with t1:
    col_l, col_r = st.columns([3, 1], gap="large")

    with col_l:
        _sec(f"SVIX price — {sel_p}", top=8)
        fig = go.Figure()
        # Optional overlays, rebased to SVIX start value
        for sym, col_line, show in [("SVXY","#059669",show_svxy),("VXX","#dc2626",show_vxx)]:
            if show:
                s = raw.get(sym, pd.Series()).reindex(svix_w.index).dropna()
                if len(s) > 5:
                    base_svix = float(svix_w.reindex(s.index).dropna().iloc[0])
                    s_rb = s / float(s.iloc[0]) * base_svix
                    fig.add_trace(go.Scatter(
                        x=s_rb.index, y=s_rb.values, mode="lines", name=sym,
                        line=dict(color=col_line, width=1.2, dash="dot"),
                        hovertemplate=f"{sym} (rebased): %{{y:.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=svix_w.index, y=svix_w.values, mode="lines", name="SVIX",
            line=dict(color=BLUE, width=2),
            hovertemplate="SVIX: %{y:.2f}<extra></extra>"))
        fig.update_layout(**_layout(height=280, margin=dict(l=54,r=16,t=28,b=44)),
                          legend=_LEG_TOP)
        fig.update_xaxes(**_xax()); fig.update_yaxes(**_yax())
        st.plotly_chart(fig, width="stretch")

        _sec("daily returns distribution")
        ret_pct = ret_d.dropna() * 100
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=ret_pct.values, nbinsx=55, showlegend=False,
            marker=dict(color=[RED if v < 0 else GREEN for v in ret_pct.values],
                        line=dict(width=0)),
            opacity=0.82,
            hovertemplate="Ret %{x:.2f}% — %{y} days<extra></extra>"))
        fig_h.add_vline(x=0, line_color=BORDER, line_width=1.2)
        mu = float(ret_pct.mean())
        fig_h.add_vline(x=mu, line_color=BLUE, line_width=1.5, line_dash="dot",
                        annotation_text=f"avg {mu:+.2f}%",
                        annotation_position="top right",
                        annotation_font=dict(size=9, family=FONT, color=BLUE))
        fig_h.update_layout(**_layout(height=220, hovermode="x",
                                      margin=dict(l=54,r=16,t=28,b=44)))
        fig_h.update_xaxes(**_xax(ticksuffix="%"))
        fig_h.update_yaxes(**_yax(title="days"))
        st.plotly_chart(fig_h, width="stretch")

    with col_r:
        _sec("rolling returns", top=8)
        windows = [("1D",1),("1W",5),("2W",10),("1M",21),
                   ("3M",63),("6M",126),("1Y",252),("2Y",504)]
        rows = []
        for lbl, w in windows:
            if len(svix) > w:
                r = (svix.iloc[-1] / svix.iloc[-w-1] - 1) * 100
                rows.append({"Period":lbl, "Return":r})
        if rows:
            df_r = pd.DataFrame(rows)
            st.dataframe(
                df_r.style.set_table_styles(_TS)
                .applymap(lambda v: f"color:{GREEN}" if v>0 else f"color:{RED}",
                          subset=["Return"])
                .format({"Return": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"}),
                use_container_width=True, height=320, hide_index=True)

        # Stats box
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _sec("statistics")
        skew = float(ret_d.skew()) if len(ret_d) > 3 else np.nan
        kurt = float(ret_d.kurt()) if len(ret_d) > 3 else np.nan
        pct_pos = (ret_d > 0).mean() * 100
        vol_now = float(vol_21.iloc[-1]) if not vol_21.empty else np.nan
        stats = [
            ("Vol 21D", f"{vol_now:.2f}%" if not np.isnan(vol_now) else "—"),
            ("% days +", f"{pct_pos:.1f}%"),
            ("Skewness", f"{skew:.2f}" if not np.isnan(skew) else "—"),
            ("Kurtosis", f"{kurt:.2f}" if not np.isnan(kurt) else "—"),
            ("Sharpe",   f"{sharpe:.2f}" if not np.isnan(sharpe) else "—"),
            ("Max DD",   f"{max_dd:.2f}%"),
        ]
        for lbl, val in stats:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 0;border-bottom:1px solid {BORDER};"
                f"font-family:{FONT}'>"
                f"<span style='font-size:11px;color:{TEXT_DIM}'>{lbl}</span>"
                f"<span style='font-size:12px;font-weight:500;color:{TEXT}'>{val}</span>"
                f"</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — TERM STRUCTURE
# ─────────────────────────────────────────────────────────────────
with t2:
    if ratio.empty:
        st.info("VIX3M not available in DB.")
    else:
        # Row 1: ratio line + rolling % contango
        ca, cb = st.columns(2, gap="medium")

        with ca:
            _sec("VIX / VIX3M ratio", top=8)
            fig_r = go.Figure()
            # Contango fill (below 1)
            fig_r.add_trace(go.Scatter(
                x=ratio.index.tolist() + ratio.index.tolist()[::-1],
                y=np.minimum(ratio.values, 1).tolist() + [1]*len(ratio),
                fill="toself", fillcolor="rgba(5,150,105,0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            # Backwardation fill (above 1)
            fig_r.add_trace(go.Scatter(
                x=ratio.index.tolist() + ratio.index.tolist()[::-1],
                y=np.maximum(ratio.values, 1).tolist() + [1]*len(ratio),
                fill="toself", fillcolor="rgba(220,38,38,0.08)",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_r.add_hline(y=1.0, line_color=BORDER, line_width=1.5)
            fig_r.add_trace(go.Scatter(
                x=ratio.index, y=ratio.values, mode="lines", name="VIX/VIX3M",
                line=dict(color=TEXT_MID, width=1.8),
                hovertemplate="Ratio: %{y:.3f}<extra></extra>"))
            # Current level annotation
            fig_r.add_trace(go.Scatter(
                x=[ratio.index[-1]], y=[ratio_now],
                mode="markers+text",
                marker=dict(color=regime_col, size=7),
                text=[f"{ratio_now:.3f}"],
                textposition="top right",
                textfont=dict(size=10, family=FONT, color=regime_col),
                showlegend=False, hoverinfo="skip"))
            fig_r.update_layout(**_layout(height=280, margin=dict(l=54,r=16,t=28,b=44)))
            fig_r.update_xaxes(**_xax()); fig_r.update_yaxes(**_yax())
            st.plotly_chart(fig_r, width="stretch")

        with cb:
            _sec(f"rolling 63D % in contango", top=8)
            roll_ctg = in_ctg.rolling(63).mean() * 100
            fig_ctg = go.Figure()
            fig_ctg.add_hrect(y0=70, y1=100,
                              fillcolor="rgba(5,150,105,0.07)", line_width=0)
            fig_ctg.add_hrect(y0=0, y1=50,
                              fillcolor="rgba(220,38,38,0.05)", line_width=0)
            fig_ctg.add_hline(y=70, line_color="#059669", line_width=1, line_dash="dot")
            fig_ctg.add_hline(y=50, line_color=BORDER, line_width=1)
            fig_ctg.add_trace(go.Scatter(
                x=roll_ctg.index, y=roll_ctg.values,
                mode="lines", name="% contango (63D)",
                fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                line=dict(color=BLUE, width=1.8),
                hovertemplate="%{y:.1f}% of days in contango<extra></extra>"))
            fig_ctg.update_layout(**_layout(height=280, margin=dict(l=54,r=16,t=28,b=44)))
            fig_ctg.update_xaxes(**_xax())
            fig_ctg.update_yaxes(**_yax(ticksuffix="%", range=[0,105]))
            st.plotly_chart(fig_ctg, width="stretch")

        # VIX vs VIX3M levels + regime rug
        _sec("VIX vs VIX3M — absolute levels")
        fig_lvl = go.Figure()
        fig_lvl.add_trace(go.Scatter(
            x=vix_w.reindex(_common).index,
            y=vix_w.reindex(_common).values,
            mode="lines", name="VIX",
            line=dict(color=RED, width=1.6),
            hovertemplate="VIX: %{y:.2f}<extra></extra>"))
        fig_lvl.add_trace(go.Scatter(
            x=vix3m_w.reindex(_common).index,
            y=vix3m_w.reindex(_common).values,
            mode="lines", name="VIX3M",
            line=dict(color=BLUE, width=1.4, dash="dot"),
            hovertemplate="VIX3M: %{y:.2f}<extra></extra>"))
        # Fill between: red when VIX > VIX3M (backwardation)
        v_arr  = vix_w.reindex(_common).values
        v3_arr = vix3m_w.reindex(_common).values
        x_arr  = list(_common)
        fig_lvl.add_trace(go.Scatter(
            x=x_arr + x_arr[::-1],
            y=np.maximum(v_arr, v3_arr).tolist() + np.minimum(v_arr, v3_arr).tolist()[::-1],
            fill="toself",
            fillcolor=[
                "rgba(220,38,38,0.10)" if vi > v3i else "rgba(5,150,105,0.08)"
                for vi, v3i in zip(v_arr, v3_arr)
            ][0] if len(v_arr) else "rgba(0,0,0,0)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_lvl.update_layout(**_layout(height=260, margin=dict(l=54,r=16,t=28,b=44)),
                              legend=_LEG_TOP)
        fig_lvl.update_xaxes(**_xax()); fig_lvl.update_yaxes(**_yax())
        st.plotly_chart(fig_lvl, width="stretch")

        # Regime rug strip
        _sec("daily regime — contango (green) vs backwardation (red)")
        ctg_days = in_ctg.index[in_ctg]
        bwd_days = in_ctg.index[~in_ctg]
        fig_rug = go.Figure()
        if len(ctg_days):
            fig_rug.add_trace(go.Scatter(
                x=ctg_days, y=[1]*len(ctg_days), mode="markers",
                marker=dict(color=GREEN, size=3, symbol="square"),
                name="Contango", hoverinfo="skip"))
        if len(bwd_days):
            fig_rug.add_trace(go.Scatter(
                x=bwd_days, y=[1]*len(bwd_days), mode="markers",
                marker=dict(color=RED, size=3, symbol="square"),
                name="Backwardation", hoverinfo="skip"))
        fig_rug.update_layout(**_layout(height=80, hovermode=False,
                                        margin=dict(l=54,r=16,t=8,b=28)),
                              legend=_LEG_TOP,
                              yaxis=dict(showticklabels=False, showgrid=False,
                                         zeroline=False))
        fig_rug.update_xaxes(**_xax())
        st.plotly_chart(fig_rug, width="stretch")

        # Stats: days in each regime
        n_ctg = int(in_ctg.sum()); n_bwd = int((~in_ctg).sum())
        c1,c2,c3,c4 = st.columns(4)
        for col, lbl, val, vc in [
            (c1,"days contango",   f"{n_ctg}",      GREEN),
            (c2,"days backwardation",f"{n_bwd}",    RED),
            (c3,"% contango",      f"{pct_ctg:.1f}%", GREEN if pct_ctg>70 else YELLOW),
            (c4,"current",         regime_lbl,      regime_col),
        ]:
            _kpi(col, lbl, val, vc=vc)


# ─────────────────────────────────────────────────────────────────
# TAB 3 — DRAWDOWNS
# ─────────────────────────────────────────────────────────────────
with t3:
    dl, dr = st.columns([3,1], gap="large")

    with dl:
        _sec(f"SVIX underwater curve — {sel_p}", top=8)
        fig_dd = go.Figure()
        fig_dd.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_dd.add_trace(go.Scatter(
            x=dd_svix.index, y=dd_svix.values,
            mode="lines", name="DD",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.10)",
            line=dict(color=RED, width=1.6),
            hovertemplate="DD: %{y:.2f}%<extra></extra>"))
        # Mark max DD
        worst_dt = dd_svix.idxmin()
        fig_dd.add_trace(go.Scatter(
            x=[worst_dt], y=[max_dd],
            mode="markers+text",
            marker=dict(color=RED, size=8, symbol="circle"),
            text=[f"{max_dd:.1f}%"],
            textposition="bottom center",
            textfont=dict(size=9, family=FONT, color=RED),
            showlegend=False))
        fig_dd.update_layout(**_layout(height=260, margin=dict(l=54,r=16,t=28,b=44)))
        fig_dd.update_xaxes(**_xax()); fig_dd.update_yaxes(**_yax(ticksuffix="%"))
        st.plotly_chart(fig_dd, width="stretch")

        _sec("drawdown vs VIX — spikes explain losses")
        fig_jt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_jt.add_trace(go.Scatter(
            x=dd_svix.index, y=dd_svix.values,
            mode="lines", name="SVIX DD",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            line=dict(color=RED, width=1.4),
            hovertemplate="DD: %{y:.2f}%<extra></extra>"), secondary_y=False)
        vix_al = vix.reindex(dd_svix.index)
        fig_jt.add_trace(go.Scatter(
            x=vix_al.index, y=vix_al.values,
            mode="lines", name="VIX",
            line=dict(color=YELLOW, width=1.3, dash="dot"),
            hovertemplate="VIX: %{y:.2f}<extra></extra>"), secondary_y=True)
        fig_jt.update_layout(**_layout(height=240, margin=dict(l=54,r=54,t=28,b=44)),
                             legend=_LEG_TOP)
        fig_jt.update_xaxes(**_xax())
        fig_jt.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="#f3f4f6",
                            tickfont=dict(size=9,color=RED), secondary_y=False)
        fig_jt.update_yaxes(showgrid=False, tickfont=dict(size=9,color=YELLOW),
                            secondary_y=True)
        st.plotly_chart(fig_jt, width="stretch")

    with dr:
        _sec("worst days", top=8)
        worst = (ret_d.dropna()*100).nsmallest(12).reset_index()
        worst.columns = ["Date","Return"]
        worst["Date"] = worst["Date"].dt.strftime("%Y-%m-%d")
        worst["VIX"]  = worst["Date"].apply(
            lambda d: float(vix.get(pd.Timestamp(d), np.nan))
            if not vix.empty else np.nan)
        st.dataframe(
            worst.style.set_table_styles(_TS)
            .applymap(lambda v: f"color:{RED};font-weight:600", subset=["Return"])
            .format({"Return": "{:.2f}%",
                     "VIX": lambda x: f"{x:.1f}" if pd.notna(x) else "—"}),
            use_container_width=True, height=460, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# TAB 4 — VOLATILITY
# ─────────────────────────────────────────────────────────────────
with t4:
    _sec("SVIX realized volatility — 21D vs 63D annualized", top=8)
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=vol_63.index, y=vol_63.values,
        mode="lines", name="Vol 63D",
        fill="tozeroy", fillcolor="rgba(209,213,219,0.35)",
        line=dict(color=GRAY, width=1.2),
        hovertemplate="Vol 63D: %{y:.2f}%<extra></extra>"))
    fig_v.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values,
        mode="lines", name="Vol 21D",
        line=dict(color=BLUE, width=1.8),
        hovertemplate="Vol 21D: %{y:.2f}%<extra></extra>"))
    # VIX on secondary axis
    vix_ral = vix.reindex(vol_21.index)
    fig_v_joint = make_subplots(specs=[[{"secondary_y": True}]])
    fig_v_joint.add_trace(go.Scatter(
        x=vol_63.index, y=vol_63.values, mode="lines", name="Vol 63D",
        fill="tozeroy", fillcolor="rgba(209,213,219,0.35)",
        line=dict(color=GRAY, width=1.2),
        hovertemplate="Vol 63D: %{y:.2f}%<extra></extra>"), secondary_y=False)
    fig_v_joint.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values, mode="lines", name="Vol 21D",
        line=dict(color=BLUE, width=1.8),
        hovertemplate="Vol 21D: %{y:.2f}%<extra></extra>"), secondary_y=False)
    fig_v_joint.add_trace(go.Scatter(
        x=vix_ral.index, y=vix_ral.values, mode="lines", name="VIX",
        line=dict(color=RED, width=1.1, dash="dot"),
        hovertemplate="VIX: %{y:.2f}<extra></extra>"), secondary_y=True)
    fig_v_joint.update_layout(**_layout(height=300, margin=dict(l=54,r=54,t=28,b=44)),
                              legend=_LEG_TOP)
    fig_v_joint.update_xaxes(**_xax())
    fig_v_joint.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="#f3f4f6",
                             tickfont=dict(size=9,color=BLUE), secondary_y=False)
    fig_v_joint.update_yaxes(showgrid=False, tickfont=dict(size=9,color=RED),
                             secondary_y=True)
    st.plotly_chart(fig_v_joint, width="stretch")

    # Vol scatter vs VIX
    _sec("SVIX realized vol 21D vs VIX — how risky is each vol level?")
    vc_df = pd.concat([vol_21.rename("v"), vix.rename("vix")], axis=1).dropna()
    vc_df["color"] = vc_df["vix"]
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=vc_df["vix"], y=vc_df["v"], mode="markers",
        marker=dict(color=vc_df["vix"],
                    colorscale=[[0,"#f0fdf4"],[0.45,"#fef3c7"],[1,"#fef2f2"]],
                    size=4, opacity=0.65,
                    colorbar=dict(title="VIX", thickness=10,
                                  tickfont=dict(size=8,family=FONT,color=TEXT_DIM))),
        hovertemplate="VIX: %{x:.1f} → SVIX vol 21D: %{y:.1f}%<extra></extra>",
        showlegend=False))
    fig_sc.update_layout(**_layout(height=280, hovermode="closest",
                                   margin=dict(l=54,r=80,t=28,b=52)))
    fig_sc.update_xaxes(**_xax(showgrid=True, title="VIX level"))
    fig_sc.update_yaxes(**_yax(ticksuffix="%", title="Realized vol 21D"))
    st.plotly_chart(fig_sc, width="stretch")


# ─────────────────────────────────────────────────────────────────
# TAB 5 — VIX & VVIX
# ─────────────────────────────────────────────────────────────────
with t5:
    vl, vr = st.columns(2, gap="medium")

    with vl:
        _sec("VIX — spot level", top=8)
        fig_vix = go.Figure()
        fig_vix.add_hrect(y0=0,  y1=15, fillcolor="rgba(5,150,105,0.06)",  line_width=0)
        fig_vix.add_hrect(y0=20, y1=30, fillcolor="rgba(217,119,6,0.05)",  line_width=0)
        fig_vix.add_hrect(y0=30, y1=90, fillcolor="rgba(220,38,38,0.05)",  line_width=0)
        for lvl, col_l in [(15, GRAY),(20, YELLOW),(30, RED)]:
            fig_vix.add_hline(y=lvl, line_color=col_l, line_width=1, line_dash="dot")
        fig_vix.add_trace(go.Scatter(
            x=vix_w.index, y=vix_w.values,
            mode="lines", name="VIX",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            line=dict(color=RED, width=1.7),
            hovertemplate="VIX: %{y:.2f}<extra></extra>"))
        # Annotation: current level
        fig_vix.add_annotation(
            x=vix_w.index[-1], y=vix_now,
            text=f"  {vix_now:.1f}", showarrow=False,
            font=dict(size=11, family=FONT,
                      color=RED if vix_now>20 else YELLOW if vix_now>15 else GREEN),
            xanchor="left")
        fig_vix.update_layout(**_layout(height=280, margin=dict(l=54,r=16,t=28,b=44)))
        fig_vix.update_xaxes(**_xax()); fig_vix.update_yaxes(**_yax())
        st.plotly_chart(fig_vix, width="stretch")

        # VIX distribution
        _sec("VIX distribution — where do we stand?")
        p10 = float(np.percentile(vix_w.dropna(),10))
        p90 = float(np.percentile(vix_w.dropna(),90))
        fig_vd = go.Figure()
        fig_vd.add_trace(go.Histogram(
            x=vix_w.values, nbinsx=45, showlegend=False,
            marker=dict(color=RED, opacity=0.55, line=dict(width=0)),
            hovertemplate="VIX %{x:.1f}: %{y} days<extra></extra>"))
        fig_vd.add_vline(x=vix_now, line_color=BLUE, line_width=2,
                         annotation_text=f"now {vix_now:.1f}",
                         annotation_font=dict(size=9, family=FONT, color=BLUE))
        fig_vd.add_vline(x=p10, line_color=GREEN, line_width=1, line_dash="dot",
                         annotation_text="p10",
                         annotation_font=dict(size=8, family=FONT, color=GREEN))
        fig_vd.add_vline(x=p90, line_color=RED, line_width=1, line_dash="dot",
                         annotation_text="p90",
                         annotation_font=dict(size=8, family=FONT, color=RED))
        fig_vd.update_layout(**_layout(height=220, hovermode="x",
                                       margin=dict(l=54,r=16,t=28,b=44)))
        fig_vd.update_xaxes(**_xax()); fig_vd.update_yaxes(**_yax())
        st.plotly_chart(fig_vd, width="stretch")

    with vr:
        _sec("VVIX — vol of vol", top=8)
        if vvix_w.empty:
            st.info("VVIX not available.")
        else:
            fig_vv = go.Figure()
            fig_vv.add_hline(y=100, line_color=YELLOW, line_width=1, line_dash="dot",
                             annotation_text="100",
                             annotation_font=dict(size=9,family=FONT,color=YELLOW))
            fig_vv.add_hline(y=120, line_color=RED, line_width=1, line_dash="dot",
                             annotation_text="120",
                             annotation_font=dict(size=9,family=FONT,color=RED))
            fig_vv.add_trace(go.Scatter(
                x=vvix_w.index, y=vvix_w.values,
                mode="lines", name="VVIX",
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
                line=dict(color="#7c3aed", width=1.7),
                hovertemplate="VVIX: %{y:.2f}<extra></extra>"))
            fig_vv.add_annotation(
                x=vvix_w.index[-1], y=vvix_now,
                text=f"  {vvix_now:.1f}", showarrow=False,
                font=dict(size=11,family=FONT,
                          color=RED if vvix_now>120 else YELLOW if vvix_now>100 else TEXT),
                xanchor="left")
            fig_vv.update_layout(**_layout(height=280, margin=dict(l=54,r=16,t=28,b=44)))
            fig_vv.update_xaxes(**_xax()); fig_vv.update_yaxes(**_yax())
            st.plotly_chart(fig_vv, width="stretch")

            # VVIX vs VIX scatter — regime quadrant
            _sec("VVIX vs VIX — regime quadrant")
            q_df = pd.concat([vix.rename("vix"), vvix.rename("vvix")], axis=1).dropna()
            # Color: last 63 days orange, rest gray
            is_recent = (q_df.index >= q_df.index[-63]) if len(q_df)>63 else pd.Series(True,index=q_df.index)
            fig_q = go.Figure()
            fig_q.add_hline(y=100, line_color=BORDER, line_width=1, line_dash="dot")
            fig_q.add_vline(x=20,  line_color=BORDER, line_width=1, line_dash="dot")
            fig_q.add_trace(go.Scatter(
                x=q_df["vix"], y=q_df["vvix"], mode="markers",
                marker=dict(color=GRAY, size=3, opacity=0.35),
                name="history", hoverinfo="skip"))
            if len(q_df) > 63:
                rc = q_df[q_df.index >= q_df.index[-63]]
                fig_q.add_trace(go.Scatter(
                    x=rc["vix"], y=rc["vvix"], mode="markers",
                    marker=dict(color=YELLOW, size=4, opacity=0.75),
                    name="last 63D",
                    hovertemplate="VIX: %{x:.1f} VVIX: %{y:.1f}<extra></extra>"))
            # Current point
            fig_q.add_trace(go.Scatter(
                x=[vix_now], y=[vvix_now], mode="markers+text",
                marker=dict(color=RED, size=9),
                text=[" now"], textposition="top right",
                textfont=dict(size=10,family=FONT,color=RED),
                name="now",
                hovertemplate=f"VIX:{vix_now:.1f} VVIX:{vvix_now:.1f}<extra></extra>"))
            # Quadrant labels
            for txt, ax, ay in [
                ("Low VIX\nLow VVIX", 12, 85),
                ("High VIX\nLow VVIX", 35, 85),
                ("Low VIX\nHigh VVIX", 12, 130),
                ("High VIX\nHigh VVIX", 35, 130),
            ]:
                fig_q.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                    font=dict(size=8,family=FONT,color="#d1d5db"), align="center")
            fig_q.update_layout(**_layout(height=280, hovermode="closest",
                                          margin=dict(l=54,r=16,t=28,b=52)),
                                legend=_LEG_TOP)
            fig_q.update_xaxes(**_xax(showgrid=True, title="VIX"))
            fig_q.update_yaxes(**_yax(title="VVIX"))
            st.plotly_chart(fig_q, width="stretch")