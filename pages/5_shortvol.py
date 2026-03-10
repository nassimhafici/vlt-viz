"""
Short Vol — SVIX strategy monitor
3 strategies: SVIX (VXX proxy), VIX/VIX3M ratio carry, Entropy filter
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.stats import percentileofscore
except ImportError:
    def percentileofscore(a, score):
        a = np.asarray(a)
        return float((a <= score).mean() * 100)

from core.db import load_prices
from core.formatting import (
    kpi_card,
    FONT, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

st.title("shortvol strategy")

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
_LEG_TOP = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM),
                orientation="h", x=0, y=1.08)

_C_SVIX    = BLUE
_C_RATIO   = "#d97706"
_C_ENTROPY = "#7c3aed"
_C_VXX     = "#dc2626"
_C_SVXY    = "#059669"

# FIX — Term structure: distinct colors for current vs historical snapshots
_C_TS_NOW  = BLUE
_C_TS_1D   = "#f59e0b"
_C_TS_1W   = "#545f6e"
_C_TS_1M   = "#94a3b8"

def _layout(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG2,
        font=dict(family=FONT, size=11, color=TEXT),
        hovermode="x unified",
        margin=dict(l=48, r=16, t=28, b=44),
        height=300,
    )
    base.update(kw)
    return base

def _xax(**kw):
    base = dict(showgrid=False, linecolor=BORDER,
                tickfont=dict(size=9, color=TEXT_DIM),
                rangebreaks=[dict(bounds=["sat", "mon"])])
    base.update(kw)
    return base

def _yax(**kw):
    base = dict(showgrid=True, gridcolor="#f3f4f6", linecolor=BORDER,
                tickfont=dict(size=9, color=TEXT_DIM))
    base.update(kw)
    return base

def _sec(title, top=24):
    st.markdown(
        f"<h3 style='margin-top:{top}px;margin-bottom:10px;font-size:10px;"
        f"font-weight:600;color:{TEXT_DIM};letter-spacing:0.1em;"
        f"text-transform:uppercase'>{title}</h3>",
        unsafe_allow_html=True)

def _kpi(col, label, value, sub="", vc=TEXT):
    col.markdown(
        f"<div style='background:{BG2};border:1px solid {BORDER};border-radius:6px;"
        f"padding:10px 14px;text-align:center'>"
        f"<div style='font-size:9px;color:{TEXT_DIM};text-transform:uppercase;"
        f"letter-spacing:0.08em;font-weight:600'>{label}</div>"
        f"<div style='font-size:20px;font-weight:600;color:{vc};margin:2px 0'>{value}</div>"
        f"<div style='font-size:9px;color:{TEXT_DIM}'>{sub}</div></div>",
        unsafe_allow_html=True)

def _fmt(v, sfx=""):
    if pd.isna(v) or (isinstance(v, float) and np.isnan(v)): return "—"
    sign = "+" if v > 0 else ""
    d = 1 if abs(v) >= 10 else 2
    return f"{sign}{v:.{d}f}{sfx}"

def _dd(s):
    s = s.ffill()
    return (s / s.expanding().max() - 1) * 100

def _sharpe(ret_series):
    r = ret_series.dropna()
    if len(r) < 5 or r.std() == 0: return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))

def _cum_ret(s):
    s = s.dropna()
    if len(s) < 2: return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1) * 100

# ══════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════

def compute_entropy_window(values, bins=50, value_range=(0.5, 1.5)):
    hist, bin_edges = np.histogram(values, bins=bins, range=value_range, density=True)
    bin_width = np.diff(bin_edges)
    probs = hist * bin_width
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

def compute_rolling_entropy(slope_series, window=60, bins=50, value_range=(0.5, 1.5)):
    raw = slope_series.rolling(window).apply(
        lambda x: compute_entropy_window(x, bins=bins, value_range=value_range), raw=True)
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-10: return raw * 0.0
    return (raw - mn) / (mx - mn)

def compute_slope_dynamics(slope, window=60):
    delta = slope.diff().rolling(window).mean()
    gamma = delta.diff()
    return delta, gamma

def compute_runs(is_contango):
    changes = is_contango.diff().fillna(0) != 0
    run_id = changes.cumsum()
    current_run_id = run_id.iloc[-1]
    return bool(is_contango.iloc[-1]), int((run_id == current_run_id).sum())

def risk_regime_label(vix_val, ratio, entropy_val):
    if vix_val > 30: return "CRISIS", RED
    if vix_val > 25 or ratio > 1.0: return "HIGH RISK", RED
    if vix_val > 20 or ratio > 0.95 or entropy_val < 0.15: return "ELEVATED", YELLOW
    if ratio < 0.85 and entropy_val > 0.3: return "FAVORABLE", GREEN
    return "NEUTRAL", TEXT_DIM

def _extend_svix_with_vxx(svix_raw, vxx_raw):
    if svix_raw.empty or vxx_raw.empty: return svix_raw
    first_svix = svix_raw.first_valid_index()
    if first_svix is None: return svix_raw
    vxx_before = vxx_raw.loc[:first_svix]
    if len(vxx_before) < 2: return svix_raw
    start_val = float(svix_raw.loc[first_svix])
    vxx_rets = vxx_before.pct_change().dropna()
    synthetic_rets = -vxx_rets
    fwd = (1 + synthetic_rets[::-1]).cumprod()[::-1]
    scale = start_val / float(fwd.iloc[-1])
    synthetic_px = (fwd * scale).iloc[:-1]
    return synthetic_px.combine_first(svix_raw).sort_index()

# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
SYMS = ["SVIX", "^VIX", "^VIX3M", "^VVIX", "SVXY", "VXX", "^VIX1D", "^VIX6M", "^VIX9D"]

@st.cache_data(ttl=3600)
def _load():
    out = {}
    for s in SYMS:
        df = load_prices(s)
        if not df.empty:
            df = df.set_index("datetime").sort_index()
            df.index = pd.to_datetime(df.index)
            out[s] = df["close"].dropna()
    if "SVIX" in out and "VXX" in out:
        out["SVIX"] = _extend_svix_with_vxx(out["SVIX"], out["VXX"])
    return out

raw = _load()
def S(sym):
    return raw.get(sym, pd.Series(dtype=float))

svix  = S("SVIX");   vix   = S("^VIX")
vix3m = S("^VIX3M"); vvix  = S("^VVIX")
vix1d = S("^VIX1D"); vix6m = S("^VIX6M")
vix9d = S("^VIX9D")

if svix.empty or vix.empty:
    st.warning("SVIX or VIX data not available. Run the pipeline first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    _PERIODS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260}
    sel_p = st.selectbox("Period", list(_PERIODS.keys()), index=2)
    N = _PERIODS[sel_p]
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    show_svxy = st.checkbox("Show SVXY", value=True)
    show_vxx  = st.checkbox("Show VXX",  value=False)
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    entropy_window    = st.slider("Entropy window", 30, 120, 60, step=10)
    entropy_threshold = st.slider("Entropy threshold", 0.05, 0.50, 0.20, step=0.05)

# ── Windowed ──────────────────────────────────────────────────────
svix_w  = svix.tail(N);  vix_w  = vix.tail(N)
vix3m_w = vix3m.tail(N) if not vix3m.empty else pd.Series(dtype=float)
vvix_w  = vvix.tail(N)  if not vvix.empty  else pd.Series(dtype=float)
vix1d_w = vix1d.tail(N) if not vix1d.empty else pd.Series(dtype=float)
vix6m_w = vix6m.tail(N) if not vix6m.empty else pd.Series(dtype=float)
vix9d_w = vix9d.tail(N) if not vix9d.empty else pd.Series(dtype=float)
# ── SVIX core ─────────────────────────────────────────────────────
ret_d  = svix_w.pct_change()

def _rolling_ret(n):
    return (svix.iloc[-1] / svix.iloc[-n-1] - 1) * 100 if len(svix) > n else np.nan

r1d = _rolling_ret(1); r1w = _rolling_ret(5)
r1m = _rolling_ret(21); r3m = _rolling_ret(63)
cum_ret_svix = _cum_ret(svix_w)
dd_svix  = _dd(svix_w)
max_dd   = float(dd_svix.min())
curr_dd  = float(dd_svix.iloc[-1])
worst_dt = dd_svix.idxmin()
sharpe_svix = _sharpe(ret_d)

# ── Slope ─────────────────────────────────────────────────────────
if not vix3m_w.empty:
    _idx = vix_w.index.intersection(vix3m_w.index)
    slope   = (vix.reindex(_idx) / vix3m.reindex(_idx)).dropna()
    slope_w = slope.tail(N)
    in_ctg  = slope_w < 1.0
    ratio_now = float(slope.iloc[-1]) if len(slope) > 0 else np.nan
    pct_ctg   = float(in_ctg.mean() * 100)
else:
    slope = slope_w = pd.Series(dtype=float)
    in_ctg = pd.Series(dtype=bool)
    ratio_now = pct_ctg = np.nan

vix_now  = float(vix.iloc[-1])
vvix_now = float(vvix.iloc[-1]) if not vvix.empty else np.nan
vix_pctile = percentileofscore(vix.dropna().values, vix_now)

# ── Entropy ───────────────────────────────────────────────────────
if len(slope) > entropy_window:
    entropy_full = compute_rolling_entropy(slope, window=entropy_window)
    entropy_now  = float(entropy_full.iloc[-1])
    entropy_w    = entropy_full.tail(N)
    delta_slope, gamma_slope = compute_slope_dynamics(slope, window=entropy_window)
    delta_now = float(delta_slope.iloc[-1]) if len(delta_slope) > 0 else np.nan
else:
    entropy_full = entropy_w = delta_slope = gamma_slope = pd.Series(dtype=float)
    entropy_now = delta_now = np.nan

# ── Runs / regime ─────────────────────────────────────────────────
if len(slope) > 1:
    run_dir, run_len = compute_runs((slope < 1.0).astype(int))
else:
    run_dir, run_len = True, 0

regime_lbl, regime_col = risk_regime_label(
    vix_now,
    ratio_now if not np.isnan(ratio_now) else 1.0,
    entropy_now if not np.isnan(entropy_now) else 0.0
)

# ── VVIX/VIX ─────────────────────────────────────────────────────
if not vvix.empty:
    vvix_vix_ratio = (vvix / vix.reindex(vvix.index)).dropna()
    vvix_vix_now   = float(vvix_vix_ratio.iloc[-1]) if len(vvix_vix_ratio) > 0 else np.nan
else:
    vvix_vix_ratio = pd.Series(dtype=float)
    vvix_vix_now   = np.nan

# ══════════════════════════════════════════════════════════════════
# STRATEGY BACKTESTS
# ══════════════════════════════════════════════════════════════════
svix_ret_daily = svix.pct_change()

# Strategy 1 — ratio carry: long SVIX when VIX/VIX3M < 1
if len(slope) > 1:
    sig_ratio_full   = (slope < 1.0).astype(int)
    sig_ratio_w      = sig_ratio_full.reindex(svix_w.index).ffill().fillna(0)
    strat_ratio_rets = (sig_ratio_full.shift(1) * svix_ret_daily).dropna()
    strat_ratio_w    = (sig_ratio_w.shift(1) * ret_d).dropna()
    dd_ratio         = _dd((1 + strat_ratio_w).cumprod())
    cum_ratio        = _cum_ret((1 + strat_ratio_w).cumprod())
    sharpe_ratio     = _sharpe(strat_ratio_rets)
    max_dd_ratio     = float(dd_ratio.min()) if not dd_ratio.empty else np.nan
else:
    sig_ratio_w = strat_ratio_rets = strat_ratio_w = dd_ratio = pd.Series(dtype=float)
    cum_ratio = sharpe_ratio = max_dd_ratio = np.nan

# Strategy 2 — entropy: long SVIX when slope<1 AND entropy >= threshold
if not entropy_full.empty and len(slope) > 1:
    sig_entropy_full   = ((slope < 1.0) & (entropy_full >= entropy_threshold)).astype(int)
    sig_entropy_w      = sig_entropy_full.reindex(svix_w.index).ffill().fillna(0)
    strat_entropy_rets = (sig_entropy_full.shift(1) * svix_ret_daily).dropna()
    strat_entropy_w    = (sig_entropy_w.shift(1) * ret_d).dropna()
    dd_entropy         = _dd((1 + strat_entropy_w).cumprod())
    cum_entropy        = _cum_ret((1 + strat_entropy_w).cumprod())
    sharpe_entropy     = _sharpe(strat_entropy_rets)
    max_dd_entropy     = float(dd_entropy.min()) if not dd_entropy.empty else np.nan
else:
    sig_entropy_w = strat_entropy_rets = strat_entropy_w = dd_entropy = pd.Series(dtype=float)
    cum_entropy = sharpe_entropy = max_dd_entropy = np.nan

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
k = st.columns(8)
_kpi(k[0], "regime",    regime_lbl, vc=regime_col)
_kpi(k[1], "VIX",       f"{vix_now:.1f}", sub=f"p{vix_pctile:.0f}",
     vc=RED if vix_now > 20 else YELLOW if vix_now > 15 else GREEN)
_kpi(k[2], "VIX/VIX3M", f"{ratio_now:.3f}" if not np.isnan(ratio_now) else "—",
     sub="contango" if not np.isnan(ratio_now) and ratio_now < 1 else "backwardation",
     vc=GREEN if not np.isnan(ratio_now) and ratio_now < 0.9
        else RED if np.isnan(ratio_now) or ratio_now >= 1 else YELLOW)
_kpi(k[3], "entropy",   f"{entropy_now:.2f}" if not np.isnan(entropy_now) else "—",
     sub=f"w={entropy_window}",
     vc=GREEN if not np.isnan(entropy_now) and entropy_now > 0.3
        else RED if np.isnan(entropy_now) or entropy_now < 0.15 else YELLOW)
_kpi(k[4], "SVIX return",  _fmt(cum_ret_svix, "%"), sub=f"{sel_p} underlying",
     vc=GREEN if cum_ret_svix > 0 else RED)
_kpi(k[5], "slope ret", _fmt(cum_ratio, "%"), sub=f"{sel_p} carry",
     vc=GREEN if not np.isnan(cum_ratio) and cum_ratio > 0 else RED)
_kpi(k[6], "entropy strat ret", _fmt(cum_entropy, "%"), sub=f"{sel_p} filtered",
     vc=GREEN if not np.isnan(cum_entropy) and cum_entropy > 0 else RED)
_kpi(k[7], "VVIX",
     f"{vvix_now:.1f}" if not np.isnan(vvix_now) else "—",
     sub=f"ratio {vvix_vix_now:.1f}" if not np.isnan(vvix_vix_now) else "",
     vc=RED if not np.isnan(vvix_now) and vvix_now > 120
        else YELLOW if not np.isnan(vvix_now) and vvix_now > 100 else TEXT)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "Performance", "Term Structure & Entropy", "Signals",
    "Drawdowns", "VIX Dashboard"
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with t1:
    col_l, col_r = st.columns([3, 1], gap="large")
    with col_l:
        _sec(f"SVIX underlying — {sel_p}", top=8)
        fig_und = go.Figure()
        for sym, col_line, show in [("SVXY",_C_SVXY,show_svxy),("VXX",_C_VXX,show_vxx)]:
            if show:
                s = raw.get(sym, pd.Series()).reindex(svix_w.index).dropna()
                if len(s) > 5:
                    base_v = float(svix_w.reindex(s.index).dropna().iloc[0])
                    s_rb = s / float(s.iloc[0]) * base_v
                    fig_und.add_trace(go.Scatter(
                        x=s_rb.index, y=s_rb.values, mode="lines", name=sym,
                        line=dict(color=col_line, width=1.2, dash="dot"),
                        hovertemplate=f"{sym} (rebased): %{{y:.2f}}<extra></extra>"))
        fig_und.add_trace(go.Scatter(
            x=svix_w.index, y=svix_w.values, mode="lines", name="SVIX (underlying)",
            line=dict(color=_C_SVIX, width=2),
            hovertemplate="SVIX: %{y:.2f}<extra></extra>"))
        fig_und.update_layout(**_layout(height=240), legend=_LEG_TOP)
        fig_und.update_xaxes(**_xax()); fig_und.update_yaxes(**_yax())
        st.plotly_chart(fig_und, use_container_width=True)

        _sec(f"strategy performance — {sel_p}")
        fig_strat = go.Figure()
        fig_strat.add_hline(y=1, line_color=BORDER, line_width=1)
        svix_cum = (1 + ret_d.fillna(0)).cumprod()
        fig_strat.add_trace(go.Scatter(
            x=svix_cum.index, y=svix_cum.values, mode="lines",
            name="SVIX buy & hold (underlying)",
            line=dict(color=_C_SVIX, width=1.5, dash="dot"),
            hovertemplate="B&H: %{y:.3f}<extra></extra>"))
        if not strat_ratio_w.empty:
            rc = (1 + strat_ratio_w.fillna(0)).cumprod()
            fig_strat.add_trace(go.Scatter(
                x=rc.index, y=rc.values, mode="lines",
                name="VIX/VIX3M slope carry",
                line=dict(color=_C_RATIO, width=2),
                hovertemplate="Slope carry: %{y:.3f}<extra></extra>"))
        if not strat_entropy_w.empty:
            ec = (1 + strat_entropy_w.fillna(0)).cumprod()
            fig_strat.add_trace(go.Scatter(
                x=ec.index, y=ec.values, mode="lines",
                name="Entropy filtered",
                line=dict(color=_C_ENTROPY, width=2),
                hovertemplate="Entropy: %{y:.3f}<extra></extra>"))
        fig_strat.update_layout(**_layout(height=260), legend=_LEG_TOP)
        fig_strat.update_xaxes(**_xax()); fig_strat.update_yaxes(**_yax())
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_r:
        _sec("SVIX underlying", top=8)
        for lbl_r, val_r in [("1D",r1d),("1W",r1w),("1M",r1m),("3M",r3m),(sel_p,cum_ret_svix)]:
            c = GREEN if not pd.isna(val_r) and val_r > 0 else RED
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {BORDER};font-size:12px'>"
                f"<span style='color:{TEXT_DIM}'>{lbl_r}</span>"
                f"<span style='color:{c};font-weight:500'>{_fmt(val_r,'%')}</span></div>",
                unsafe_allow_html=True)

        _sec("strategy comparison", top=20)
        for lbl_s, v1, v2, c1h, c2h in [
            ("", "Slope Strat", "Entropy Strat", _C_RATIO, _C_ENTROPY),
            ("Return",  _fmt(cum_ratio,"%"),    _fmt(cum_entropy,"%"),     _C_RATIO, _C_ENTROPY),
            ("Sharpe",  f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "—",
                        f"{sharpe_entropy:.2f}" if not np.isnan(sharpe_entropy) else "—",
                        _C_RATIO, _C_ENTROPY),
            ("Max DD",  f"{max_dd_ratio:.1f}%" if not np.isnan(max_dd_ratio) else "—",
                        f"{max_dd_entropy:.1f}%" if not np.isnan(max_dd_entropy) else "—",
                        RED, RED),
        ]:
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1.1fr 1fr 1fr;"
                f"padding:5px 0;border-bottom:1px solid {BORDER};font-size:10px'>"
                f"<span style='color:{TEXT_DIM};font-weight:600'>{lbl_s}</span>"
                f"<span style='color:{c1h}'>{v1}</span>"
                f"<span style='color:{c2h}'>{v2}</span>"
                f"</div>", unsafe_allow_html=True)

        _sec("underlying snapshot", top=20)
        for lbl_s, val_s, c_s in [
            ("Sharpe",  f"{sharpe_svix:.2f}" if not np.isnan(sharpe_svix) else "—",
             GREEN if not np.isnan(sharpe_svix) and sharpe_svix > 1 else TEXT),
            ("Run",     f"{'C' if run_dir else 'B'} {run_len}d", GREEN if run_dir else RED),
            ("% Ctg",   f"{pct_ctg:.0f}%" if not np.isnan(pct_ctg) else "—",
             GREEN if not np.isnan(pct_ctg) and pct_ctg > 70 else YELLOW),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {BORDER};font-size:12px'>"
                f"<span style='color:{TEXT_DIM}'>{lbl_s}</span>"
                f"<span style='color:{c_s};font-weight:500'>{val_s}</span></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TAB 2 — TERM STRUCTURE & ENTROPY
# ─────────────────────────────────────────────────────────────────
with t2:
    _sec("VIX term structure — current snapshot", top=8)

    _ts_maturities = [
        ("VIX 1D",  1,   vix1d),
        ("VIX 9D", 1, vix9d),
        ("VIX", 30,  vix),      # ^VIX = 30-day implied vol (CBOE definition)
        ("VIX 3M",  93,  vix3m),
        ("VIX 6M",  180, vix6m),
    ]

    # FIX — renamed loop variable from `lbl` to `ts_lbl` to avoid shadowing outer `lbl` functions
    _ts_pts = [
        (ts_lbl, days, float(s.iloc[-1]))
        for ts_lbl, days, s in _ts_maturities
        if not s.empty
    ]

    if len(_ts_pts) >= 2:
        _ts_labels = [p[0] for p in _ts_pts]
        _ts_vals   = [p[2] for p in _ts_pts]

        # FIX — term structure point colors: contango = green (upward sloping = normal),
        # backwardation = red (downward slope = risk-on alert)
        # First point gets the current curve color, subsequent points colored by slope direction
        _ts_dot_colors = []
        for i, v in enumerate(_ts_vals):
            if i == 0:
                _ts_dot_colors.append(_C_TS_NOW)
            else:
                # Contango = each maturity higher than previous = upward sloping = GREEN
                _ts_dot_colors.append(GREEN if v >= _ts_vals[i-1] else RED)

        fig_ts = go.Figure()

        # FIX — current curve line: bold blue, clearly visible
        fig_ts.add_trace(go.Scatter(
            x=_ts_labels, y=_ts_vals, mode="lines",
            line=dict(color=_C_TS_NOW, width=2.5),
            showlegend=False, hoverinfo="skip"))

        # Current dots — large, color-coded by slope direction
        fig_ts.add_trace(go.Scatter(
            x=_ts_labels, y=_ts_vals, mode="markers+text",
            name="Current",
            marker=dict(
                color=_ts_dot_colors,
                size=18,
                line=dict(color=BG2, width=2),
            ),
            text=[f"{v:.1f}" for v in _ts_vals],
            textposition="top center",
            textfont=dict(size=11, family=FONT, color=TEXT, weight="bold"),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>"))

        # Historical snapshots — FIX: use distinct visible colors
        for lookback_n, snap_lbl, snap_col, snap_width, snap_size in [
            (1,  "1D ago", _C_TS_1D, 1.8, 8),
            (5, "1W ago", _C_TS_1W, 1.2, 6),
            (21, "1M ago", _C_TS_1M, 1.1, 5),
        ]:
            _snap_pts = []
            for ts_lbl, days, s in _ts_maturities:   # FIX — ts_lbl, not lbl
                if not s.empty and len(s) > lookback_n:
                    _snap_pts.append((ts_lbl, float(s.iloc[-lookback_n-1])))
            if _snap_pts:
                fig_ts.add_trace(go.Scatter(
                    x=[p[0] for p in _snap_pts],
                    y=[p[1] for p in _snap_pts],
                    mode="lines+markers", name=snap_lbl,
                    line=dict(color=snap_col, width=snap_width, dash="dot"),
                    marker=dict(color=snap_col, size=snap_size,
                                line=dict(color=BG2, width=1)),
                    hovertemplate=f"{snap_lbl} — %{{x}}: %{{y:.2f}}<extra></extra>"))

        fig_ts.update_layout(
            **_layout(height=300, hovermode="x"),
            legend=_LEG_TOP,
            xaxis=dict(showgrid=False, linecolor=BORDER,
                       tickfont=dict(size=12, color=TEXT, family=FONT),
                       categoryorder="array", categoryarray=_ts_labels),
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", linecolor=BORDER,
                       tickfont=dict(size=9, color=TEXT_DIM), title="VIX level"),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    else:
        st.info("Not enough VIX term structure data available.")

    if not slope_w.empty:
        _sec("VIX/VIX3M slope & entropy")
        fig_se = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.6,0.4], vertical_spacing=0.06)
        fig_se.add_trace(go.Scatter(
            x=slope_w.dropna().index, y=slope_w.dropna().values,
            mode="lines", name="VIX/VIX3M",
            line=dict(color=_C_RATIO, width=1.8),
            hovertemplate="Slope: %{y:.3f}<extra></extra>"), row=1, col=1)
        fig_se.add_hline(y=1.0, line_color=RED, line_width=1, line_dash="dot", row=1, col=1)
        fig_se.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(5,150,105,0.06)", line_width=0, row=1, col=1)
        fig_se.add_hrect(y0=1.0, y1=1.5, fillcolor="rgba(220,38,38,0.06)", line_width=0, row=1, col=1)
        if not entropy_w.empty:
            fig_se.add_trace(go.Scatter(
                x=entropy_w.dropna().index, y=entropy_w.dropna().values,
                mode="lines", name="Entropy",
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
                line=dict(color=_C_ENTROPY, width=1.4),
                hovertemplate="Entropy: %{y:.3f}<extra></extra>"), row=2, col=1)
            fig_se.add_hline(y=entropy_threshold, line_color=YELLOW, line_width=1,
                             line_dash="dot", row=2, col=1,
                             annotation_text=f"threshold {entropy_threshold}",
                             annotation_font=dict(size=8, color=YELLOW))
        fig_se.update_layout(**_layout(height=380), legend=_LEG_TOP)
        fig_se.update_xaxes(**_xax())
        fig_se.update_yaxes(**_yax(), row=1, col=1)
        fig_se.update_yaxes(**_yax(), row=2, col=1)
        st.plotly_chart(fig_se, use_container_width=True)

        _sec("contango (green) vs backwardation (red)")
        ctg_days = in_ctg.index[in_ctg]; bwd_days = in_ctg.index[~in_ctg]
        fig_rug = go.Figure()
        if len(ctg_days):
            fig_rug.add_trace(go.Scatter(x=ctg_days, y=[1]*len(ctg_days), mode="markers",
                marker=dict(color=GREEN, size=3, symbol="square"), name="Contango", hoverinfo="skip"))
        if len(bwd_days):
            fig_rug.add_trace(go.Scatter(x=bwd_days, y=[1]*len(bwd_days), mode="markers",
                marker=dict(color=RED, size=3, symbol="square"), name="Backwardation", hoverinfo="skip"))
        fig_rug.update_layout(**_layout(height=70, hovermode=False,
                                        margin=dict(l=48,r=16,t=8,b=28)),
                              legend=_LEG_TOP,
                              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
        fig_rug.update_xaxes(**_xax())
        st.plotly_chart(fig_rug, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        n_ctg = int(in_ctg.sum()); n_bwd = int((~in_ctg).sum())
        _kpi(c1, "days contango", f"{n_ctg}", vc=GREEN)
        _kpi(c2, "days backwdn",  f"{n_bwd}", vc=RED)
        _kpi(c3, "% contango", f"{pct_ctg:.1f}%" if not np.isnan(pct_ctg) else "—",
             vc=GREEN if not np.isnan(pct_ctg) and pct_ctg > 70 else YELLOW)
        _kpi(c4, "current run", f"{'Contango' if run_dir else 'Backwdn'} — {run_len}d",
             vc=GREEN if run_dir else RED)

        if not delta_slope.empty:
            _sec("slope dynamics — delta & gamma")
            fig_dg = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.5,0.5], vertical_spacing=0.08)
            d_plt = delta_slope.tail(N).dropna()
            g_plt = gamma_slope.tail(N).dropna()
            fig_dg.add_trace(go.Scatter(x=d_plt.index, y=d_plt.values, mode="lines",
                name="Δ slope", line=dict(color=_C_RATIO, width=1.3),
                hovertemplate="Delta: %{y:.5f}<extra></extra>"), row=1, col=1)
            fig_dg.add_hline(y=0, line_color=GRAY, line_width=1, row=1, col=1)
            fig_dg.add_trace(go.Scatter(x=g_plt.index, y=g_plt.values, mode="lines",
                name="Γ slope", line=dict(color=_C_ENTROPY, width=1.3),
                hovertemplate="Gamma: %{y:.6f}<extra></extra>"), row=2, col=1)
            fig_dg.add_hline(y=0, line_color=GRAY, line_width=1, row=2, col=1)
            fig_dg.update_layout(**_layout(height=280), legend=_LEG_TOP)
            fig_dg.update_xaxes(**_xax())
            fig_dg.update_yaxes(**_yax(), row=1, col=1)
            fig_dg.update_yaxes(**_yax(), row=2, col=1)
            st.plotly_chart(fig_dg, use_container_width=True)
    else:
        st.info("VIX3M not available.")

# ─────────────────────────────────────────────────────────────────
# TAB 3 — SIGNALS
# ─────────────────────────────────────────────────────────────────
with t3:
    if slope_w.empty or entropy_w.empty:
        st.info("Signals require VIX3M + sufficient history for entropy computation.")
    else:
        _sec("strategy signals — ratio carry & entropy filter", top=8)
        fig_sig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.45,0.275,0.275], vertical_spacing=0.05,
                                subplot_titles=["SVIX underlying", "VIX/VIX3M slope carry signal", "Entropy filtered signal"])
        fig_sig.add_trace(go.Scatter(
            x=svix_w.index, y=svix_w.values, mode="lines", name="SVIX",
            line=dict(color=_C_SVIX, width=1.6),
            hovertemplate="SVIX: %{y:.2f}<extra></extra>"), row=1, col=1)

        # Shade ratio ON
        sr_al = sig_ratio_w.reindex(svix_w.index).fillna(0)
        on_r = sr_al[sr_al==1].index
        if len(on_r):
            fig_sig.add_trace(go.Scatter(
                x=on_r, y=svix_w.reindex(on_r).values, mode="markers",
                name="Slope ON", marker=dict(color=_C_RATIO, size=2, opacity=0.5),
                hoverinfo="skip"), row=1, col=1)

        # Shade entropy ON
        if not sig_entropy_w.empty:
            se_al = sig_entropy_w.reindex(svix_w.index).fillna(0)
            on_e = se_al[se_al==1].index
            if len(on_e):
                fig_sig.add_trace(go.Scatter(
                    x=on_e, y=svix_w.reindex(on_e).values, mode="markers",
                    name="Entropy ON", marker=dict(color=_C_ENTROPY, size=2, opacity=0.5),
                    hoverinfo="skip"), row=1, col=1)

        # Row 2 — slope
        fig_sig.add_trace(go.Scatter(
            x=slope_w.index, y=slope_w.values, mode="lines", name="VIX/VIX3M",
            line=dict(color=_C_RATIO, width=1.3),
            hovertemplate="Slope: %{y:.3f}<extra></extra>"), row=2, col=1)
        fig_sig.add_hline(y=1.0, line_color=RED, line_width=1, line_dash="dot", row=2, col=1)
        slope_ctg = slope_w.copy(); slope_ctg[slope_ctg>=1] = np.nan
        fig_sig.add_trace(go.Scatter(
            x=slope_ctg.index, y=slope_ctg.values, mode="none",
            fill="tozeroy", fillcolor="rgba(217,119,6,0.12)",
            showlegend=False, hoverinfo="skip"), row=2, col=1)

        # Row 3 — entropy signal
        if not sig_entropy_w.empty:
            fig_sig.add_trace(go.Scatter(
                x=sig_entropy_w.index, y=sig_entropy_w.values,
                mode="lines", name="Entropy signal",
                fill="tozeroy", fillcolor="rgba(124,58,237,0.15)",
                line=dict(color=_C_ENTROPY, width=1),
                hovertemplate="Signal: %{y}<extra></extra>"), row=3, col=1)

        fig_sig.update_layout(**_layout(height=500, margin=dict(l=48,r=16,t=40,b=44)),
                              legend=_LEG_TOP)
        fig_sig.update_xaxes(**_xax())
        for r in [1,2,3]:
            fig_sig.update_yaxes(**_yax(), row=r, col=1)
        for ann in fig_sig.layout.annotations:
            ann.update(font=dict(size=9, color=TEXT_DIM, family=FONT), x=0, xanchor="left")
        st.plotly_chart(fig_sig, use_container_width=True)

        ratio_on_pct   = float(sig_ratio_w.mean()*100) if not sig_ratio_w.empty else np.nan
        entropy_on_pct = float(sig_entropy_w.mean()*100) if not sig_entropy_w.empty else np.nan
        ratio_now_on   = not np.isnan(ratio_now) and ratio_now < 1.0
        entropy_now_on = (not np.isnan(entropy_now) and not np.isnan(ratio_now)
                          and ratio_now < 1.0 and entropy_now >= entropy_threshold)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        _kpi(c1, "Slope signal",  "ON" if ratio_now_on else "OFF",   vc=GREEN if ratio_now_on else RED)
        _kpi(c2, f"Slope % ON",   f"{ratio_on_pct:.0f}%" if not np.isnan(ratio_on_pct) else "—",
             vc=GREEN if not np.isnan(ratio_on_pct) and ratio_on_pct > 60 else YELLOW)
        _kpi(c3, "entropy signal","ON" if entropy_now_on else "OFF", vc=GREEN if entropy_now_on else RED)
        _kpi(c4, f"entropy % ON", f"{entropy_on_pct:.0f}%" if not np.isnan(entropy_on_pct) else "—",
             vc=GREEN if not np.isnan(entropy_on_pct) and entropy_on_pct > 60 else YELLOW)
        _kpi(c5, "slope < 1",     "YES" if ratio_now_on else "NO",   vc=GREEN if ratio_now_on else RED)
        _kpi(c6, f"ent ≥ {entropy_threshold}",
             "YES" if not np.isnan(entropy_now) and entropy_now >= entropy_threshold else "NO",
             vc=GREEN if not np.isnan(entropy_now) and entropy_now >= entropy_threshold else RED)

# ─────────────────────────────────────────────────────────────────
# TAB 4 — DRAWDOWNS
# ─────────────────────────────────────────────────────────────────
with t4:
    _sec("underwater curves — all strategies", top=8)
    fig_dd_all = go.Figure()
    fig_dd_all.add_hline(y=0, line_color=BORDER, line_width=1)
    fig_dd_all.add_trace(go.Scatter(
        x=dd_svix.index, y=dd_svix.values, mode="lines",
        name="SVIX underlying",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        line=dict(color=_C_SVIX, width=1.3, dash="dot"),
        hovertemplate="SVIX B&H: %{y:.2f}%<extra></extra>"))
    if not dd_ratio.empty:
        fig_dd_all.add_trace(go.Scatter(
            x=dd_ratio.index, y=dd_ratio.values, mode="lines",
            name="VIX/VIX3M slope carry",
            line=dict(color=_C_RATIO, width=1.8),
            hovertemplate="Slope carry: %{y:.2f}%<extra></extra>"))
    if not dd_entropy.empty:
        fig_dd_all.add_trace(go.Scatter(
            x=dd_entropy.index, y=dd_entropy.values, mode="lines",
            name="Entropy filtered",
            line=dict(color=_C_ENTROPY, width=1.8),
            hovertemplate="Entropy: %{y:.2f}%<extra></extra>"))
    fig_dd_all.update_layout(**_layout(height=320), legend=_LEG_TOP)
    fig_dd_all.update_xaxes(**_xax()); fig_dd_all.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_dd_all, use_container_width=True)

    dl, dr = st.columns(2, gap="large")
    with dl:
        _sec("worst 5 — SVIX underlying", top=8)
        dd_full = _dd(svix); dd_copy = dd_full.copy()
        for _ in range(5):
            if dd_copy.empty or dd_copy.min() >= -0.5: break
            idx_m = dd_copy.idxmin(); val_m = float(dd_copy.loc[idx_m])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                f"border-bottom:1px solid {BORDER};font-size:11px'>"
                f"<span style='color:{TEXT_DIM}'>{idx_m.strftime('%Y-%m-%d')}</span>"
                f"<span style='color:{_C_SVIX};font-weight:500'>{val_m:.1f}%</span></div>",
                unsafe_allow_html=True)
            dd_copy.loc[idx_m-pd.Timedelta(days=60):idx_m+pd.Timedelta(days=60)] = 0.0

    with dr:
        if not dd_entropy.empty:
            _sec("worst 5 — entropy strategy", top=8)
            dd_ent_c = dd_entropy.copy()
            for _ in range(5):
                if dd_ent_c.empty or dd_ent_c.min() >= -0.5: break
                idx_m = dd_ent_c.idxmin(); val_m = float(dd_ent_c.loc[idx_m])
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                    f"border-bottom:1px solid {BORDER};font-size:11px'>"
                    f"<span style='color:{TEXT_DIM}'>{idx_m.strftime('%Y-%m-%d')}</span>"
                    f"<span style='color:{_C_ENTROPY};font-weight:500'>{val_m:.1f}%</span></div>",
                    unsafe_allow_html=True)
                dd_ent_c.loc[idx_m-pd.Timedelta(days=60):idx_m+pd.Timedelta(days=60)] = 0.0

# ─────────────────────────────────────────────────────────────────
# TAB 5 — VIX DASHBOARD
# ─────────────────────────────────────────────────────────────────
with t5:
    row1_l, row1_r = st.columns(2, gap="medium")

    with row1_l:
        _sec("VIX — spot level", top=8)
        fig_vix = go.Figure()
        fig_vix.add_hrect(y0=0,  y1=15, fillcolor="rgba(5,150,105,0.06)",  line_width=0)
        fig_vix.add_hrect(y0=20, y1=30, fillcolor="rgba(217,119,6,0.05)",  line_width=0)
        fig_vix.add_hrect(y0=30, y1=90, fillcolor="rgba(220,38,38,0.05)",  line_width=0)
        for lvl, col_l in [(15,GRAY),(20,YELLOW),(30,RED)]:
            fig_vix.add_hline(y=lvl, line_color=col_l, line_width=1, line_dash="dot")
        fig_vix.add_trace(go.Scatter(
            x=vix_w.index, y=vix_w.values, mode="lines", name="VIX",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            line=dict(color=RED, width=1.7),
            hovertemplate="VIX: %{y:.2f}<extra></extra>"))
        fig_vix.add_annotation(x=vix_w.index[-1], y=vix_now,
            text=f"  {vix_now:.1f}", showarrow=False,
            font=dict(size=11,family=FONT,
                      color=RED if vix_now>20 else YELLOW if vix_now>15 else GREEN),
            xanchor="left")
        fig_vix.update_layout(**_layout(height=240))
        fig_vix.update_xaxes(**_xax()); fig_vix.update_yaxes(**_yax())
        st.plotly_chart(fig_vix, use_container_width=True)

        _sec("VIX distribution")
        p10 = float(np.percentile(vix_w.dropna(), 10))
        p90 = float(np.percentile(vix_w.dropna(), 90))
        fig_vd = go.Figure()
        fig_vd.add_trace(go.Histogram(x=vix_w.values, nbinsx=45, showlegend=False,
            marker=dict(color=RED, opacity=0.55, line=dict(width=0)),
            hovertemplate="VIX %{x:.1f}: %{y} days<extra></extra>"))
        fig_vd.add_vline(x=vix_now, line_color=BLUE, line_width=2,
                         annotation_text=f"now {vix_now:.1f}",
                         annotation_font=dict(size=9,family=FONT,color=BLUE))
        for x_v, col_v, lbl_v in [(p10,GREEN,"p10"),(p90,RED,"p90")]:
            fig_vd.add_vline(x=x_v, line_color=col_v, line_width=1, line_dash="dot",
                             annotation_text=lbl_v,
                             annotation_font=dict(size=8,family=FONT,color=col_v))
        fig_vd.update_layout(**_layout(height=200, hovermode="x"))
        fig_vd.update_xaxes(**_xax()); fig_vd.update_yaxes(**_yax())
        st.plotly_chart(fig_vd, use_container_width=True)

    with row1_r:
        _sec("VVIX — vol of vol", top=8)
        if not vvix_w.empty:
            fig_vv = go.Figure()
            for lvl, col_v in [(100,YELLOW),(120,RED)]:
                fig_vv.add_hline(y=lvl, line_color=col_v, line_width=1, line_dash="dot",
                                 annotation_text=str(lvl),
                                 annotation_font=dict(size=9,family=FONT,color=col_v))
            fig_vv.add_trace(go.Scatter(
                x=vvix_w.index, y=vvix_w.values, mode="lines", name="VVIX",
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
                line=dict(color=_C_ENTROPY, width=1.7),
                hovertemplate="VVIX: %{y:.2f}<extra></extra>"))
            fig_vv.add_annotation(x=vvix_w.index[-1], y=vvix_now,
                text=f"  {vvix_now:.1f}", showarrow=False,
                font=dict(size=11,family=FONT,
                          color=RED if vvix_now>120 else YELLOW if vvix_now>100 else TEXT),
                xanchor="left")
            fig_vv.update_layout(**_layout(height=240))
            fig_vv.update_xaxes(**_xax()); fig_vv.update_yaxes(**_yax())
            st.plotly_chart(fig_vv, use_container_width=True)
        else:
            st.info("VVIX not available.")

        _sec("VIX/VIX3M slope distribution")
        if not slope_w.empty:
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Histogram(x=slope_w.dropna().values, nbinsx=40, showlegend=False,
                marker=dict(color=_C_RATIO, opacity=0.6, line=dict(width=0)),
                hovertemplate="Slope %{x:.3f}: %{y} days<extra></extra>"))
            fig_sr.add_vline(x=ratio_now, line_color=BLUE, line_width=2,
                             annotation_text=f"now {ratio_now:.3f}",
                             annotation_font=dict(size=9,family=FONT,color=BLUE))
            fig_sr.add_vline(x=1.0, line_color=RED, line_width=1, line_dash="dot",
                             annotation_text="1.0 (backwdn)",
                             annotation_font=dict(size=8,family=FONT,color=RED))
            fig_sr.update_layout(**_layout(height=200, hovermode="x"))
            fig_sr.update_xaxes(**_xax()); fig_sr.update_yaxes(**_yax())
            st.plotly_chart(fig_sr, use_container_width=True)

    if not vvix_vix_ratio.empty:
        _sec("VVIX / VIX ratio")
        vvix_vix_w = vvix_vix_ratio.tail(N)
        fig_vr = go.Figure()
        fig_vr.add_trace(go.Scatter(
            x=vvix_vix_w.index, y=vvix_vix_w.values, mode="lines",
            name="VVIX/VIX", line=dict(color="#0891b2", width=1.5),
            fill="tozeroy", fillcolor="rgba(8,145,178,0.06)",
            hovertemplate="VVIX/VIX: %{y:.2f}<extra></extra>"))
        med = float(vvix_vix_w.median())
        fig_vr.add_hline(y=med, line_color=GRAY, line_width=1, line_dash="dot",
                         annotation_text=f"median {med:.1f}",
                         annotation_font=dict(size=8,color=GRAY))
        fig_vr.update_layout(**_layout(height=200))
        fig_vr.update_xaxes(**_xax()); fig_vr.update_yaxes(**_yax())
        st.plotly_chart(fig_vr, use_container_width=True)