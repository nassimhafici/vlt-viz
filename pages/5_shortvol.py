"""
Short Vol — SVIX strategy monitor
Enhanced with entropy, slope dynamics, regime analysis & risk signals
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore

from core.db import load_prices
from core.formatting import (
    kpi_card,
    FONT, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

st.title("short vol")

# ══════════════════════════════════════════════════════════════════
# HELPERS — layout, section headers, kpi
# ══════════════════════════════════════════════════════════════════
_LEG_TOP = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM),
                orientation="h", x=0, y=1.08)

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
    base = dict(showgrid=False, linecolor=BORDER, tickfont=dict(size=9, color=TEXT_DIM),
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
    if pd.isna(v): return "—"
    sign = "+" if v > 0 else ""
    d = 1 if abs(v) >= 10 else 2
    return f"{sign}{v:.{d}f}{sfx}"


# ══════════════════════════════════════════════════════════════════
# ANALYTICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def compute_entropy_window(values, bins=50, value_range=(0.5, 1.5)):
    """Shannon entropy of slope distribution in a rolling window."""
    hist, bin_edges = np.histogram(values, bins=bins, range=value_range, density=True)
    bin_width = np.diff(bin_edges)
    probs = hist * bin_width
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

def compute_rolling_entropy(slope_series, window=60, bins=50, value_range=(0.5, 1.5)):
    """Normalized rolling Shannon entropy of a slope series."""
    raw = slope_series.rolling(window).apply(
        lambda x: compute_entropy_window(x, bins=bins, value_range=value_range), raw=True)
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-10:
        return raw * 0.0
    return (raw - mn) / (mx - mn)

def compute_slope_dynamics(slope, window=60):
    """Delta (1st derivative) and gamma (2nd derivative) of the slope."""
    delta = slope.diff().rolling(window).mean()
    gamma = delta.diff()
    return delta, gamma

def compute_runs(is_contango):
    """Run-length analysis: current run direction & length."""
    changes = is_contango.diff().fillna(0) != 0
    run_id = changes.cumsum()
    current_run_id = run_id.iloc[-1]
    current_run_direction = is_contango.iloc[-1]
    current_run_len = (run_id == current_run_id).sum()
    return current_run_direction, current_run_len

def risk_regime_label(vix_val, ratio, entropy_val):
    """Classify the current environment into a risk regime."""
    if vix_val > 30:
        return "CRISIS", RED
    if vix_val > 25 or ratio > 1.0:
        return "HIGH RISK", RED
    if vix_val > 20 or ratio > 0.95 or entropy_val < 0.15:
        return "ELEVATED", YELLOW
    if ratio < 0.85 and entropy_val > 0.3:
        return "FAVORABLE", GREEN
    return "NEUTRAL", TEXT_DIM


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
SYMS = ["SVIX", "^SHORTVOL", "^VIX", "^VIX3M", "^VVIX", "SVXY", "VXX", "^VIX1D", "^VIX6M"]


def proxy_series_backward(base, proxy):
    """Extend SVIX history backward using ^SHORTVOL returns."""
    base, proxy = base.align(proxy, join="inner")
    rets_proxy = proxy.pct_change()
    first_idx = base.first_valid_index()
    if first_idx is None:
        return proxy.copy()
    start_val = base.loc[first_idx]
    backward_rets = rets_proxy.loc[:first_idx].iloc[:-1]
    if backward_rets.empty:
        return base
    reconstructed = start_val / (1 + backward_rets[::-1]).cumprod()[::-1]
    return reconstructed.combine_first(base)


@st.cache_data(ttl=3600)
def _load():
    out = {}
    for s in SYMS:
        df = load_prices(s)
        if not df.empty:
            df = df.set_index("datetime").sort_index()
            df.index = pd.to_datetime(df.index)
            out[s] = df["close"].dropna()
    # Extend SVIX with SHORTVOL proxy for pre-ETF history
    if "SVIX" in out and "^SHORTVOL" in out:
        out["SVIX"] = proxy_series_backward(out["SVIX"], out["^SHORTVOL"])
    return out

raw = _load()
def S(sym): return raw.get(sym, pd.Series(dtype=float))

svix  = S("SVIX");   vix   = S("^VIX")
vix3m = S("^VIX3M"); vvix  = S("^VVIX")
vix1d = S("^VIX1D"); vix6m = S("^VIX6M")

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
    show_vxx  = st.checkbox("Show VXX", value=False)
    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    entropy_window = st.slider("Entropy window", 30, 120, 60, step=10)
    entropy_threshold = st.slider("Entropy threshold", 0.05, 0.50, 0.20, step=0.05)

# ── Windowed series ───────────────────────────────────────────────
svix_w  = svix.tail(N)
vix_w   = vix.tail(N)
vix3m_w = vix3m.tail(N) if not vix3m.empty else pd.Series(dtype=float)
vvix_w  = vvix.tail(N)  if not vvix.empty  else pd.Series(dtype=float)

# ── Core computations ─────────────────────────────────────────────
ret_d   = svix_w.pct_change()
vol_21  = ret_d.rolling(21).std() * np.sqrt(252) * 100
vol_63  = ret_d.rolling(63).std() * np.sqrt(252) * 100
cum_ret = (svix_w.iloc[-1] / svix_w.iloc[0] - 1) * 100

def _rolling_ret(n):
    return (svix.iloc[-1] / svix.iloc[-n - 1] - 1) * 100 if len(svix) > n else np.nan

r1d = _rolling_ret(1); r1w = _rolling_ret(5)
r1m = _rolling_ret(21); r3m = _rolling_ret(63)

# ── Term structure slope ──────────────────────────────────────────
idx = vix_w.index.intersection(vix3m_w.index)
slope = (vix.reindex(idx) / vix3m.reindex(idx)).dropna()
slope_w = slope.tail(N)
in_ctg = slope_w < 1.0

ratio_now = float(slope.iloc[-1]) if len(slope) > 0 else np.nan
vix_now   = float(vix.iloc[-1])
vvix_now  = float(vvix.iloc[-1]) if not vvix.empty else np.nan
pct_ctg   = float(in_ctg.mean() * 100) if len(in_ctg) > 0 else np.nan

# ── Entropy ───────────────────────────────────────────────────────
entropy_full = compute_rolling_entropy(slope, window=entropy_window)
entropy_now  = float(entropy_full.iloc[-1]) if len(entropy_full) > 0 else np.nan
entropy_w    = entropy_full.tail(N)

# ── Slope dynamics ────────────────────────────────────────────────
delta_slope, gamma_slope = compute_slope_dynamics(slope, window=entropy_window)
delta_now = float(delta_slope.iloc[-1]) if len(delta_slope) > 0 else np.nan
gamma_now = float(gamma_slope.iloc[-1]) if len(gamma_slope) > 0 else np.nan

# ── Runs ──────────────────────────────────────────────────────────
is_contango_full = (slope < 1.0).astype(int)
run_dir, run_len = compute_runs(is_contango_full)

# ── Risk regime ───────────────────────────────────────────────────
regime_lbl, regime_col = risk_regime_label(vix_now, ratio_now, entropy_now)

# ── Drawdown ──────────────────────────────────────────────────────
def _dd(s):
    s = s.ffill()
    return (s / s.expanding().max() - 1) * 100

dd_svix = _dd(svix_w)
max_dd  = float(dd_svix.min())
curr_dd = float(dd_svix.iloc[-1])

# ── Sharpe ────────────────────────────────────────────────────────
mu  = ret_d.mean()
sig = ret_d.std()
sharpe = (mu / sig * np.sqrt(252)) if sig > 0 else np.nan

# ── VIX percentile ────────────────────────────────────────────────
vix_pctile = percentileofscore(vix.dropna().values, vix_now)

# ── VVIX/VIX ratio ───────────────────────────────────────────────
vvix_vix_ratio = (vvix / vix.reindex(vvix.index)).dropna()
vvix_vix_now = float(vvix_vix_ratio.iloc[-1]) if len(vvix_vix_ratio) > 0 else np.nan


# ══════════════════════════════════════════════════════════════════
# KPI STRIP — top-level summary
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
k = st.columns(8)
_kpi(k[0], "regime", regime_lbl, vc=regime_col)
_kpi(k[1], f"ret {sel_p}", _fmt(cum_ret, "%"),
     vc=GREEN if cum_ret > 0 else RED)
_kpi(k[2], "VIX", f"{vix_now:.1f}",
     sub=f"p{vix_pctile:.0f}",
     vc=RED if vix_now > 20 else YELLOW if vix_now > 15 else GREEN)
_kpi(k[3], "VIX/VIX3M", f"{ratio_now:.3f}",
     sub="contango" if ratio_now < 1 else "backwardation",
     vc=GREEN if ratio_now < 0.9 else RED if ratio_now >= 1 else YELLOW)
_kpi(k[4], "entropy", f"{entropy_now:.2f}",
     sub=f"w={entropy_window}",
     vc=GREEN if entropy_now > 0.3 else RED if entropy_now < 0.15 else YELLOW)
_kpi(k[5], "Δ slope", f"{delta_now:.4f}" if not pd.isna(delta_now) else "—",
     sub="flattening" if delta_now > 0 else "steepening",
     vc=RED if delta_now > 0.01 else GREEN if delta_now < -0.01 else TEXT_DIM)
_kpi(k[6], "max DD", f"{max_dd:.1f}%",
     sub=f"now {curr_dd:.1f}%",
     vc=RED if max_dd < -20 else YELLOW if max_dd < -10 else TEXT)
_kpi(k[7], "VVIX", f"{vvix_now:.1f}" if not np.isnan(vvix_now) else "—",
     sub=f"VVIX/VIX {vvix_vix_now:.1f}" if not np.isnan(vvix_vix_now) else "",
     vc=RED if not np.isnan(vvix_now) and vvix_now > 120 else
        YELLOW if not np.isnan(vvix_now) and vvix_now > 100 else TEXT)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Performance", "Term Structure & Entropy", "Risk Signals",
    "Drawdowns", "Volatility", "VIX & VVIX"
])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with t1:
    col_l, col_r = st.columns([3, 1], gap="large")

    with col_l:
        _sec(f"SVIX price — {sel_p}", top=8)
        fig = go.Figure()
        for sym, col_line, show in [("SVXY", "#059669", show_svxy), ("VXX", "#dc2626", show_vxx)]:
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
        fig.update_layout(**_layout(height=280, margin=dict(l=54, r=16, t=28, b=44)),
                          legend=_LEG_TOP)
        fig.update_xaxes(**_xax()); fig.update_yaxes(**_yax())
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _sec("returns", top=8)
        ret_data = [
            ("1D", r1d), ("1W", r1w), ("1M", r1m), ("3M", r3m),
            (sel_p, cum_ret)
        ]
        for lbl, val in ret_data:
            c = GREEN if not pd.isna(val) and val > 0 else RED
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {BORDER};font-size:12px'>"
                f"<span style='color:{TEXT_DIM}'>{lbl}</span>"
                f"<span style='color:{c};font-weight:500'>{_fmt(val, '%')}</span></div>",
                unsafe_allow_html=True)

        _sec("risk snapshot", top=20)
        snap = [
            ("Sharpe", f"{sharpe:.2f}" if not pd.isna(sharpe) else "—",
             GREEN if not pd.isna(sharpe) and sharpe > 1 else TEXT),
            ("Vol 21D", f"{float(vol_21.iloc[-1]):.1f}%" if not vol_21.empty else "—", TEXT),
            ("Vol 63D", f"{float(vol_63.iloc[-1]):.1f}%" if not vol_63.empty else "—", TEXT),
            ("Run", f"{'C' if run_dir else 'B'} {run_len}d", GREEN if run_dir else RED),
            ("% Ctg", f"{pct_ctg:.0f}%", GREEN if pct_ctg > 70 else YELLOW),
        ]
        for lbl, val, c in snap:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {BORDER};font-size:12px'>"
                f"<span style='color:{TEXT_DIM}'>{lbl}</span>"
                f"<span style='color:{c};font-weight:500'>{val}</span></div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — TERM STRUCTURE & ENTROPY
# ─────────────────────────────────────────────────────────────────
with t2:
    # ── Slope + Entropy dual chart ──
    _sec("VIX/VIX3M slope & entropy", top=8)
    fig_se = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.06)

    slope_plt = slope_w.dropna()
    fig_se.add_trace(go.Scatter(
        x=slope_plt.index, y=slope_plt.values, mode="lines", name="VIX/VIX3M",
        line=dict(color=BLUE, width=1.6),
        hovertemplate="Slope: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig_se.add_hline(y=1.0, line_color=RED, line_width=1, line_dash="dot", row=1, col=1)

    # Color the background: green for contango, red for backwardation
    fig_se.add_hrect(y0=0.5, y1=1.0, fillcolor="rgba(5,150,105,0.06)", line_width=0, row=1, col=1)
    fig_se.add_hrect(y0=1.0, y1=1.5, fillcolor="rgba(220,38,38,0.06)", line_width=0, row=1, col=1)

    # Entropy
    ent_plt = entropy_w.dropna()
    fig_se.add_trace(go.Scatter(
        x=ent_plt.index, y=ent_plt.values, mode="lines", name="Entropy",
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="#7c3aed", width=1.4),
        hovertemplate="Entropy: %{y:.3f}<extra></extra>"), row=2, col=1)
    fig_se.add_hline(y=entropy_threshold, line_color=YELLOW, line_width=1,
                     line_dash="dot", row=2, col=1,
                     annotation_text=f"threshold {entropy_threshold}",
                     annotation_font=dict(size=8, color=YELLOW))

    fig_se.update_layout(**_layout(height=400, margin=dict(l=54, r=16, t=28, b=44)),
                         legend=_LEG_TOP)
    fig_se.update_xaxes(**_xax())
    fig_se.update_yaxes(**_yax(), row=1, col=1)
    fig_se.update_yaxes(**_yax(), row=2, col=1)
    st.plotly_chart(fig_se, use_container_width=True)

    # ── Contango / backwardation rug ──
    _sec("contango (green) vs backwardation (red)")
    ctg_days = in_ctg.index[in_ctg]
    bwd_days = in_ctg.index[~in_ctg]
    fig_rug = go.Figure()
    if len(ctg_days):
        fig_rug.add_trace(go.Scatter(
            x=ctg_days, y=[1] * len(ctg_days), mode="markers",
            marker=dict(color=GREEN, size=3, symbol="square"),
            name="Contango", hoverinfo="skip"))
    if len(bwd_days):
        fig_rug.add_trace(go.Scatter(
            x=bwd_days, y=[1] * len(bwd_days), mode="markers",
            marker=dict(color=RED, size=3, symbol="square"),
            name="Backwardation", hoverinfo="skip"))
    fig_rug.update_layout(**_layout(height=70, hovermode=False,
                                    margin=dict(l=54, r=16, t=8, b=28)),
                          legend=_LEG_TOP,
                          yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    fig_rug.update_xaxes(**_xax())
    st.plotly_chart(fig_rug, use_container_width=True)

    # ── Stats row ──
    c1, c2, c3, c4 = st.columns(4)
    n_ctg = int(in_ctg.sum()); n_bwd = int((~in_ctg).sum())
    _kpi(c1, "days contango", f"{n_ctg}", vc=GREEN)
    _kpi(c2, "days backwardation", f"{n_bwd}", vc=RED)
    _kpi(c3, "% contango", f"{pct_ctg:.1f}%", vc=GREEN if pct_ctg > 70 else YELLOW)
    _kpi(c4, "current run", f"{'Contango' if run_dir else 'Backwdn'} — {run_len}d",
         vc=GREEN if run_dir else RED)

    # ── Slope delta & gamma ──
    _sec("slope dynamics — delta (momentum) & gamma (acceleration)")
    fig_dg = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.5, 0.5], vertical_spacing=0.08)

    d_plt = delta_slope.tail(N).dropna()
    g_plt = gamma_slope.tail(N).dropna()

    fig_dg.add_trace(go.Scatter(
        x=d_plt.index, y=d_plt.values, mode="lines", name="Δ slope (delta)",
        line=dict(color=BLUE, width=1.3),
        hovertemplate="Delta: %{y:.5f}<extra></extra>"), row=1, col=1)
    fig_dg.add_hline(y=0, line_color=GRAY, line_width=1, row=1, col=1)

    fig_dg.add_trace(go.Scatter(
        x=g_plt.index, y=g_plt.values, mode="lines", name="Γ slope (gamma)",
        line=dict(color="#d97706", width=1.3),
        hovertemplate="Gamma: %{y:.6f}<extra></extra>"), row=2, col=1)
    fig_dg.add_hline(y=0, line_color=GRAY, line_width=1, row=2, col=1)

    fig_dg.update_layout(**_layout(height=300, margin=dict(l=54, r=16, t=28, b=44)),
                         legend=_LEG_TOP)
    fig_dg.update_xaxes(**_xax())
    fig_dg.update_yaxes(**_yax(), row=1, col=1)
    fig_dg.update_yaxes(**_yax(), row=2, col=1)
    st.plotly_chart(fig_dg, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 3 — RISK SIGNALS
# ─────────────────────────────────────────────────────────────────
with t3:
    _sec("strategy signal — slope carry + entropy filter", top=8)

    # Build signal: long SVIX when slope < 1 AND entropy above threshold
    signal_slope = (slope < 1.0)
    signal_entropy = (entropy_full >= entropy_threshold)
    combined_signal = (signal_slope & signal_entropy).astype(int)
    combined_signal_w = combined_signal.tail(N)

    fig_sig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06)

    # SVIX price
    fig_sig.add_trace(go.Scatter(
        x=svix_w.index, y=svix_w.values, mode="lines", name="SVIX",
        line=dict(color=BLUE, width=1.5),
        hovertemplate="SVIX: %{y:.2f}<extra></extra>"), row=1, col=1)

    # Shade signal periods on SVIX
    sig_aligned = combined_signal_w.reindex(svix_w.index).fillna(0)
    on_idx = sig_aligned[sig_aligned == 1].index
    if len(on_idx) > 0:
        fig_sig.add_trace(go.Scatter(
            x=on_idx, y=svix_w.reindex(on_idx).values,
            mode="markers", name="Signal ON",
            marker=dict(color=GREEN, size=2, opacity=0.4),
            hoverinfo="skip"), row=1, col=1)

    # Slope
    fig_sig.add_trace(go.Scatter(
        x=slope_w.index, y=slope_w.values, mode="lines", name="VIX/VIX3M",
        line=dict(color=GRAY, width=1.2),
        hovertemplate="Slope: %{y:.3f}<extra></extra>"), row=2, col=1)
    fig_sig.add_hline(y=1.0, line_color=RED, line_width=1, line_dash="dot", row=2, col=1)

    # Signal
    fig_sig.add_trace(go.Scatter(
        x=combined_signal_w.index, y=combined_signal_w.values,
        mode="lines", name="Signal",
        fill="tozeroy", fillcolor="rgba(5,150,105,0.15)",
        line=dict(color=GREEN, width=1),
        hovertemplate="Signal: %{y}<extra></extra>"), row=3, col=1)

    fig_sig.update_layout(**_layout(height=450, margin=dict(l=54, r=16, t=28, b=44)),
                          legend=_LEG_TOP)
    fig_sig.update_xaxes(**_xax())
    for r in [1, 2, 3]:
        fig_sig.update_yaxes(**_yax(), row=r, col=1)
    st.plotly_chart(fig_sig, use_container_width=True)

    # ── Signal stats ──
    signal_on_pct = float(combined_signal_w.mean() * 100)
    signal_now = "ON" if combined_signal.iloc[-1] == 1 else "OFF"

    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "signal now", signal_now,
         vc=GREEN if signal_now == "ON" else RED)
    _kpi(c2, f"% time ON ({sel_p})", f"{signal_on_pct:.0f}%",
         vc=GREEN if signal_on_pct > 60 else YELLOW)
    _kpi(c3, "slope < 1", "YES" if ratio_now < 1 else "NO",
         vc=GREEN if ratio_now < 1 else RED)
    _kpi(c4, f"entropy ≥ {entropy_threshold}", "YES" if entropy_now >= entropy_threshold else "NO",
         vc=GREEN if entropy_now >= entropy_threshold else RED)

    # ── VIX regime heatmap — what level of VIX produces which returns ──
    _sec("VIX level vs SVIX next-day return — partial dependence")
    vix_aligned = vix.reindex(svix.index)
    ret_next = svix.pct_change().shift(-1)
    scatter_df = pd.concat([vix_aligned.rename("vix"), ret_next.rename("ret")], axis=1).dropna()

    if len(scatter_df) > 50:
        scatter_df["vix_bin"] = pd.qcut(scatter_df["vix"], 10, duplicates="drop")
        pdp = scatter_df.groupby("vix_bin")["ret"].agg(["mean", "std", "count"])
        pdp["vix_mid"] = scatter_df.groupby("vix_bin")["vix"].mean()
        pdp = pdp.sort_values("vix_mid")

        fig_pdp = go.Figure()
        colors = [GREEN if m > 0 else RED for m in pdp["mean"]]
        fig_pdp.add_trace(go.Bar(
            x=pdp["vix_mid"].round(1).astype(str),
            y=pdp["mean"] * 100,
            marker=dict(color=colors, opacity=0.8,
                        line=dict(color=BORDER, width=1)),
            error_y=dict(type="data", array=(pdp["std"] * 100).values,
                         color=GRAY, thickness=1),
            hovertemplate="VIX ~%{x}: avg ret %{y:.2f}%<extra></extra>"))
        fig_pdp.update_layout(**_layout(height=250, hovermode="x",
                                        margin=dict(l=54, r=16, t=28, b=60)))
        fig_pdp.update_xaxes(**_xax(title="VIX level (decile midpoint)"))
        fig_pdp.update_yaxes(**_yax(ticksuffix="%", title="Avg next-day return"))
        st.plotly_chart(fig_pdp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 4 — DRAWDOWNS
# ─────────────────────────────────────────────────────────────────
with t4:
    dl, dr = st.columns([3, 1], gap="large")

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
        worst_dt = dd_svix.idxmin()
        fig_dd.add_trace(go.Scatter(
            x=[worst_dt], y=[max_dd], mode="markers+text",
            marker=dict(color=RED, size=7),
            text=[f" {max_dd:.1f}%"], textposition="bottom right",
            textfont=dict(size=9, color=RED, family=FONT),
            showlegend=False,
            hovertemplate=f"Max DD: {max_dd:.2f}% on {worst_dt:%Y-%m-%d}<extra></extra>"))
        fig_dd.update_layout(**_layout(height=280, margin=dict(l=54, r=16, t=28, b=44)))
        fig_dd.update_xaxes(**_xax()); fig_dd.update_yaxes(**_yax(ticksuffix="%"))
        st.plotly_chart(fig_dd, use_container_width=True)

    with dr:
        _sec("drawdown stats", top=8)
        dd_stats = [
            ("Max DD", f"{max_dd:.2f}%", RED),
            ("Current DD", f"{curr_dd:.2f}%", RED if curr_dd < -5 else TEXT),
            ("Max DD date", f"{worst_dt:%Y-%m-%d}", TEXT_DIM),
        ]
        for lbl, val, c in dd_stats:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {BORDER};font-size:12px'>"
                f"<span style='color:{TEXT_DIM}'>{lbl}</span>"
                f"<span style='color:{c};font-weight:500'>{val}</span></div>",
                unsafe_allow_html=True)

        # Top 5 drawdowns
        _sec("worst drawdowns", top=20)
        dd_full = _dd(svix)
        dd_periods = []
        dd_copy = dd_full.copy()
        for _ in range(5):
            if dd_copy.empty or dd_copy.min() >= -0.5:
                break
            idx_min = dd_copy.idxmin()
            val_min = float(dd_copy.loc[idx_min])
            dd_periods.append((idx_min.strftime("%Y-%m-%d"), f"{val_min:.1f}%"))
            # Blank out ±30 days around this drawdown
            mask_start = idx_min - pd.Timedelta(days=60)
            mask_end   = idx_min + pd.Timedelta(days=60)
            dd_copy.loc[mask_start:mask_end] = 0.0

        for dt, val in dd_periods:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                f"border-bottom:1px solid {BORDER};font-size:11px'>"
                f"<span style='color:{TEXT_DIM}'>{dt}</span>"
                f"<span style='color:{RED};font-weight:500'>{val}</span></div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 5 — VOLATILITY
# ─────────────────────────────────────────────────────────────────
with t5:
    _sec("SVIX realized volatility — 21D vs 63D annualized", top=8)
    fig_v = make_subplots(specs=[[{"secondary_y": True}]])
    fig_v.add_trace(go.Scatter(
        x=vol_63.index, y=vol_63.values, mode="lines", name="Vol 63D",
        fill="tozeroy", fillcolor="rgba(209,213,219,0.35)",
        line=dict(color=GRAY, width=1.2),
        hovertemplate="Vol 63D: %{y:.2f}%<extra></extra>"), secondary_y=False)
    fig_v.add_trace(go.Scatter(
        x=vol_21.index, y=vol_21.values, mode="lines", name="Vol 21D",
        line=dict(color=BLUE, width=1.8),
        hovertemplate="Vol 21D: %{y:.2f}%<extra></extra>"), secondary_y=False)

    vix_ral = vix.reindex(vol_21.index)
    fig_v.add_trace(go.Scatter(
        x=vix_ral.index, y=vix_ral.values, mode="lines", name="VIX",
        line=dict(color=RED, width=1.1, dash="dot"),
        hovertemplate="VIX: %{y:.2f}<extra></extra>"), secondary_y=True)
    fig_v.update_layout(**_layout(height=300, margin=dict(l=54, r=54, t=28, b=44)),
                        legend=_LEG_TOP)
    fig_v.update_xaxes(**_xax())
    fig_v.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="#f3f4f6",
                       tickfont=dict(size=9, color=BLUE), secondary_y=False)
    fig_v.update_yaxes(showgrid=False, tickfont=dict(size=9, color=RED), secondary_y=True)
    st.plotly_chart(fig_v, use_container_width=True)

    # Vol scatter vs VIX
    _sec("SVIX realized vol 21D vs VIX — risk map")
    vc_df = pd.concat([vol_21.rename("v"), vix.rename("vix")], axis=1).dropna()
    if len(vc_df) > 20:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=vc_df["vix"], y=vc_df["v"], mode="markers",
            marker=dict(color=vc_df["vix"],
                        colorscale=[[0, "#16a34a"], [0.35, "#ca8a04"],
                                    [0.65, "#ea580c"], [1, "#991b1b"]],
                        size=5, opacity=0.75,
                        colorbar=dict(title="VIX", thickness=10,
                                      tickfont=dict(size=8, family=FONT, color=TEXT_DIM))),
            hovertemplate="VIX: %{x:.1f} → Vol 21D: %{y:.1f}%<extra></extra>",
            showlegend=False))
        fig_sc.update_layout(**_layout(height=280, hovermode="closest",
                                       margin=dict(l=54, r=80, t=28, b=52)))
        fig_sc.update_xaxes(**_xax(showgrid=True, title="VIX level"))
        fig_sc.update_yaxes(**_yax(ticksuffix="%", title="Realized vol 21D"))
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 6 — VIX & VVIX
# ─────────────────────────────────────────────────────────────────
with t6:
    vl, vr = st.columns(2, gap="medium")

    with vl:
        _sec("VIX — spot level", top=8)
        fig_vix = go.Figure()
        fig_vix.add_hrect(y0=0, y1=15, fillcolor="rgba(5,150,105,0.06)", line_width=0)
        fig_vix.add_hrect(y0=20, y1=30, fillcolor="rgba(217,119,6,0.05)", line_width=0)
        fig_vix.add_hrect(y0=30, y1=90, fillcolor="rgba(220,38,38,0.05)", line_width=0)
        for lvl, col_l in [(15, GRAY), (20, YELLOW), (30, RED)]:
            fig_vix.add_hline(y=lvl, line_color=col_l, line_width=1, line_dash="dot")
        fig_vix.add_trace(go.Scatter(
            x=vix_w.index, y=vix_w.values, mode="lines", name="VIX",
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            line=dict(color=RED, width=1.7),
            hovertemplate="VIX: %{y:.2f}<extra></extra>"))
        fig_vix.add_annotation(
            x=vix_w.index[-1], y=vix_now,
            text=f"  {vix_now:.1f}", showarrow=False,
            font=dict(size=11, family=FONT,
                      color=RED if vix_now > 20 else YELLOW if vix_now > 15 else GREEN),
            xanchor="left")
        fig_vix.update_layout(**_layout(height=280, margin=dict(l=54, r=16, t=28, b=44)))
        fig_vix.update_xaxes(**_xax()); fig_vix.update_yaxes(**_yax())
        st.plotly_chart(fig_vix, use_container_width=True)

        # VIX distribution
        _sec("VIX distribution — where do we stand?")
        p10 = float(np.percentile(vix_w.dropna(), 10))
        p90 = float(np.percentile(vix_w.dropna(), 90))
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
                                       margin=dict(l=54, r=16, t=28, b=44)))
        fig_vd.update_xaxes(**_xax()); fig_vd.update_yaxes(**_yax())
        st.plotly_chart(fig_vd, use_container_width=True)

    with vr:
        _sec("VVIX — vol of vol", top=8)
        if vvix_w.empty:
            st.info("VVIX not available.")
        else:
            fig_vv = go.Figure()
            fig_vv.add_hline(y=100, line_color=YELLOW, line_width=1, line_dash="dot",
                             annotation_text="100",
                             annotation_font=dict(size=9, family=FONT, color=YELLOW))
            fig_vv.add_hline(y=120, line_color=RED, line_width=1, line_dash="dot",
                             annotation_text="120",
                             annotation_font=dict(size=9, family=FONT, color=RED))
            fig_vv.add_trace(go.Scatter(
                x=vvix_w.index, y=vvix_w.values, mode="lines", name="VVIX",
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
                line=dict(color="#7c3aed", width=1.7),
                hovertemplate="VVIX: %{y:.2f}<extra></extra>"))
            fig_vv.add_annotation(
                x=vvix_w.index[-1], y=vvix_now,
                text=f"  {vvix_now:.1f}", showarrow=False,
                font=dict(size=11, family=FONT,
                          color=RED if vvix_now > 120 else YELLOW if vvix_now > 100 else TEXT),
                xanchor="left")
            fig_vv.update_layout(**_layout(height=280, margin=dict(l=54, r=16, t=28, b=44)))
            fig_vv.update_xaxes(**_xax()); fig_vv.update_yaxes(**_yax())
            st.plotly_chart(fig_vv, use_container_width=True)

        # VVIX/VIX ratio
        _sec("VVIX / VIX ratio — tail risk gauge")
        vvix_vix_w = vvix_vix_ratio.tail(N)
        if not vvix_vix_w.empty:
            fig_vr = go.Figure()
            fig_vr.add_trace(go.Scatter(
                x=vvix_vix_w.index, y=vvix_vix_w.values, mode="lines",
                name="VVIX/VIX",
                line=dict(color="#0891b2", width=1.5),
                fill="tozeroy", fillcolor="rgba(8,145,178,0.06)",
                hovertemplate="VVIX/VIX: %{y:.2f}<extra></extra>"))
            med = float(vvix_vix_w.median())
            fig_vr.add_hline(y=med, line_color=GRAY, line_width=1, line_dash="dot",
                             annotation_text=f"median {med:.1f}",
                             annotation_font=dict(size=8, color=GRAY))
            fig_vr.update_layout(**_layout(height=220, margin=dict(l=54, r=16, t=28, b=44)))
            fig_vr.update_xaxes(**_xax()); fig_vr.update_yaxes(**_yax())
            st.plotly_chart(fig_vr, use_container_width=True)