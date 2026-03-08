"""
Compare — Professional universe analytics
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core.db import load_assets, load_multi_prices, load_returns_snapshot
from core.formatting import (
    fmt_pct, FONT,
    GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID, BG, BG2, BG3, GRAY, BLUE, YELLOW
)

st.title("compare")

# ── Period pill selector ──────────────────────────────────────────
PERIODS = ["1M","3M","6M","1Y","2Y","3Y","5Y"]
if "cmp_period" not in st.session_state:
    st.session_state.cmp_period = "1Y"

st.markdown("""
<style>
.period-row div[data-testid="stButton"] > button {
    height: 28px !important;
    padding: 0 14px !important;
    border-radius: 14px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border: 1.5px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #9ca3af !important;
    box-shadow: none !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    transition: all 0.1s !important;
}
.period-row div[data-testid="stButton"] > button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #eff6ff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='period-row'>", unsafe_allow_html=True)
_cols = st.columns([1.4] + [0.75]*len(PERIODS) + [8], gap="small")
_cols[0].markdown(
    f"<div style='height:28px;display:flex;align-items:center;"
    f"font-family:{FONT};font-size:9px;font-weight:700;color:#9ca3af;"
    f"letter-spacing:0.1em;text-transform:uppercase'>Period</div>",
    unsafe_allow_html=True)
for i, _p in enumerate(PERIODS):
    if _p == st.session_state.cmp_period:
        _cols[i+1].markdown(
            f"<div style='height:28px;padding:0 14px;border-radius:14px;"
            f"background:#2563eb;display:inline-flex;align-items:center;"
            f"font-family:{FONT};font-size:11px;font-weight:700;"
            f"letter-spacing:0.06em;color:#fff'>{_p}</div>",
            unsafe_allow_html=True)
    else:
        if _cols[i+1].button(_p, key=f"cmp_{_p}"):
            st.session_state.cmp_period = _p
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

period = st.session_state.cmp_period

# ══════════════════════════════════════════════════════════════════
# A. UNIVERSE CATALOG
# ══════════════════════════════════════════════════════════════════
CATEGORY_LABELS = {
    "country":       "Countries",
    "sector-us":     "US Sectors",
    "sector-world":  "World Sectors",
    "sector-ca":     "Canada Sectors",
    "style":         "Styles & Factors",
    "thematic":      "Thematics",
    "commodity":     "Commodities",
    "benchmark":     "Benchmarks",
    "vol-etf":       "Vol ETFs",
}
CATEGORY_ORDER = [
    "country", "sector-us", "sector-world", "sector-ca",
    "style", "thematic", "commodity", "benchmark", "vol-etf",
]

_STRIP = [
    "iShares MSCI ","iShares Core S&P/TSX Capped ",
    "iShares S&P/TSX Capped ","iShares Global ",
    "iShares ","SPDR Bloomberg ","SPDR S&P ","SPDR ",
    "ProShares Ultra ","ProShares Short ","ProShares ",
    "Vanguard ","Invesco ","VanEck ",
]
def _shorten(name):
    for p in _STRIP:
        if name.startswith(p):
            name = name[len(p):]
            break
    return name[:28].rstrip()

assets = load_assets()

_sym_name    = {r["symbol"]: _shorten(r["name"]) for _, r in assets.iterrows()}
_sym_country = {r["symbol"]: (r.get("country_cd") or "") for _, r in assets.iterrows()}
_sym_full    = {r["symbol"]: r["name"] for _, r in assets.iterrows()}

if "category" in assets.columns:
    UNIVERSE_CATALOG = {}
    for cat in CATEGORY_ORDER:
        syms = assets[assets["category"] == cat]["symbol"].tolist()
        if syms:
            UNIVERSE_CATALOG[CATEGORY_LABELS[cat]] = syms
else:
    UNIVERSE_CATALOG = {
        ac: grp["symbol"].tolist()
        for ac, grp in assets[assets["asset_class"] != "Fixed Income"].groupby("asset_class")
    }

investable = assets[
    (assets["asset_class"] != "Fixed Income") &
    (assets["asset_type"]  != "index")
].copy()

# ══════════════════════════════════════════════════════════════════
# B. SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    group_name = st.selectbox("Universe", list(UNIVERSE_CATALOG.keys()))
    corr_freq  = st.selectbox("Correlation frequency", ["1D","1W","1M"], index=0)
    display_as = st.radio("Labels", ["Name","Ticker"], horizontal=True)
    st.markdown("<hr style='border-color:#e5e7eb;margin:12px 0'>", unsafe_allow_html=True)
    benchmark  = st.selectbox("Benchmark", ["None","SPY","QQQ","VT"], index=1)

# ══════════════════════════════════════════════════════════════════
# C. UNIVERSE RESOLUTION
# ══════════════════════════════════════════════════════════════════
symbols = [s for s in UNIVERSE_CATALOG[group_name] if s in investable["symbol"].values]
if not symbols:
    st.warning(f"No active symbols for **{group_name}**."); st.stop()

def lbl(sym):
    if display_as == "Ticker": return sym
    if group_name == "Countries":
        c = _sym_country.get(sym,"")
        return c if c else _sym_name.get(sym, sym)
    return _sym_name.get(sym, sym)

def tooltip(sym): return _sym_full.get(sym, sym)

# ══════════════════════════════════════════════════════════════════
# D. PRICE DATA
# ══════════════════════════════════════════════════════════════════
PERIOD_BARS = {"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504,"3Y":756,"5Y":1260}
CORR_SHIFT  = {"1D":1,"1W":5,"1M":21}
n_bars      = PERIOD_BARS[period]

bm_sym    = benchmark if benchmark != "None" else None
load_syms = list(dict.fromkeys(symbols + ([bm_sym] if bm_sym else [])))

px_all = load_multi_prices(load_syms)
if px_all is None or px_all.empty:
    st.warning("No price data."); st.stop()

px_all = px_all.sort_index()
px_win = px_all.reindex(columns=symbols).tail(n_bars).copy()
px_win = px_win.dropna(axis=1, how="all")
min_obs = max(5, n_bars // 2)
px_win  = px_win.loc[:, px_win.notna().sum() >= min_obs]
active  = px_win.columns.tolist()
if not active:
    st.warning("Not enough price history."); st.stop()

daily    = px_win.pct_change(1).ffill(limit=5)
ret_corr = px_win.pct_change(CORR_SHIFT[corr_freq]).dropna(how="all")
cum_ret  = (1 + daily.fillna(0)).cumprod() - 1
last_cum = cum_ret.iloc[-1]

first_valid = px_win.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
px_norm     = px_win.div(first_valid) * 100

def _dd(prices):
    p = prices.ffill()
    return p / p.expanding().max() - 1

dd_mat   = pd.DataFrame({s: _dd(px_win[s]) for s in active}).reindex(px_win.index)
roll_vol = daily.rolling(21).std() * np.sqrt(252) * 100
vol_1m   = daily.tail(21).std() * np.sqrt(252) * 100
vol_3m   = daily.tail(63).std() * np.sqrt(252) * 100
max_dd   = dd_mat.min()
curr_dd  = dd_mat.iloc[-1]
sharpe   = (daily.mean() / daily.std()) * np.sqrt(252)
calmar   = last_cum / max_dd.abs().replace(0, np.nan)

beta = pd.Series(np.nan, index=active)
if bm_sym and bm_sym in px_all.columns:
    bm_prices = px_all[bm_sym].dropna()
    if len(bm_prices) > 30:
        bm_ret = bm_prices.pct_change(1)
        for s in active:
            s_ret = px_all[s].pct_change(1).dropna() if s in px_all.columns else pd.Series()
            aligned = pd.concat([s_ret, bm_ret], axis=1, join="inner").dropna()
            aligned.columns = ["s","bm"]
            if len(aligned) > 20:
                bm_var = aligned["bm"].var()
                if bm_var > 1e-10:
                    beta[s] = aligned["s"].cov(aligned["bm"]) / bm_var

snap_df = load_returns_snapshot(active)

# ══════════════════════════════════════════════════════════════════
# E. UTILITIES
# ══════════════════════════════════════════════════════════════════
LINE_COLORS = [
    "#2563eb","#d97706","#059669","#dc2626","#7c3aed",
    "#0891b2","#db2777","#16a34a","#ea580c","#2dd4bf",
    "#6366f1","#f59e0b","#10b981","#ef4444","#a78bfa",
    "#34d399","#fb923c","#60a5fa",
]
def _c(i): return LINE_COLORS[i % len(LINE_COLORS)]

def _hex_rgba(h, a=0.12):
    h = h.lstrip("#")
    r,g,b = int(h[:2],16),int(h[2:4],16),int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"

# Pre-defined legend configs — pass as legend= to update_layout
_LEGEND_SIDE = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=9, color=TEXT_DIM),
                    orientation="v", x=1.02, y=1, xanchor="left")
_LEGEND_TOP  = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=9, color=TEXT_DIM),
                    orientation="h", x=0, y=1.08)

def _layout(**kw):
    """Base Plotly layout — NO legend key so callers set it freely."""
    base = dict(
        paper_bgcolor=BG2,
        plot_bgcolor=BG2,
        font=dict(family=FONT, color=TEXT_DIM, size=10),
        margin=dict(l=58, r=20, t=24, b=52),
        hovermode="x unified",
    )
    base.update(kw)
    return base

def _xax(**kw):
    base = dict(showgrid=False, linecolor="#e5e7eb", zeroline=False,
                tickfont=dict(size=9, color=TEXT_DIM))
    base.update(kw)
    return base

def _yax(**kw):
    base = dict(showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
                zeroline=False, tickfont=dict(size=9, color=TEXT_DIM))
    base.update(kw)
    return base

def _sec(text, top=32):
    st.markdown(
        f"<p style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
        f"text-transform:uppercase;letter-spacing:0.12em;"
        f"margin:{top}px 0 10px 0'>{text}</p>",
        unsafe_allow_html=True)

def _kpi(col, label, value, sub="", vc=TEXT):
    if vc == GREEN:    bg, accent, tc = "#f0fdf4", "#16a34a", "#14532d"
    elif vc == RED:    bg, accent, tc = "#fef2f2", "#dc2626", "#7f1d1d"
    elif vc == YELLOW: bg, accent, tc = "#fffbeb", "#d97706", "#78350f"
    else:              bg, accent, tc = "#f8fafc", "#64748b", "#1e293b"
    col.markdown(
        f"<div style='background:{bg};border-radius:10px;padding:14px 16px 12px;"
        f"border-top:3px solid {accent}'>"
        f"<div style='font-family:{FONT};font-size:8px;font-weight:700;"
        f"letter-spacing:0.14em;text-transform:uppercase;color:{accent};"
        f"margin-bottom:8px'>{label}</div>"
        f"<div style='font-family:{FONT};font-size:20px;font-weight:300;"
        f"color:{tc};line-height:1;letter-spacing:-0.01em'>{value}</div>"
        f"<div style='font-family:{FONT};font-size:10px;color:{accent};"
        f"opacity:0.7;margin-top:6px;white-space:nowrap;overflow:hidden;"
        f"text-overflow:ellipsis'>{sub}</div>"
        f"</div>", unsafe_allow_html=True)

_TS = [
    {"selector":"th","props":[
        ("background",BG3),("color",TEXT_DIM),("font-family",FONT),
        ("font-size","10px"),("font-weight","600"),("text-transform","uppercase"),
        ("letter-spacing","0.06em"),("border-bottom",f"1px solid {BORDER}"),
        ("border-top",f"1px solid {BORDER}"),("padding","7px 12px"),("white-space","nowrap"),
    ]},
    {"selector":"td","props":[
        ("background",BG2),("font-family",FONT),("font-size","12px"),
        ("color",TEXT),("border-bottom",f"1px solid {BORDER}"),("padding","5px 12px"),
    ]},
    {"selector":"tr:hover td","props":[("background",BG3)]},
]

def _cr(v):
    if pd.isna(v): return f"color:{GRAY}"
    return f"color:{GREEN}" if v > 0 else f"color:{RED}"
def _cv(v):
    if pd.isna(v): return f"color:{GRAY}"
    if v > 35: return f"color:{RED};font-weight:600"
    if v > 22: return f"color:{YELLOW}"
    return f"color:{TEXT}"
def _csh(v):
    if pd.isna(v): return f"color:{GRAY}"
    if v > 1.2: return f"color:{GREEN};font-weight:600"
    if v > 0.5: return f"color:{GREEN}"
    if v < 0:   return f"color:{RED}"
    return f"color:{YELLOW}"
def _cdd(v):
    if pd.isna(v): return f"color:{GRAY}"
    if v < -0.30: return f"color:{RED};font-weight:600"
    if v < -0.12: return f"color:{YELLOW}"
    return f"color:{TEXT}"
def _cb(v):
    if pd.isna(v): return f"color:{GRAY}"
    if v > 1.5: return f"color:{RED}"
    if v > 1.0: return f"color:{YELLOW}"
    if v < 0:   return f"color:{GREEN}"
    return f"color:{TEXT}"

# ══════════════════════════════════════════════════════════════════
# F. SUMMARY BLOCK
# ══════════════════════════════════════════════════════════════════
valid_cum  = last_cum.dropna()
best_s     = valid_cum.idxmax()        if not valid_cum.empty else None
worst_s    = valid_cum.idxmin()        if not valid_cum.empty else None
best_sh_s  = sharpe.dropna().idxmax() if not sharpe.dropna().empty else None
worst_dd_s = max_dd.dropna().idxmin() if not max_dd.dropna().empty else None
hitvol_s   = vol_3m.dropna().idxmax() if not vol_3m.dropna().empty else None
n_pos      = int((valid_cum > 0).sum()) if not valid_cum.empty else 0
med_ret    = float(valid_cum.median())  if not valid_cum.empty else None

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
_kpi(c1, f"best ({period})",   fmt_pct(valid_cum[best_s])  if best_s else "—",  lbl(best_s)  if best_s else "", GREEN)
_kpi(c2, f"worst ({period})",  fmt_pct(valid_cum[worst_s]) if worst_s else "—", lbl(worst_s) if worst_s else "", RED)
_kpi(c3, "median",             fmt_pct(med_ret) if med_ret is not None else "—", "")
_kpi(c4, "positive",           str(n_pos), f"of {len(valid_cum)}")
_kpi(c5, "best sharpe",        f"{sharpe[best_sh_s]:.2f}" if best_sh_s else "—", lbl(best_sh_s) if best_sh_s else "", GREEN)
_kpi(c6, "worst drawdown",     f"{max_dd[worst_dd_s]*100:.2f}%" if worst_dd_s else "—", lbl(worst_dd_s) if worst_dd_s else "", RED)
_kpi(c7, "most volatile",      f"{vol_3m[hitvol_s]:.2f}%" if hitvol_s else "—", lbl(hitvol_s) if hitvol_s else "", YELLOW)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Summary table ─────────────────────────────────────────────────
sorted_syms = last_cum.sort_values(ascending=False).index.tolist()
ret_col     = f"Ret {period}"
beta_col    = f"β {bm_sym}" if bm_sym else None

summ_rows = []
for s in sorted_syms:
    row = {"Asset": lbl(s), ret_col: last_cum.get(s),
           "Vol 3M": vol_3m.get(s), "Max DD": max_dd.get(s),
           "Sharpe": sharpe.get(s), "Calmar": calmar.get(s)}
    if beta_col: row[beta_col] = beta.get(s)
    summ_rows.append(row)

summ_df = pd.DataFrame(summ_rows).reset_index(drop=True)
fmt_s = {
    ret_col:  lambda x: fmt_pct(x)       if pd.notna(x) else "—",
    "Vol 3M": lambda x: f"{x:.2f}%"     if pd.notna(x) else "—",
    "Max DD": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—",
    "Sharpe": lambda x: f"{x:.2f}"      if pd.notna(x) else "—",
    "Calmar": lambda x: f"{x:.2f}"      if pd.notna(x) else "—",
}
if beta_col: fmt_s[beta_col] = lambda x: f"{x:.2f}" if pd.notna(x) else "—"

s_obj = summ_df.style.set_table_styles(_TS).format(fmt_s)
for col_n, fn in [(ret_col,_cr),("Vol 3M",_cv),("Max DD",_cdd),("Sharpe",_csh),("Calmar",_csh)]:
    if col_n in summ_df.columns: s_obj = s_obj.applymap(fn, subset=[col_n])
if beta_col and beta_col in summ_df.columns:
    s_obj = s_obj.applymap(_cb, subset=[beta_col])
st.dataframe(s_obj, use_container_width=True,
             height=min(58 + len(summ_df)*35, 420), hide_index=True)

# ── Returns heatmap ───────────────────────────────────────────────
if not snap_df.empty:
    _sec("period returns", top=24)
    snap_cols = {"r1d":"1D","r1w":"1W","r1m":"1M","r3m":"3M","rytd":"YTD","r1y":"1Y"}
    s_avail   = {k:v for k,v in snap_cols.items() if k in snap_df.columns}
    snap_show = snap_df.reindex(sorted_syms)[list(s_avail.keys())].rename(columns=s_avail)
    snap_show.index = [lbl(s) for s in snap_show.index]
    z_vals = snap_show.values * 100
    z_text = [[f"{v:+.2f}%" if not np.isnan(v) else "" for v in row] for row in z_vals]
    h_heat = max(160, len(snap_show)*26 + 60)
    fig_h = go.Figure(go.Heatmap(
        z=z_vals, x=snap_show.columns.tolist(), y=snap_show.index.tolist(),
        colorscale=[[0,"#fef2f2"],[0.35,"#dc2626"],[0.5,"#f8fafc"],[0.65,"#059669"],[1,"#f0fdf4"]],
        zmid=0, text=z_text, texttemplate="%{text}",
        textfont=dict(size=9, family=FONT),
        hovertemplate="%{y} — %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=8,family=FONT,color=TEXT_DIM)),
    ))
    fig_h.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2, height=h_heat,
        font=dict(family=FONT, size=9, color=TEXT_DIM),
        margin=dict(l=130, r=60, t=10, b=30),
    )
    fig_h.update_xaxes(showgrid=False, tickfont=dict(size=10, color=TEXT_MID))
    fig_h.update_yaxes(showgrid=False, tickfont=dict(size=10, color=TEXT_MID), autorange="reversed")
    st.plotly_chart(fig_h, width="stretch")

# ══════════════════════════════════════════════════════════════════
# G. TABS
# ══════════════════════════════════════════════════════════════════
tab_corr, tab_perf, tab_dd, tab_vol, tab_risk = st.tabs([
    "Correlation", "Performance", "Drawdowns", "Volatility", "Risk / Return",
])

# ── TAB 1 — CORRELATION ───────────────────────────────────────────
with tab_corr:
    corr = ret_corr[active].corr()
    lbls = [lbl(s) for s in active]

    _sec(f"correlation matrix — {corr_freq} returns · {period}")
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=lbls, y=lbls,
        colorscale=[[0,"#fef2f2"],[0.25,"#dc2626"],[0.5,"#f8fafc"],[0.75,"#2563eb"],[1,"#eff6ff"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=8, family=FONT),
        hoverongaps=False,
        hovertemplate="%{y} × %{x}  ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.8, tickvals=[-1,-0.5,0,0.5,1],
                      tickfont=dict(size=9,family=FONT,color=TEXT_DIM)),
    ))
    h_c = max(400, len(active)*28 + 80)
    fig_c.update_layout(**_layout(height=h_c, hovermode=False,
                                   margin=dict(l=120,r=50,t=16,b=120)),
                        legend=_LEGEND_SIDE)
    fig_c.update_xaxes(showgrid=False, tickangle=-42, tickfont=dict(size=9,color=TEXT_MID))
    fig_c.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=9,color=TEXT_MID))
    st.plotly_chart(fig_c, width="stretch")

    pairs = [(active[i],active[j],corr.iloc[i,j])
             for i in range(len(active)) for j in range(i+1,len(active))]
    pf = pd.DataFrame(pairs, columns=["A","B","ρ"]).dropna()
    pf["A"] = pf["A"].map(lbl); pf["B"] = pf["B"].map(lbl)

    def _c_rho(v):
        if pd.isna(v): return f"color:{GRAY}"
        if v > 0.75:   return f"color:{RED};font-weight:600"
        if v > 0.45:   return f"color:{YELLOW}"
        if v < -0.2:   return f"color:{GREEN}"
        return f"color:{TEXT}"

    ca, cb = st.columns(2)
    for col_w, df_p, ttl in [(ca, pf.nlargest(12,"ρ"),"most correlated"),
                              (cb, pf.nsmallest(8,"ρ"), "most diverging")]:
        col_w.markdown(f"<p style='font-family:{FONT};font-size:9px;color:{TEXT_DIM};"
                       f"text-transform:uppercase;letter-spacing:0.1em;margin:24px 0 8px'>{ttl}</p>",
                       unsafe_allow_html=True)
        col_w.dataframe(df_p.style.applymap(_c_rho,subset=["ρ"])
                        .format({"ρ":"{:.2f}"}).set_table_styles(_TS),
                        width='stretch', hide_index=True)

    avg_corr   = corr.apply(lambda r: r.drop(r.name).mean())
    avg_corr_s = avg_corr.sort_values(ascending=False)
    _sec("average pairwise correlation")
    fig_ac = go.Figure(go.Bar(
        x=[lbl(s) for s in avg_corr_s.index], y=avg_corr_s.values,
        marker=dict(color=avg_corr_s.values,
                    colorscale=[[0,"#2563eb"],[0.5,"#d97706"],[1,"#dc2626"]],
                    line=dict(width=0)),
        text=[f"{v:.2f}" for v in avg_corr_s.values],
        textposition="outside", textfont=dict(size=8,family=FONT,color=TEXT_DIM),
        hovertemplate="%{x}: avg ρ = %{y:.2f}<extra></extra>",
    ))
    fig_ac.update_layout(**_layout(height=220, hovermode="x",
                                   margin=dict(l=52,r=16,t=16,b=60)),
                         legend=_LEGEND_SIDE)
    fig_ac.update_xaxes(**_xax(tickangle=-38))
    fig_ac.update_yaxes(**_yax())
    st.plotly_chart(fig_ac, width="stretch")


# ── TAB 2 — PERFORMANCE ───────────────────────────────────────────
with tab_perf:
    ordered_perf = last_cum.sort_values(ascending=False).index.tolist()

    _sec(f"total return ranking — {period}")
    fig_bar = go.Figure(go.Bar(
        x=last_cum.reindex(ordered_perf).values * 100,
        y=[lbl(s) for s in ordered_perf],
        orientation="h",
        marker=dict(color=[GREEN if v>=0 else RED
                           for v in last_cum.reindex(ordered_perf).values],
                    line=dict(width=0)),
        text=[f"{v:+.2f}%" for v in last_cum.reindex(ordered_perf).values * 100],
        textposition="outside", textfont=dict(size=9,family=FONT,color=TEXT_DIM),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig_bar.update_layout(**_layout(height=max(280,len(active)*24+60),
                                    hovermode="y unified",
                                    margin=dict(l=140,r=90,t=16,b=36)),
                          legend=_LEGEND_SIDE)
    fig_bar.update_xaxes(**_xax(showgrid=True, ticksuffix="%"))
    fig_bar.update_yaxes(**_yax(showgrid=False, tickfont=dict(size=10,color=TEXT_MID)))
    st.plotly_chart(fig_bar, width="stretch")

    _sec("rebased to 100")
    fig_norm = go.Figure()
    fig_norm.add_hline(y=100, line_color="#e5e7eb", line_width=1)
    for i, sym in enumerate(ordered_perf):
        if sym not in px_norm.columns: continue
        s = px_norm[sym].dropna()
        fig_norm.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=lbl(sym),
                                      line=dict(color=_c(i),width=1.5),
                                      hovertemplate=f"{lbl(sym)}: %{{y:.1f}}<extra></extra>"))
    fig_norm.update_layout(**_layout(height=340, margin=dict(l=58,r=160,t=20,b=48)),
                           legend=_LEGEND_SIDE)
    fig_norm.update_xaxes(**_xax())
    fig_norm.update_yaxes(**_yax())
    st.plotly_chart(fig_norm, width="stretch")

    _sec("cumulative return (%)")
    fig_cum = go.Figure()
    fig_cum.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    for i, sym in enumerate(ordered_perf):
        if sym not in cum_ret.columns: continue
        s = cum_ret[sym].dropna() * 100
        fig_cum.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=lbl(sym),
                                     line=dict(color=_c(i),width=1.5),
                                     hovertemplate=f"{lbl(sym)}: %{{y:.2f}}%<extra></extra>"))
    fig_cum.update_layout(**_layout(height=320, margin=dict(l=58,r=160,t=20,b=48)),
                          legend=_LEGEND_SIDE)
    fig_cum.update_xaxes(**_xax())
    fig_cum.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_cum, width="stretch")


# ── TAB 3 — DRAWDOWNS ────────────────────────────────────────────
with tab_dd:
    dd_ordered = max_dd.sort_values().index.tolist()

    _sec(f"underwater equity curve — {period}")
    fig_uw = go.Figure()
    fig_uw.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    for i, sym in enumerate(dd_ordered):
        if sym not in dd_mat.columns: continue
        s = dd_mat[sym].dropna() * 100
        color = _c(i)
        fig_uw.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=lbl(sym),
            line=dict(color=color, width=1.3),
            fill="tozeroy", fillcolor=_hex_rgba(color, 0.07),
            hovertemplate=f"{lbl(sym)}: %{{y:.2f}}%<extra></extra>",
        ))
    fig_uw.update_layout(**_layout(height=360, margin=dict(l=58,r=160,t=20,b=48)),
                         legend=_LEGEND_SIDE)
    fig_uw.update_xaxes(**_xax())
    fig_uw.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_uw, width="stretch")

    _sec("max drawdown — ranked")
    fig_dbar = go.Figure(go.Bar(
        x=[lbl(s) for s in dd_ordered],
        y=[max_dd[s]*100 for s in dd_ordered],
        marker=dict(color=[max_dd[s]*100 for s in dd_ordered],
                    colorscale=[[0,"#dc2626"],[0.5,"#d97706"],[1,"#f3f4f6"]],
                    cmin=-60, cmax=0, line=dict(width=0)),
        text=[f"{max_dd[s]*100:.2f}%" for s in dd_ordered],
        textposition="outside", textfont=dict(size=9,family=FONT,color=TEXT_DIM),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig_dbar.update_layout(**_layout(height=240, hovermode="x",
                                     margin=dict(l=52,r=16,t=16,b=60)),
                           legend=_LEGEND_SIDE)
    fig_dbar.update_xaxes(**_xax(tickangle=-38))
    fig_dbar.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_dbar, width="stretch")

    _sec("drawdown detail")
    dd_tbl = pd.DataFrame({
        "Asset":        [lbl(s) for s in dd_ordered],
        "Max DD":       [max_dd[s] for s in dd_ordered],
        "Current DD":   [curr_dd.get(s,np.nan) for s in dd_ordered],
        f"Ret {period}":[last_cum.get(s,np.nan) for s in dd_ordered],
    })
    st.dataframe(
        dd_tbl.style
        .applymap(_cdd, subset=["Max DD","Current DD"])
        .applymap(_cr,  subset=[f"Ret {period}"])
        .format({"Max DD": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—",
                 "Current DD": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—",
                 f"Ret {period}": lambda x: fmt_pct(x) if pd.notna(x) else "—"})
        .set_table_styles(_TS),
        use_container_width=True,
        height=min(60+len(dd_tbl)*34,520), hide_index=True)


# ── TAB 4 — VOLATILITY ────────────────────────────────────────────
with tab_vol:
    vol_ordered = vol_3m.sort_values(ascending=False).index.tolist()

    _sec("rolling 21D volatility — annualized")
    fig_rv = go.Figure()
    for i, sym in enumerate(vol_ordered):
        if sym not in roll_vol.columns: continue
        s = roll_vol[sym].dropna()
        fig_rv.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=lbl(sym),
                                    line=dict(color=_c(i),width=1.3),
                                    hovertemplate=f"{lbl(sym)}: %{{y:.2f}}%<extra></extra>"))
    fig_rv.update_layout(**_layout(height=340, margin=dict(l=58,r=160,t=20,b=48)),
                         legend=_LEGEND_SIDE)
    fig_rv.update_xaxes(**_xax())
    fig_rv.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_rv, width="stretch")

    _sec("vol snapshot — 1M vs 3M")
    xv = [lbl(s) for s in vol_ordered]
    fig_vbar = go.Figure()
    fig_vbar.add_trace(go.Bar(name="Vol 3M", x=xv,
        y=[vol_3m.get(s,np.nan) for s in vol_ordered],
        marker=dict(color=BG3, line=dict(color=BORDER,width=1)),
        hovertemplate="%{x} — 3M: %{y:.2f}%<extra></extra>"))
    fig_vbar.add_trace(go.Bar(name="Vol 1M", x=xv,
        y=[vol_1m.get(s,np.nan) for s in vol_ordered],
        marker=dict(color=BLUE, line=dict(width=0)),
        hovertemplate="%{x} — 1M: %{y:.2f}%<extra></extra>"))
    fig_vbar.update_layout(**_layout(height=260, hovermode="x",
                                     margin=dict(l=52,r=16,t=32,b=60),
                                     barmode="overlay"),
                           legend=_LEGEND_TOP)
    fig_vbar.update_xaxes(**_xax(tickangle=-38))
    fig_vbar.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_vbar, width="stretch")

    _sec("vol 1M vs 3M — above diagonal = volatility expanding")
    vs = pd.DataFrame({"lbl":[lbl(s) for s in active],
                        "v1m":vol_1m.reindex(active).values,
                        "v3m":vol_3m.reindex(active).values}).dropna()
    vs["delta"] = vs["v1m"] - vs["v3m"]
    rng_max = max(vs["v3m"].max(), vs["v1m"].max()) * 1.08
    fig_vs = go.Figure()
    fig_vs.add_trace(go.Scatter(x=[0,rng_max], y=[0,rng_max], mode="lines",
                                line=dict(color="#e5e7eb",width=1,dash="dot"),
                                showlegend=False, hoverinfo="skip"))
    fig_vs.add_trace(go.Scatter(
        x=vs["v3m"], y=vs["v1m"], mode="markers+text",
        marker=dict(color=vs["delta"],
                    colorscale=[[0,"#059669"],[0.5,"#f8fafc"],[1,"#dc2626"]],
                    cmid=0, size=11, line=dict(width=1,color="#ffffff"),
                    colorbar=dict(title="1M−3M",thickness=10,
                                  tickfont=dict(size=8,family=FONT,color=TEXT_DIM))),
        text=vs["lbl"], textposition="top center",
        textfont=dict(size=8,family=FONT,color=TEXT_DIM),
        hovertemplate="%{text}<br>Vol 3M: %{x:.2f}%<br>Vol 1M: %{y:.2f}%<extra></extra>",
        showlegend=False))
    fig_vs.update_layout(**_layout(height=340, hovermode="closest",
                                   margin=dict(l=56,r=80,t=20,b=52)),
                         legend=_LEGEND_SIDE)
    fig_vs.update_xaxes(**_xax(showgrid=True, title="Vol 3M (%)", ticksuffix="%"))
    fig_vs.update_yaxes(**_yax(title="Vol 1M (%)", ticksuffix="%"))
    st.plotly_chart(fig_vs, width="stretch")


# ── TAB 5 — RISK / RETURN ─────────────────────────────────────────
with tab_risk:
    _sec(f"risk / return  ·  bubble size ∝ |sharpe|  ·  {period}")
    sc = pd.DataFrame({
        "sym":  active, "lbl": [lbl(s) for s in active],
        "tip":  [tooltip(s) for s in active],
        "vol":  vol_3m.reindex(active).values,
        "ret":  last_cum.reindex(active).values * 100,
        "dd":   max_dd.reindex(active).values * 100,
        "sh":   sharpe.reindex(active).values,
        "beta": beta.reindex(active).values,
    }).dropna(subset=["vol","ret"])
    sc["size"] = (sc["sh"].abs().clip(0,3) * 4 + 7).clip(7,20)

    fig_sc = go.Figure()
    fig_sc.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig_sc.add_vline(x=sc["vol"].mean(), line_color="#e5e7eb",
                     line_width=1, line_dash="dot")
    fig_sc.add_trace(go.Scatter(
        x=sc["vol"], y=sc["ret"], mode="markers+text",
        marker=dict(
            color=sc["ret"],
            colorscale=[[0,"#fef2f2"],[0.45,"#dc2626"],[0.5,"#f8fafc"],[0.55,"#059669"],[1,"#f0fdf4"]],
            cmid=0, size=sc["size"], line=dict(width=1,color="#ffffff"),
            colorbar=dict(title="Ret%",thickness=10,
                          tickfont=dict(size=8,family=FONT,color=TEXT_DIM),ticksuffix="%"),
        ),
        text=sc["lbl"], textposition="top center",
        textfont=dict(size=8,family=FONT,color=TEXT_DIM),
        hovertemplate=(
            "<b>%{text}</b><br>Return: %{y:.2f}%<br>"
            "Vol 3M: %{x:.2f}%<br>Max DD: %{customdata[0]:.2f}%<br>"
            "Sharpe: %{customdata[1]:.2f}<extra></extra>"
        ),
        customdata=sc[["dd","sh"]].values, showlegend=False))
    fig_sc.update_layout(**_layout(height=460, hovermode="closest",
                                   margin=dict(l=58,r=80,t=24,b=56)),
                         legend=_LEGEND_SIDE)
    fig_sc.update_xaxes(**_xax(showgrid=True, title="Volatility 3M (%)", ticksuffix="%"))
    fig_sc.update_yaxes(**_yax(title=f"Return {period} (%)", ticksuffix="%"))
    st.plotly_chart(fig_sc, width="stretch")

    _sec("risk metrics — sorted by sharpe")
    risk_rows = []
    for s in sharpe.sort_values(ascending=False).index:
        row = {"Asset": lbl(s),
               "Last": f"{px_win[s].dropna().iloc[-1]:.2f}" if s in px_win.columns else "—",
               f"Ret {period}": last_cum.get(s),
               "Vol 1M": vol_1m.get(s), "Vol 3M": vol_3m.get(s),
               "Max DD": max_dd.get(s), "Curr DD": curr_dd.get(s),
               "Sharpe": sharpe.get(s), "Calmar": calmar.get(s)}
        if bm_sym: row[f"β {bm_sym}"] = beta.get(s)
        risk_rows.append(row)

    risk_df = pd.DataFrame(risk_rows).reset_index(drop=True)
    r_ret   = f"Ret {period}"
    r_beta  = f"β {bm_sym}" if bm_sym else None
    r_fmt   = {
        r_ret:     lambda x: fmt_pct(x)       if pd.notna(x) else "—",
        "Vol 1M":  lambda x: f"{x:.2f}%"     if pd.notna(x) else "—",
        "Vol 3M":  lambda x: f"{x:.2f}%"     if pd.notna(x) else "—",
        "Max DD":  lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—",
        "Curr DD": lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—",
        "Sharpe":  lambda x: f"{x:.2f}"      if pd.notna(x) else "—",
        "Calmar":  lambda x: f"{x:.2f}"      if pd.notna(x) else "—",
    }
    if r_beta: r_fmt[r_beta] = lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    r_obj = risk_df.style.set_table_styles(_TS).format(r_fmt)
    for col_n, fn in [(r_ret,_cr),("Vol 1M",_cv),("Vol 3M",_cv),
                      ("Max DD",_cdd),("Curr DD",_cdd),("Sharpe",_csh),("Calmar",_csh)]:
        if col_n in risk_df.columns: r_obj = r_obj.applymap(fn, subset=[col_n])
    if r_beta and r_beta in risk_df.columns:
        r_obj = r_obj.applymap(_cb, subset=[r_beta])
    st.dataframe(r_obj, use_container_width=True,
                 height=min(60+len(risk_df)*34,620), hide_index=True)