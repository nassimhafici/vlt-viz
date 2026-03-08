"""
Residual Momentum × SVIX — Multi-horizon backtest & trade recommendations
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import warnings

from core.db import load_multi_prices, load_prices, load_assets
from core.formatting import (
    kpi_card, FONT, GREEN, RED, BORDER, TEXT, TEXT_DIM, TEXT_MID,
    BG, BG2, BG3, GRAY, BLUE, YELLOW
)

warnings.filterwarnings("ignore")
st.title("residual momentum")

# ══════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════
_LEG_TOP = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=9, color=TEXT_DIM), orientation="h", x=0, y=1.08)

def _layout(**kw):
    b = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG2,
             font=dict(family=FONT, size=11, color=TEXT),
             hovermode="x unified", margin=dict(l=48, r=16, t=28, b=44), height=300)
    b.update(kw); return b

def _xax(**kw):
    b = dict(showgrid=False, linecolor=BORDER, tickfont=dict(size=9, color=TEXT_DIM),
             rangebreaks=[dict(bounds=["sat", "mon"])])
    b.update(kw); return b

def _yax(**kw):
    b = dict(showgrid=True, gridcolor="#f3f4f6", linecolor=BORDER,
             tickfont=dict(size=9, color=TEXT_DIM))
    b.update(kw); return b

def _sec(title, top=24):
    st.markdown(
        f"<h3 style='margin-top:{top}px;margin-bottom:10px;font-size:10px;"
        f"font-weight:600;color:{TEXT_DIM};letter-spacing:0.1em;"
        f"text-transform:uppercase'>{title}</h3>", unsafe_allow_html=True)

COLORS = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed",
          "#0891b2", "#db2777", "#16a34a", "#ea580c", "#6366f1"]


# ══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def compute_ff_factors(ff_prices):
    """Compute Fama-French 3-factor proxies from ETF monthly prices."""
    monthly = ff_prices.resample("ME").last()
    rets = monthly.pct_change()
    rf = monthly.get("DGS1MO")
    rf_monthly = (rf / 100 / 12) if rf is not None else 0.0
    factors = pd.DataFrame({
        "MKT_RF": rets.get("SPY", 0) - rf_monthly,
        "SMB":    rets.get("IJR", 0) - rets.get("VV", 0),
        "HML":    rets.get("IWD", 0) - rets.get("IWF", 0),
    }).dropna()
    return factors


def compute_residuals(returns, factors, lookback=36):
    """Rolling OLS residuals: actual − predicted from FF regression."""
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    factor_cols = factors.columns.tolist()
    for t in returns.columns:
        df = pd.concat([returns[[t]], factors[factor_cols]], axis=1, join="inner").dropna()
        if len(df) < lookback + 1:
            continue
        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i + 1]
            y = window[t]
            X = window[factor_cols]
            try:
                model = sm.OLS(y, X).fit()
                residuals.loc[df.index[i], t] = model.resid.iloc[-1]
            except Exception:
                pass
    return residuals


def normalize_residuals_zscore(residuals, lookback=36):
    """Rolling z-score normalization per asset."""
    mu = residuals.rolling(lookback, min_periods=12).mean()
    sigma = residuals.rolling(lookback, min_periods=12).std()
    return (residuals - mu) / sigma.replace(0, np.nan)


def cross_sectional_zscore(signal):
    """Cross-sectional z-score: rank assets relative to peers at each date."""
    mu = signal.mean(axis=1)
    sigma = signal.std(axis=1)
    return signal.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)


def residual_momentum_signal(residuals_z, lookback=12, skip=1, method="sharpe"):
    """Compute momentum signal from normalized residuals."""
    shifted = residuals_z.shift(skip)
    if method == "sharpe":
        mean = shifted.rolling(lookback).mean()
        vol = shifted.rolling(lookback).std()
        return mean / vol.replace(0, np.nan)
    elif method == "trend":
        def lin_trend(x):
            if len(x) < 3: return np.nan
            cum = np.cumprod(1 + x) - 1
            t = np.arange(len(cum))
            return np.polyfit(t, cum, 1)[0]
        return shifted.rolling(lookback).apply(lin_trend, raw=False)
    elif method == "tstat":
        from scipy.stats import linregress
        def t_stat(x):
            if len(x) < 3: return np.nan
            t = np.arange(len(x))
            slope, _, _, _, stderr = linregress(t, x)
            return slope / stderr if stderr != 0 else np.nan
        return shifted.rolling(lookback).apply(t_stat, raw=False)
    else:  # sum
        return shifted.rolling(lookback).sum()


def multi_horizon_signal(residuals_z, horizons, skip=1, method="sharpe", weights=None):
    """Combine momentum signals across multiple lookback horizons."""
    if weights is None:
        weights = [1.0 / len(horizons)] * len(horizons)
    combined = None
    for h, w in zip(horizons, weights):
        sig = residual_momentum_signal(residuals_z, lookback=h, skip=skip, method=method)
        sig_cs = cross_sectional_zscore(sig)
        if combined is None:
            combined = sig_cs * w
        else:
            combined = combined.add(sig_cs * w, fill_value=0)
    return combined


def assign_ranks(signal, n_deciles=5):
    """Rank assets into deciles at each date. 1 = best momentum."""
    ranks = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
    for date, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < n_deciles:
            continue
        r = valid.rank(method="first", ascending=False)
        try:
            q = pd.qcut(r, n_deciles, labels=False, duplicates="drop") + 1
        except ValueError:
            q = ((r - 1) / (len(r) / n_deciles)).astype(int).clip(0, n_deciles - 1) + 1
        ranks.loc[date, valid.index] = q.values
    return ranks


def build_portfolio_returns(returns, ranks, top_rank=1, vol_lb=36, corr_threshold=0.85):
    """Build inverse-vol weighted portfolio from top-ranked assets."""
    vol = returns.rolling(vol_lb, min_periods=12).std()
    inv_vol = 1 / vol.replace([np.inf, -np.inf], np.nan)
    port_rets = pd.Series(index=ranks.index, dtype=float)

    for i, date in enumerate(ranks.index):
        if date not in returns.index:
            continue
        row = ranks.loc[date].dropna()
        buys = row[row == top_rank].index.tolist()
        if not buys:
            continue
        # correlation filter
        if len(buys) > 1:
            loc = returns.index.get_loc(date)
            start = max(0, loc - vol_lb)
            corr_mat = returns.iloc[start:loc][buys].corr()
            keep = []
            for etf in buys:
                if not keep:
                    keep.append(etf)
                else:
                    corrs = corr_mat.loc[etf, keep].abs()
                    if (corrs < corr_threshold).all():
                        keep.append(etf)
            buys = keep if keep else buys[:1]
        if not buys:
            continue
        w = inv_vol.loc[date, buys].dropna()
        weights = (w / w.sum()) if w.sum() > 0 else pd.Series(1.0 / len(buys), index=buys)
        try:
            next_date = returns.index[returns.index.get_loc(date) + 1]
        except (IndexError, KeyError):
            continue
        port_rets.loc[date] = (returns.loc[next_date, buys] * weights).sum()
    return port_rets.dropna()


def perf_stats(rets, freq=12):
    """Compute annualized performance stats from a return series."""
    rets = rets.dropna()
    if len(rets) < 2:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "max_dd": np.nan, "calmar": np.nan, "hit_rate": np.nan}
    ann_ret = (1 + rets).prod() ** (freq / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + rets).cumprod()
    dd = 1 - cum / cum.cummax()
    max_dd = dd.max()
    calmar = ann_ret / max_dd if max_dd > 0 else np.nan
    hit_rate = (rets > 0).mean()
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar, "hit_rate": hit_rate}


def monthly_trade_table(ranks, returns, top_rank=1, vol_lb=36, corr_threshold=0.85):
    """Generate trade recommendations with inverse-vol weights."""
    vol = returns.rolling(vol_lb, min_periods=12).std()
    inv_vol = 1 / vol.replace([np.inf, -np.inf], np.nan)
    records = []
    for date, row in ranks.iterrows():
        buys = row[row == top_rank].dropna().index.tolist()
        if not buys or date not in returns.index:
            continue
        loc = returns.index.get_loc(date)
        start = max(0, loc - vol_lb)
        if len(buys) > 1:
            corr_mat = returns.iloc[start:loc][buys].corr()
            keep = []
            for etf in buys:
                if not keep:
                    keep.append(etf)
                elif (corr_mat.loc[etf, keep].abs() < corr_threshold).all():
                    keep.append(etf)
            buys = keep if keep else buys[:1]
        w = inv_vol.loc[date, buys].dropna()
        weights = (w / w.sum()).round(3) if w.sum() > 0 else pd.Series(1.0 / len(buys), index=buys).round(3)
        records.append({"date": date, "assets": buys, "weights": weights.values.tolist(),
                        "n_assets": len(buys)})
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_all_data():
    assets = load_assets()
    # ETFs for momentum universe: equity + commodity + bond
    etf_mask = (assets["asset_type"] == "etf") & (
        assets["asset_class"].isin(["Equity", "Commodity", "Bond"]))
    etf_syms = assets.loc[etf_mask, "symbol"].tolist()
    # FF factor proxies
    ff_syms = ["IJR", "VV", "SPY", "IWD", "IWF", "DGS1MO"]
    # VIX data
    vix_syms = ["SVIX", "^VIX", "^VIX3M", "^SHORTVOL"]

    all_syms = list(set(etf_syms + ff_syms + vix_syms))
    px = load_multi_prices(all_syms)
    if px is None or px.empty:
        return None, None, None, None, assets
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # Separate datasets
    ff_cols = [c for c in ff_syms if c in px.columns]
    etf_cols = [c for c in etf_syms if c in px.columns and c not in ff_syms]
    vix_cols = [c for c in vix_syms if c in px.columns]

    # Extend SVIX with SHORTVOL proxy for pre-ETF history
    vix_df = px[vix_cols].copy()
    if "SVIX" in vix_df.columns and "^SHORTVOL" in vix_df.columns:
        svix_raw = vix_df["SVIX"].dropna()
        shortvol_raw = vix_df["^SHORTVOL"].dropna()
        vix_df["SVIX"] = _proxy_series_backward(svix_raw, shortvol_raw)

    return px[etf_cols], px[ff_cols], vix_df, assets, etf_cols


def _proxy_series_backward(base, proxy):
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


data_result = load_all_data()
if data_result[0] is None:
    st.warning("No price data available."); st.stop()

etf_px, ff_px, vix_px, assets, etf_universe = data_result

# ── Asset name lookup ─────────────────────────────────────────────
name_map = assets.set_index("symbol")["name"].to_dict()
def lbl(sym):
    n = name_map.get(sym, sym)
    return n[:30] if len(n) > 30 else n

# ══════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMETERS
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    _sec("momentum params", top=0)
    lookback = st.selectbox("FF regression lookback (months)", [24, 36, 48, 60], index=1)
    horizons_str = st.multiselect("Momentum horizons (months)",
                                   [3, 6, 9, 12, 18, 24], default=[6, 12])
    mom_method = st.selectbox("Signal method", ["trend", "sharpe", "tstat", "sum"], index=0)
    skip_months = st.selectbox("Skip last N months", [0, 1, 2], index=1)
    n_deciles = st.selectbox("Deciles", [5, 10, 15, 20], index=1)
    corr_thresh = st.slider("Correlation filter", 0.5, 1.0, 0.80, 0.05)

    st.markdown("<hr style='border-color:#e5e7eb;margin:10px 0'>", unsafe_allow_html=True)
    _sec("SVIX params", top=0)
    w_momentum = st.slider("Weight: momentum", 0.0, 1.0, 0.70, 0.05)
    w_svix = round(1.0 - w_momentum, 2)
    st.markdown(f"<p style='font-size:11px;color:{TEXT_DIM}'>Weight SVIX: {w_svix:.0%}</p>",
                unsafe_allow_html=True)
    start_date = st.date_input("Backtest start", value=pd.Timestamp("2015-01-01"))

if not horizons_str:
    st.warning("Select at least one momentum horizon."); st.stop()

# ══════════════════════════════════════════════════════════════════
# COMPUTATION PIPELINE
# ══════════════════════════════════════════════════════════════════
with st.spinner("Computing residual momentum pipeline..."):
    # 1. Monthly ETF returns
    etf_monthly = etf_px.resample("ME").last()
    # Remove columns with too many NaNs
    valid_cols = etf_monthly.columns[etf_monthly.notna().mean() > 0.5]
    etf_monthly = etf_monthly[valid_cols]
    etf_returns = etf_monthly.pct_change().dropna(how="all")

    # 2. FF factors
    factors = compute_ff_factors(ff_px)

    # 3. Residuals
    residuals = compute_residuals(etf_returns, factors, lookback=lookback)

    # 4. Normalize + cross-sectional z-score
    residuals_z = normalize_residuals_zscore(residuals, lookback=lookback)

    # 5. Multi-horizon signal
    signal = multi_horizon_signal(residuals_z, horizons=horizons_str,
                                  skip=skip_months, method=mom_method)

    # 6. Ranks
    ranks = assign_ranks(signal, n_deciles=n_deciles)

    # 7. Portfolio returns (top decile)
    mom_returns = build_portfolio_returns(etf_returns, ranks, top_rank=1,
                                          vol_lb=lookback, corr_threshold=corr_thresh)

    # 8. SVIX carry signal
    vix_spot = vix_px.get("^VIX", pd.Series(dtype=float))
    vix3m = vix_px.get("^VIX3M", pd.Series(dtype=float))
    svix_px = vix_px.get("SVIX", pd.Series(dtype=float))
    if not svix_px.empty and not vix_spot.empty and not vix3m.empty:
        svix_signal = (vix_spot < vix3m).astype(int)
        svix_daily_ret = (svix_signal.shift(1) * svix_px.pct_change()).fillna(0)
        # Resample to monthly for combination
        svix_monthly = svix_daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    else:
        svix_monthly = pd.Series(dtype=float)

    # 9. Combine strategies
    combined = pd.concat([mom_returns.rename("MOM"), svix_monthly.rename("SVIX")],
                         axis=1).dropna()
    combined = combined.loc[str(start_date):]
    combined["COMBINED"] = combined["MOM"] * w_momentum + combined["SVIX"] * w_svix

    # 10. Benchmark
    spy_px = ff_px.get("SPY", pd.Series(dtype=float))
    if not spy_px.empty:
        bench_monthly = spy_px.resample("ME").last().pct_change()
        bench = bench_monthly.reindex(combined.index).fillna(0).rename("SPY")
    else:
        bench = pd.Series(0, index=combined.index, name="SPY")

    # 11. Trade table
    trades = monthly_trade_table(ranks, etf_returns, top_rank=1,
                                  vol_lb=lookback, corr_threshold=corr_thresh)

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

stats_mom = perf_stats(combined["MOM"])
stats_svix = perf_stats(combined["SVIX"])
stats_comb = perf_stats(combined["COMBINED"])
stats_bench = perf_stats(bench)

k = st.columns(8)
kpi_card(k[0], "combined sharpe", f"{stats_comb['sharpe']:.2f}" if not np.isnan(stats_comb['sharpe']) else "—",
         vc="green" if stats_comb['sharpe'] > 0.5 else "red")
kpi_card(k[1], "combined return", f"{stats_comb['ann_ret']*100:.1f}%",
         vc="green" if stats_comb['ann_ret'] > 0 else "red")
kpi_card(k[2], "combined vol", f"{stats_comb['ann_vol']*100:.1f}%", vc="neutral")
kpi_card(k[3], "max DD", f"{stats_comb['max_dd']*100:.1f}%",
         vc="red" if stats_comb['max_dd'] > 0.15 else "yellow")
kpi_card(k[4], "mom sharpe", f"{stats_mom['sharpe']:.2f}" if not np.isnan(stats_mom['sharpe']) else "—",
         vc="green" if stats_mom['sharpe'] > 0.5 else "neutral")
kpi_card(k[5], "SVIX sharpe", f"{stats_svix['sharpe']:.2f}" if not np.isnan(stats_svix['sharpe']) else "—",
         vc="green" if stats_svix['sharpe'] > 0.5 else "neutral")
kpi_card(k[6], "SPY sharpe", f"{stats_bench['sharpe']:.2f}" if not np.isnan(stats_bench['sharpe']) else "—",
         vc="neutral")
kpi_card(k[7], "hit rate", f"{stats_comb['hit_rate']*100:.0f}%",
         vc="green" if stats_comb['hit_rate'] > 0.55 else "neutral")

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "Backtest", "Drawdowns", "Trade Recommendations", "Decile Analysis", "Stats"
])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — BACKTEST
# ─────────────────────────────────────────────────────────────────
with t1:
    _sec("cumulative returns", top=8)
    cum_mom   = (1 + combined["MOM"]).cumprod()
    cum_svix  = (1 + combined["SVIX"]).cumprod()
    cum_comb  = (1 + combined["COMBINED"]).cumprod()
    cum_bench = (1 + bench).cumprod()

    fig = go.Figure()
    for s, c, nm in [(cum_comb, BLUE, "Combined"), (cum_mom, "#059669", "Momentum"),
                      (cum_svix, "#d97706", "SVIX carry"), (cum_bench, GRAY, "SPY")]:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=nm,
                                  line=dict(color=c, width=2 if nm == "Combined" else 1.3,
                                            dash="solid" if nm == "Combined" else "dot" if nm == "SPY" else "solid"),
                                  hovertemplate=f"{nm}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=1, line_color=BORDER, line_width=1)
    fig.update_layout(**_layout(height=350), legend=_LEG_TOP)
    fig.update_xaxes(**_xax(rangebreaks=[])); fig.update_yaxes(**_yax())
    st.plotly_chart(fig, use_container_width=True)

    # Rolling sharpe
    _sec("rolling 12-month sharpe")
    roll_w = 12
    roll_sh_comb = combined["COMBINED"].rolling(roll_w).apply(
        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
    roll_sh_bench = bench.rolling(roll_w).apply(
        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sh_comb.index, y=roll_sh_comb.values,
                                 mode="lines", name="Combined",
                                 line=dict(color=BLUE, width=1.6)))
    fig_rs.add_trace(go.Scatter(x=roll_sh_bench.index, y=roll_sh_bench.values,
                                 mode="lines", name="SPY",
                                 line=dict(color=GRAY, width=1.2, dash="dot")))
    fig_rs.add_hline(y=0, line_color=BORDER, line_width=1)
    fig_rs.update_layout(**_layout(height=220), legend=_LEG_TOP)
    fig_rs.update_xaxes(**_xax(rangebreaks=[])); fig_rs.update_yaxes(**_yax())
    st.plotly_chart(fig_rs, use_container_width=True)

    # Monthly return bar
    _sec("monthly returns — combined strategy")
    fig_bar = go.Figure()
    colors_bar = [GREEN if v > 0 else RED for v in combined["COMBINED"]]
    fig_bar.add_trace(go.Bar(x=combined.index, y=combined["COMBINED"] * 100,
                              marker=dict(color=colors_bar, opacity=0.8, line=dict(width=0)),
                              hovertemplate="%{x|%Y-%m}: %{y:.2f}%<extra></extra>"))
    fig_bar.update_layout(**_layout(height=200, hovermode="x"), legend=_LEG_TOP)
    fig_bar.update_xaxes(**_xax(rangebreaks=[])); fig_bar.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — DRAWDOWNS
# ─────────────────────────────────────────────────────────────────
with t2:
    _sec("underwater curves", top=8)
    fig_dd = go.Figure()
    for s, c, nm in [(combined["COMBINED"], BLUE, "Combined"),
                      (combined["MOM"], "#059669", "Momentum"),
                      (combined["SVIX"], "#d97706", "SVIX"),
                      (bench, GRAY, "SPY")]:
        cum = (1 + s).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=nm,
            fill="tozeroy" if nm == "Combined" else None,
            fillcolor="rgba(37,99,235,0.08)" if nm == "Combined" else None,
            line=dict(color=c, width=1.6 if nm == "Combined" else 1.1,
                      dash="dot" if nm == "SPY" else "solid"),
            hovertemplate=f"{nm}: %{{y:.2f}}%<extra></extra>"))
    fig_dd.add_hline(y=0, line_color=BORDER, line_width=1)
    fig_dd.update_layout(**_layout(height=300), legend=_LEG_TOP)
    fig_dd.update_xaxes(**_xax(rangebreaks=[])); fig_dd.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_dd, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 3 — TRADE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
with t3:
    if trades.empty:
        st.info("Not enough data to generate trade recommendations.")
    else:
        _sec("latest recommendations", top=8)
        # Show last 12 months
        recent = trades.tail(12).copy()
        recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m")

        for _, row in recent.iloc[::-1].iterrows():
            syms = row["assets"]
            wts = row["weights"]
            pairs = [f"**{s}** ({lbl(s)}) → {w:.0%}" for s, w in zip(syms, wts)]
            with st.expander(f"📅 {row['date']}  —  {row['n_assets']} positions"):
                for p in pairs:
                    st.markdown(p)

        _sec("current month signal — full ranking snapshot", top=24)
        if not signal.empty:
            last_signal = signal.iloc[-1].dropna().sort_values(ascending=False)
            last_ranks = ranks.iloc[-1].dropna().sort_values()

            snap = pd.DataFrame({
                "symbol": last_signal.index,
                "signal": last_signal.values,
                "rank": last_ranks.reindex(last_signal.index).values,
                "name": [lbl(s) for s in last_signal.index]
            }).sort_values("signal", ascending=False).reset_index(drop=True)

            snap["signal"] = snap["signal"].round(3)
            st.dataframe(snap.style.applymap(
                lambda v: f"color:{GREEN}" if v == 1 else f"color:{RED}" if v == n_deciles else f"color:{TEXT}",
                subset=["rank"]
            ).format({"signal": "{:.3f}", "rank": "{:.0f}"}),
                use_container_width=True, height=400, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# TAB 4 — DECILE ANALYSIS
# ─────────────────────────────────────────────────────────────────
with t4:
    _sec("performance by decile", top=8)

    with st.spinner("Computing decile portfolios..."):
        decile_returns = {}
        for dec in range(1, n_deciles + 1):
            dr = build_portfolio_returns(etf_returns, ranks, top_rank=dec,
                                          vol_lb=lookback, corr_threshold=corr_thresh)
            if not dr.empty:
                decile_returns[dec] = dr

    if decile_returns:
        # Bar chart: annualized return by decile
        dec_stats = {d: perf_stats(r) for d, r in decile_returns.items()}
        dec_df = pd.DataFrame(dec_stats).T

        fig_dec = go.Figure()
        colors_dec = [GREEN if v > 0 else RED for v in dec_df["ann_ret"]]
        fig_dec.add_trace(go.Bar(
            x=[f"D{d}" for d in dec_df.index],
            y=dec_df["ann_ret"] * 100,
            marker=dict(color=colors_dec, opacity=0.85, line=dict(color=BORDER, width=1)),
            hovertemplate="Decile %{x}: %{y:.1f}%<extra></extra>"))
        fig_dec.update_layout(**_layout(height=250, hovermode="x"))
        fig_dec.update_xaxes(**_xax(rangebreaks=[])); fig_dec.update_yaxes(**_yax(ticksuffix="%",
                                                                                    title="Ann. return"))
        st.plotly_chart(fig_dec, use_container_width=True)

        # Sharpe by decile
        _sec("sharpe ratio by decile")
        fig_sh = go.Figure()
        sh_colors = [GREEN if v > 0 else RED for v in dec_df["sharpe"]]
        fig_sh.add_trace(go.Bar(
            x=[f"D{d}" for d in dec_df.index], y=dec_df["sharpe"],
            marker=dict(color=sh_colors, opacity=0.85, line=dict(color=BORDER, width=1)),
            hovertemplate="Decile %{x}: %{y:.2f}<extra></extra>"))
        fig_sh.update_layout(**_layout(height=220, hovermode="x"))
        fig_sh.update_xaxes(**_xax(rangebreaks=[])); fig_sh.update_yaxes(**_yax(title="Sharpe"))
        st.plotly_chart(fig_sh, use_container_width=True)

        # Cumulative returns all deciles
        _sec("cumulative returns by decile")
        fig_cum_d = go.Figure()
        for d, r in decile_returns.items():
            cum = (1 + r).cumprod()
            fig_cum_d.add_trace(go.Scatter(
                x=cum.index, y=cum.values, mode="lines", name=f"D{d}",
                line=dict(color=COLORS[d % len(COLORS)], width=1.3 if d == 1 else 0.9,
                          dash="solid" if d == 1 else "dot"),
                hovertemplate=f"D{d}: %{{y:.3f}}<extra></extra>"))
        fig_cum_d.update_layout(**_layout(height=300), legend=_LEG_TOP)
        fig_cum_d.update_xaxes(**_xax(rangebreaks=[])); fig_cum_d.update_yaxes(**_yax())
        st.plotly_chart(fig_cum_d, use_container_width=True)
    else:
        st.info("Not enough data for decile analysis.")


# ─────────────────────────────────────────────────────────────────
# TAB 5 — STATS TABLE
# ─────────────────────────────────────────────────────────────────
with t5:
    _sec("strategy comparison", top=8)
    all_stats = pd.DataFrame({
        "Combined": stats_comb,
        "Momentum": stats_mom,
        "SVIX carry": stats_svix,
        "SPY": stats_bench,
    }).T
    all_stats.columns = ["Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Calmar", "Hit Rate"]
    fmt = {
        "Ann. Return": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
        "Ann. Vol":    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
        "Sharpe":      lambda x: f"{x:.2f}" if pd.notna(x) else "—",
        "Max DD":      lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
        "Calmar":      lambda x: f"{x:.2f}" if pd.notna(x) else "—",
        "Hit Rate":    lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—",
    }
    styled = (all_stats.style
              .format(fmt)
              .applymap(lambda v: f"color:{GREEN}" if isinstance(v, float) and v > 0.5 else f"color:{TEXT}",
                        subset=["Sharpe"])
              .set_table_styles([
                  {"selector": "th", "props": [("background", BG3), ("color", TEXT_DIM),
                   ("font-family", FONT), ("font-size", "10px"), ("font-weight", "600"),
                   ("text-transform", "uppercase"), ("letter-spacing", "0.06em"),
                   ("border-bottom", f"1px solid {BORDER}"), ("padding", "7px 12px")]},
                  {"selector": "td", "props": [("background", BG2), ("font-family", FONT),
                   ("font-size", "12px"), ("color", TEXT),
                   ("border-bottom", f"1px solid {BORDER}"), ("padding", "5px 12px")]},
                  {"selector": "tr:hover td", "props": [("background", BG3)]}
              ]))
    st.dataframe(styled, use_container_width=True, height=220)

    # Yearly returns
    _sec("yearly returns — combined strategy", top=24)
    yearly = combined["COMBINED"].groupby(combined.index.year).apply(
        lambda x: (1 + x).prod() - 1) * 100
    fig_yr = go.Figure()
    yr_colors = [GREEN if v > 0 else RED for v in yearly.values]
    fig_yr.add_trace(go.Bar(x=yearly.index.astype(str), y=yearly.values,
                             marker=dict(color=yr_colors, opacity=0.85,
                                         line=dict(color=BORDER, width=1)),
                             text=[f"{v:.1f}%" for v in yearly.values],
                             textposition="outside",
                             textfont=dict(size=9, color=TEXT_DIM),
                             hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig_yr.update_layout(**_layout(height=250, hovermode="x"))
    fig_yr.update_xaxes(**_xax(rangebreaks=[])); fig_yr.update_yaxes(**_yax(ticksuffix="%"))
    st.plotly_chart(fig_yr, use_container_width=True)

    # Weight allocation breakdown
    _sec("parameters summary", top=24)
    params = {
        "FF lookback": f"{lookback}m",
        "Horizons": ", ".join(str(h) + "m" for h in horizons_str),
        "Signal method": mom_method,
        "Skip": f"{skip_months}m",
        "Deciles": n_deciles,
        "Corr filter": f"{corr_thresh}",
        "Wt momentum": f"{w_momentum:.0%}",
        "Wt SVIX": f"{w_svix:.0%}",
        "Start": str(start_date),
        "Universe": f"{len(etf_universe)} ETFs",
    }
    for lbl_p, val in params.items():
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
            f"border-bottom:1px solid {BORDER};font-size:11px'>"
            f"<span style='color:{TEXT_DIM}'>{lbl_p}</span>"
            f"<span style='color:{TEXT};font-weight:500'>{val}</span></div>",
            unsafe_allow_html=True)