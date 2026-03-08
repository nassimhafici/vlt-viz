import pandas as pd
import plotly.graph_objects as go
from core.formatting import BG, BG2, BORDER, TEXT, TEXT_DIM, TEXT_MID, GREEN, RED, BLUE, GRAY

_BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="'IBM Plex Mono', monospace", color=TEXT_MID, size=10),
    margin=dict(l=52, r=20, t=20, b=44),
    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=BORDER,
               tickcolor=BORDER, tickfont=dict(size=9, color=TEXT_DIM)),
    yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=0.5, zeroline=False,
               showline=False, tickfont=dict(size=9, color=TEXT_DIM)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                font=dict(size=9, color=TEXT_DIM)),
)


def _layout(**overrides):
    import copy
    l = copy.deepcopy(_BASE)
    l.update(overrides)
    return l


def price_chart(df, symbol, name="", chart_type="Line"):
    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["close"], mode="lines",
            line=dict(color=BLUE, width=1.2), name=symbol,
            hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>",
        ))
    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df["datetime"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
            decreasing=dict(line=dict(color=RED, width=1), fillcolor=RED),
        ))
        fig.update_layout(xaxis_rangeslider_visible=False)
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=df["datetime"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing=dict(line=dict(color=GREEN, width=1)),
            decreasing=dict(line=dict(color=RED, width=1)),
        ))
    fig.update_layout(**_layout(height=300))
    return fig


def returns_bar(df, ret_col, label_col="symbol", top_n=15):
    valid = df[[label_col, ret_col]].dropna().copy()
    top = valid.nlargest(top_n, ret_col)
    bot = valid.nsmallest(top_n, ret_col)
    combined = pd.concat([top, bot]).drop_duplicates(label_col).sort_values(ret_col)
    colors = [GREEN if v >= 0 else RED for v in combined[ret_col]]
    fig = go.Figure(go.Bar(
        x=combined[ret_col] * 100, y=combined[label_col], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}%" for v in combined[ret_col] * 100],
        textposition="outside",
        textfont=dict(size=9, family="'IBM Plex Mono', monospace", color=TEXT_DIM),
        hovertemplate="%{y}  %{x:.2f}%<extra></extra>",
    ))
    h = max(260, len(combined) * 20 + 60)
    fig.update_layout(**_layout(height=h, xaxis_ticksuffix="%", xaxis_showgrid=True,
                                xaxis_gridcolor=BORDER))
    return fig


def returns_heatmap(df, group_col, ret_col, title="", height=440):
    agg = (
        df.groupby(group_col)[ret_col]
        .agg(["mean", "min", "max", "count"])
        .rename(columns={"mean": "avg"})
        .sort_values("avg", ascending=False)
        .reset_index()
    )
    colors = [GREEN if v >= 0 else RED for v in agg["avg"]]
    fig = go.Figure(go.Bar(
        x=agg[group_col], y=agg["avg"] * 100,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.2f}%" for v in agg["avg"] * 100],
        textposition="outside",
        textfont=dict(size=9, family="'IBM Plex Mono', monospace", color=TEXT_DIM),
        hovertemplate="%{x}  %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_layout(height=height, yaxis_ticksuffix="%",
                                xaxis_showgrid=False, xaxis_tickangle=-38))
    return fig