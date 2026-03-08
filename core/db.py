"""
core/db.py — Database connection and query helpers
All reads, no writes.
"""

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def get_engine():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        st.error("DATABASE_URL not set. Add it to your .env file.")
        st.stop()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}sslmode=require"
    return create_engine(url, pool_pre_ping=True)


# ------------------------------------------------------------------
# Assets
# ------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_assets() -> pd.DataFrame:
    q = """
        SELECT symbol, name, asset_class, category, asset_type, currency,
               exchange, country_cd, sector_cd, data_source, is_diff
        FROM ref_assets
        WHERE is_active = 1
        ORDER BY asset_class, symbol
    """
    return pd.read_sql(q, get_engine())


# ------------------------------------------------------------------
# Returns — latest snapshot
# ------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_latest_returns() -> pd.DataFrame:
    q = """
        SELECT r.symbol, r.date,
               r.r1d, r.r1w, r.r1m, r.r3m, r.rytd, r.r1y
        FROM fact_returns r
        WHERE r.date = (
            SELECT MAX(r2.date) FROM fact_returns r2
            WHERE r2.symbol = r.symbol
        )
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=3600)
def load_returns_with_meta() -> pd.DataFrame:
    """Latest returns joined with asset metadata."""
    q = """
        SELECT
            r.symbol, a.name, a.asset_class, a.category, a.asset_type,
            a.country_cd, a.sector_cd, a.currency, a.data_source, a.is_diff,
            r.date, r.r1d, r.r1w, r.r1m, r.r3m, r.rytd, r.r1y
        FROM fact_returns r
        JOIN ref_assets a ON r.symbol = a.symbol
        WHERE r.date = (
            SELECT MAX(r2.date) FROM fact_returns r2
            WHERE r2.symbol = r.symbol
        )
        ORDER BY a.asset_class, r.symbol
    """
    return pd.read_sql(q, get_engine())


# ------------------------------------------------------------------
# Returns — historical (for returns history page)
# ------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_returns_history(
    bucket: str = "asset_class",
    ret_col: str = "r1d",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load all historical returns joined with metadata.
    bucket: one of 'asset_class', 'country_cd', 'sector_cd'
    """
    date_filter = ""
    params = {}
    if start_date:
        date_filter += " AND r.date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        date_filter += " AND r.date <= :end_date"
        params["end_date"] = end_date

    q = text(f"""
        SELECT
            r.symbol, r.date,
            r.r1d, r.r1w, r.r1m, r.r3m, r.rytd, r.r1y,
            a.name, a.asset_class, a.country_cd, a.sector_cd,
            a.asset_type, a.currency, a.is_diff
        FROM fact_returns r
        JOIN ref_assets a ON r.symbol = a.symbol
        WHERE a.is_active = 1
        {date_filter}
        ORDER BY r.date ASC, a.asset_class, r.symbol
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_available_dates() -> tuple:
    """Returns (min_date, max_date) from fact_returns."""
    q = "SELECT MIN(date) as min_d, MAX(date) as max_d FROM fact_returns"
    row = pd.read_sql(q, get_engine()).iloc[0]
    return str(row["min_d"])[:10], str(row["max_d"])[:10]


# ------------------------------------------------------------------
# Prices
# ------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_prices(symbol: str) -> pd.DataFrame:
    q = text("""
        SELECT datetime, open, high, low, close, volume
        FROM fact_prices
        WHERE symbol = :symbol
        ORDER BY datetime ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"symbol": symbol})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=3600)
def load_multi_prices(symbols: list, col: str = "close") -> pd.DataFrame:
    """Returns a pivot DataFrame: index=date, columns=symbols."""
    if not symbols:
        return pd.DataFrame()
    q = text("""
        SELECT symbol, datetime, close
        FROM fact_prices
        WHERE symbol = ANY(:symbols)
        ORDER BY datetime ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"symbols": symbols})
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.pivot(index="datetime", columns="symbol", values=col)


@st.cache_data(ttl=3600)
def load_prices_by_symbol(symbols: list) -> dict:
    """Returns {symbol: DataFrame(datetime, open, high, low, close, volume)}."""
    if not symbols:
        return {}
    q = text("""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM fact_prices
        WHERE symbol = ANY(:symbols)
        ORDER BY datetime ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"symbols": symbols})
    if df.empty:
        return {}
    df["datetime"] = pd.to_datetime(df["datetime"])
    return {sym: grp.reset_index(drop=True) for sym, grp in df.groupby("symbol")}


@st.cache_data(ttl=3600)
def load_returns_snapshot(symbols: list) -> pd.DataFrame:
    """
    Latest r1d, r1w, r1m, r3m, rytd, r1y for a list of symbols.
    Returns DataFrame indexed by symbol.
    """
    if not symbols:
        return pd.DataFrame()
    q = text("""
        SELECT r.symbol, r.date, r.r1d, r.r1w, r.r1m, r.r3m, r.rytd, r.r1y
        FROM fact_returns r
        WHERE r.symbol = ANY(:syms)
          AND r.date = (
              SELECT MAX(r2.date) FROM fact_returns r2
              WHERE r2.symbol = r.symbol
          )
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"syms": symbols})
    return df.set_index("symbol") if not df.empty else pd.DataFrame()