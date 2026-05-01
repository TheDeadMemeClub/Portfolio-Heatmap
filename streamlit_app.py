from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
DEFAULT_HOLDINGS = APP_DIR / "data" / "portfolio_holdings.csv"
SNAPSHOT_FILE = APP_DIR / "snapshots" / "portfolio_history.csv"

st.set_page_config(page_title="Portfolio Heat Map", layout="wide", page_icon="🟩")

TIMEFRAMES = {
    "1D": {"period": "7d", "anchor": "previous"},
    "1W": {"period": "1mo", "days": 7},
    "1M": {"period": "3mo", "days": 30},
    "3M": {"period": "6mo", "days": 91},
    "6M": {"period": "1y", "days": 182},
    "YTD": {"period": "ytd", "ytd": True},
    "1Y": {"period": "2y", "days": 365},
    "3Y": {"period": "5y", "days": 365 * 3},
    "5Y": {"period": "10y", "days": 365 * 5},
}

ZERO_RETURN_CLASSES = {"cash", "money market", "sweep"}


def clean_symbol(symbol: object) -> str:
    if pd.isna(symbol):
        return ""
    s = str(symbol).strip().upper()
    if s in {"CASH", "CASH / SWEEP", "MMKT"}:
        return "CASH"
    if " BOND" in s or "NOTES" in s:
        return s
    return s.replace(".", "-")


def load_holdings(uploaded_file=None) -> pd.DataFrame:
    source = uploaded_file if uploaded_file is not None else DEFAULT_HOLDINGS
    df = pd.read_csv(source)
    required = {"Symbol", "Sector", "Industry"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    df = df.copy()
    df["Symbol"] = df["Symbol"].map(clean_symbol)
    df["Sector"] = df["Sector"].fillna("Unclassified").astype(str)
    df["Industry"] = df["Industry"].fillna("Unclassified").astype(str)
    if "Description" not in df.columns:
        df["Description"] = df["Symbol"]
    if "AssetClass" not in df.columns:
        df["AssetClass"] = "Security"
    if "MarketValue" not in df.columns:
        df["MarketValue"] = 0.0
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce").fillna(0.0)
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    else:
        df["Quantity"] = np.nan
    return df[df["Symbol"] != ""].reset_index(drop=True)


def tradable_symbols(df: pd.DataFrame) -> list[str]:
    skip_words = ("BOND", "CASH")
    symbols = []
    for _, row in df.iterrows():
        symbol = row["Symbol"]
        asset_text = f"{row.get('AssetClass', '')} {row.get('Industry', '')}".lower()
        if any(word in symbol for word in skip_words):
            continue
        if any(x in asset_text for x in ZERO_RETURN_CLASSES):
            continue
        symbols.append(symbol)
    return sorted(set(symbols))


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_prices(symbols: Tuple[str, ...], period: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    kwargs = {"auto_adjust": True, "progress": False, "threads": True, "group_by": "column"}
    if start:
        data = yf.download(list(symbols), start=start, end=end, **kwargs)
    else:
        data = yf.download(list(symbols), period=period, **kwargs)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data.xs("Close", axis=1, level=1, drop_level=False)
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})
    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def calculate_returns(close: pd.DataFrame, timeframe: str, custom_start: date | None = None) -> Dict[str, float]:
    if close.empty:
        return {}
    returns = {}
    last = close.ffill().iloc[-1]
    if timeframe == "1D":
        if len(close) < 2:
            return {c: 0.0 for c in close.columns}
        anchor = close.ffill().iloc[-2]
    elif timeframe == "Custom" and custom_start is not None:
        start_ts = pd.Timestamp(custom_start)
        eligible = close[close.index >= start_ts]
        if eligible.empty:
            anchor = close.ffill().iloc[0]
        else:
            anchor = eligible.ffill().iloc[0]
    elif timeframe == "YTD":
        start_ts = pd.Timestamp(date.today().year, 1, 1)
        eligible = close[close.index >= start_ts]
        anchor = eligible.ffill().iloc[0] if not eligible.empty else close.ffill().iloc[0]
    else:
        days = TIMEFRAMES[timeframe]["days"]
        target = close.index[-1] - pd.Timedelta(days=days)
        eligible = close[close.index <= target]
        anchor = eligible.ffill().iloc[-1] if not eligible.empty else close.ffill().iloc[0]
    for symbol in close.columns:
        a = anchor.get(symbol, np.nan)
        l = last.get(symbol, np.nan)
        if pd.notna(a) and pd.notna(l) and a != 0:
            returns[symbol] = float(l / a - 1)
    return returns


def latest_prices(close: pd.DataFrame) -> Dict[str, float]:
    if close.empty:
        return {}
    last = close.ffill().iloc[-1]
    return {c: float(v) for c, v in last.items() if pd.notna(v)}


def enrich_holdings(df: pd.DataFrame, timeframe: str, custom_start: date | None = None, custom_end: date | None = None) -> pd.DataFrame:
    symbols = tuple(tradable_symbols(df))
    if timeframe == "Custom":
        start = str(custom_start or (date.today() - timedelta(days=30)))
        end_date = (custom_end or date.today()) + timedelta(days=1)
        close = fetch_prices(symbols, "", start=start, end=str(end_date))
    else:
        close = fetch_prices(symbols, TIMEFRAMES[timeframe]["period"])
    rets = calculate_returns(close, timeframe, custom_start)
    prices = latest_prices(close)
    out = df.copy()
    out["LookbackReturn"] = out["Symbol"].map(rets).fillna(0.0)
    out["LatestPrice"] = out["Symbol"].map(prices)
    out["LiveMarketValue"] = np.where(
        out["Quantity"].notna() & out["LatestPrice"].notna(),
        out["Quantity"] * out["LatestPrice"],
        out["MarketValue"],
    )
    out["LiveMarketValue"] = pd.to_numeric(out["LiveMarketValue"], errors="coerce").fillna(out["MarketValue"])
    total = out["LiveMarketValue"].sum()
    out["PortfolioPct"] = np.where(total > 0, out["LiveMarketValue"] / total, 0.0)
    out["WeightedReturnContribution"] = out["PortfolioPct"] * out["LookbackReturn"]
    out["WeightedDollarChangeEstimate"] = out["LiveMarketValue"] * out["LookbackReturn"]
    return out


def save_snapshot(df: pd.DataFrame, timeframe: str) -> None:
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_value = float(df["LiveMarketValue"].sum())
    weighted_return = float(df["WeightedReturnContribution"].sum())
    row = pd.DataFrame([{
        "snapshot_datetime": datetime.now().isoformat(timespec="seconds"),
        "timeframe": timeframe,
        "total_value": total_value,
        "weighted_return": weighted_return,
        "positions": int(len(df)),
    }])
    if SNAPSHOT_FILE.exists():
        old = pd.read_csv(SNAPSHOT_FILE)
        combined = pd.concat([old, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(SNAPSHOT_FILE, index=False)


def make_heatmap(df: pd.DataFrame, timeframe: str):
    plot_df = df.copy()
    plot_df = plot_df[plot_df["LiveMarketValue"] > 0].copy()
    max_abs = max(abs(plot_df["LookbackReturn"].min()), abs(plot_df["LookbackReturn"].max()), 0.01)
    max_abs = min(max_abs, 0.25)
    plot_df["ReturnLabel"] = (plot_df["LookbackReturn"] * 100).map(lambda x: f"{x:+.2f}%")
    plot_df["ValueLabel"] = plot_df["LiveMarketValue"].map(lambda x: f"${x:,.0f}")
    fig = px.treemap(
        plot_df,
        path=["Sector", "Industry", "Symbol"],
        values="LiveMarketValue",
        color="LookbackReturn",
        color_continuous_scale="RdYlGn",
        range_color=(-max_abs, max_abs),
        hover_data={
            "Description": True,
            "LiveMarketValue": ":$,.2f",
            "PortfolioPct": ":.2%",
            "LookbackReturn": ":.2%",
            "WeightedDollarChangeEstimate": ":$,.2f",
        },
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[3]}",
        textfont=dict(size=20, color="white"),
        marker=dict(line=dict(color="#20242d", width=1)),
    )
    fig.update_layout(
        title=f"Portfolio Heat Map — {timeframe} Performance",
        paper_bgcolor="#20242d",
        plot_bgcolor="#20242d",
        font=dict(color="white"),
        margin=dict(t=48, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title="Return", tickformat="+.0%"),
        height=760,
    )
    return fig


def sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("Sector", dropna=False).agg(
        MarketValue=("LiveMarketValue", "sum"),
        WeightedReturn=("WeightedReturnContribution", "sum"),
        Positions=("Symbol", "count"),
    ).reset_index()
    total = grouped["MarketValue"].sum()
    grouped["PortfolioPct"] = np.where(total > 0, grouped["MarketValue"] / total, 0.0)
    grouped = grouped.sort_values("MarketValue", ascending=False)
    return grouped


st.title("Finviz-Style Portfolio Heat Map")
st.caption("Block size = position size. Color = selected lookback performance. Data source: Yahoo Finance via yfinance.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    timeframe = st.selectbox("Lookback period", ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "Custom"], index=1)
    custom_start = None
    custom_end = None
    if timeframe == "Custom":
        custom_start = st.date_input("Custom start", value=date.today() - timedelta(days=30))
        custom_end = st.date_input("Custom end", value=date.today())
    min_position = st.number_input("Hide positions below this market value", min_value=0, value=0, step=100)
    include_cash = st.toggle("Include cash / money market", value=True)
    refresh = st.button("Refresh market data")
    if refresh:
        fetch_prices.clear()

holdings = load_holdings(uploaded)
if not include_cash:
    mask = holdings["Sector"].str.lower().str.contains("cash", na=False) | holdings["Industry"].str.lower().str.contains("cash|money market|sweep", na=False)
    holdings = holdings[~mask].copy()

enriched = enrich_holdings(holdings, timeframe, custom_start, custom_end)
if min_position > 0:
    enriched = enriched[enriched["LiveMarketValue"] >= min_position].copy()

portfolio_value = float(enriched["LiveMarketValue"].sum())
weighted_return = float(enriched["WeightedReturnContribution"].sum())
est_dollar_change = float(enriched["WeightedDollarChangeEstimate"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Portfolio value", f"${portfolio_value:,.0f}")
k2.metric(f"Weighted {timeframe} return", f"{weighted_return:+.2%}")
k3.metric("Estimated $ change", f"${est_dollar_change:,.0f}")
k4.metric("Positions shown", f"{len(enriched):,}")

fig = make_heatmap(enriched, timeframe)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Sector breakdown")
    s = sector_summary(enriched)
    st.dataframe(
        s.style.format({"MarketValue": "${:,.0f}", "PortfolioPct": "{:.2%}", "WeightedReturn": "{:+.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
with right:
    st.subheader("Top movers")
    movers = enriched.sort_values("LookbackReturn", ascending=False)[[
        "Symbol", "Description", "Sector", "LiveMarketValue", "PortfolioPct", "LookbackReturn", "WeightedDollarChangeEstimate"
    ]]
    st.dataframe(
        movers.style.format({
            "LiveMarketValue": "${:,.0f}",
            "PortfolioPct": "{:.2%}",
            "LookbackReturn": "{:+.2%}",
            "WeightedDollarChangeEstimate": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Holdings detail")
st.dataframe(
    enriched.sort_values("LiveMarketValue", ascending=False)[[
        "Symbol", "Description", "Sector", "Industry", "AssetClass", "LiveMarketValue", "PortfolioPct", "LookbackReturn", "LatestPrice"
    ]].style.format({
        "LiveMarketValue": "${:,.0f}",
        "PortfolioPct": "{:.2%}",
        "LookbackReturn": "{:+.2%}",
        "LatestPrice": "${:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    csv = enriched.to_csv(index=False).encode("utf-8")
    st.download_button("Download enriched holdings CSV", csv, file_name="portfolio_heatmap_enriched.csv", mime="text/csv")
with c2:
    sector_csv = sector_summary(enriched).to_csv(index=False).encode("utf-8")
    st.download_button("Download sector summary CSV", sector_csv, file_name="portfolio_sector_summary.csv", mime="text/csv")
with c3:
    if st.button("Save today's snapshot"):
        save_snapshot(enriched, timeframe)
        st.success(f"Snapshot saved to {SNAPSHOT_FILE}")

if SNAPSHOT_FILE.exists():
    st.subheader("Saved daily snapshots")
    hist = pd.read_csv(SNAPSHOT_FILE)
    st.dataframe(hist.tail(60), use_container_width=True, hide_index=True)
