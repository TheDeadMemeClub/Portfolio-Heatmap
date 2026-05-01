# Portfolio Heat Map App

This is a Finviz-style portfolio heat map for your holdings. Block size is based on your position size, and color is based on performance over the selected lookback period.

## What it does

- Shows your portfolio as a sector/industry/symbol heat map.
- Lets you switch performance windows: 1D, 1W, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, or custom dates.
- Pulls live/historical price data from Yahoo Finance through `yfinance`.
- Handles stocks, ETFs, and mutual funds.
- Treats cash and money market positions as 0% return unless you edit them.
- Saves daily snapshots if you click the snapshot button or run the daily script.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How to update your portfolio

The default file is:

```text
data/portfolio_holdings.csv
```

You can replace it or upload a CSV inside the app.

Minimum needed columns:

```text
Symbol, Sector, Industry, MarketValue
```

Better columns if available:

```text
Symbol, Description, Sector, Industry, Quantity, MarketValue, AssetClass
```

If `Quantity` is included, the app can estimate live position value as `Quantity x latest price`. If not, it uses the uploaded `MarketValue` as the block size.

## Run daily snapshot manually

```bash
python run_daily_snapshot.py
```

This app is for analysis and visualization, not financial advice.
