from datetime import date
import pandas as pd
from streamlit_app import load_holdings, enrich_holdings, save_snapshot

holdings = load_holdings(None)
for timeframe in ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y"]:
    enriched = enrich_holdings(holdings, timeframe)
    save_snapshot(enriched, timeframe)
print(f"Saved portfolio snapshots for {date.today().isoformat()}.")
