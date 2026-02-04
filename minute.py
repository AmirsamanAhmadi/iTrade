from finvizfinance.quote import finvizfinance
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import time

# -----------------------------
# CONFIG
# -----------------------------
GOLD_ETF = "GLD"
GOLD_FUTURES = "GC=F"
OUTPUT_FILE = "gold_data.json"

# -----------------------------
# Helper: make pandas JSON safe
# -----------------------------
def df_to_json_safe(df):
    if df is None or df.empty:
        return []
    return json.loads(
        df.reset_index().to_json(
            orient="records",
            date_format="iso"
        )
    )

# -----------------------------
# 1) FINVIZ DATA
# -----------------------------
gold_finviz = finvizfinance(GOLD_ETF)

fundamentals = gold_finviz.ticker_fundament(raw=True, output_format="dict")
current_price = fundamentals.get("Price")

insider_df = gold_finviz.ticker_inside_trader()
news_df = gold_finviz.ticker_news()

# -----------------------------
# 2) YAHOO FINANCE – LATEST PRICE
# -----------------------------
yf_gold = yf.Ticker(GOLD_FUTURES)
latest_df = yf_gold.history(period="1d")

latest_price = (
    float(latest_df["Close"].iloc[-1])
    if not latest_df.empty
    else None
)

# -----------------------------
# 3) YAHOO FINANCE – 1-MIN DATA (PAST YEAR, ROLLING)
# -----------------------------
all_data = []

end = datetime.utcnow()
start_limit = end - timedelta(days=365)

while end > start_limit:
    start = end - timedelta(days=7)

    print(f"Fetching 1m data: {start.date()} → {end.date()}")

    try:
        df = yf.download(
            GOLD_FUTURES,
            start=start,
            end=end,
            interval="1m",
            progress=False
        )
        if not df.empty:
            all_data.append(df)
    except Exception as e:
        print("Download error:", e)

    end = start
    time.sleep(1)

gold_1m_df = (
    pd.concat(all_data)
    .sort_index()
    .drop_duplicates()
    if all_data else pd.DataFrame()
)

# -----------------------------
# 4) COMBINE EVERYTHING (JSON SAFE)
# -----------------------------
combined_data = {
    "symbol": GOLD_FUTURES,
    "latest_price": latest_price,
    "fundamentals": fundamentals,
    "insider_trading": df_to_json_safe(insider_df),
    "latest_news": df_to_json_safe(news_df),
    "price_history_1m": df_to_json_safe(gold_1m_df),
    "generated_at": datetime.utcnow().isoformat()
}

# -----------------------------
# 5) SAVE JSON
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"\n✅ Saved successfully to {OUTPUT_FILE}")
