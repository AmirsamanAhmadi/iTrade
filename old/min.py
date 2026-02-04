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
OUTPUT_FILE = "gold_data_with_news.json"

# -----------------------------
# Helper: pandas → JSON safe
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
news_df = gold_finviz.ticker_news()

# Ensure datetime + date column
news_df = news_df.copy()
news_df["date"] = pd.to_datetime(news_df["Date"]).dt.date

# Group news by date
news_by_date = (
    news_df
    .drop(columns=["Date"])
    .groupby("date")
    .apply(lambda x: x.to_dict(orient="records"))
    .to_dict()
)

# -----------------------------
# 2) YAHOO FINANCE – 1-MIN DATA (PAST YEAR)
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

# Add date column for alignment
gold_1m_df = gold_1m_df.copy()
gold_1m_df["date"] = gold_1m_df.index.date

# -----------------------------
# 3) ATTACH NEWS TO EACH DATE
# -----------------------------
daily_data = []

for date, day_df in gold_1m_df.groupby("date"):
    daily_data.append({
        "date": date.isoformat(),
        "prices": json.loads(
            day_df.drop(columns=["date"])
            .reset_index()
            .to_json(orient="records", date_format="iso")
        ),
        "news": news_by_date.get(date, [])
    })

# -----------------------------
# 4) FINAL COMBINED STRUCTURE
# -----------------------------
combined_data = {
    "symbol": GOLD_FUTURES,
    "fundamentals": fundamentals,
    "data": daily_data,
    "generated_at": datetime.utcnow().isoformat()
}

# -----------------------------
# 5) SAVE JSON
# -----------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"\n✅ Saved successfully to {OUTPUT_FILE}")
