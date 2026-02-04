from finvizfinance.quote import finvizfinance
import yfinance as yf
import json

# Choose a gold‐related ticker — e.g., GLD is a popular gold ETF
gold_ticker = "GLD"

# Create a finvizfinance object
gold = finvizfinance(gold_ticker)

# ———————————
# 1) GOLD FUNDAMENTAL / PRICE INFO
# ———————————
fundamentals = gold.ticker_fundament(raw=True, output_format="dict")
print("=== GOLD FUNDAMENTALS ===")
print(fundamentals)

# If you need a numeric price, you can usually parse the 'Price' key
current_price = fundamentals.get("Price")
print(f"\nCurrent {gold_ticker} Price: {current_price}")


# ———————————
# 2) INSIDER TRADING TABLE
# ———————————
insider_df = gold.ticker_inside_trader()
print("\n=== INSIDER TRADING ===")
print(insider_df)


# ———————————
# 3) NEWS TABLE
# ———————————
news_df = gold.ticker_news()
print("\n=== LATEST NEWS ===")
print(news_df)



symbol = "GC=F"  # Gold continuous futures
gold = yf.Ticker(symbol)

# Get last price / history
df = gold.history(period="1d")
print(df["Close"].iloc[-1])  # latest close


# Fetch latest gold price (ensure it is a plain Python type)
try:
    latest_price = float(df['Close'].iloc[-1])
except Exception:
    latest_price = df['Close'].iloc[-1]

# Fetch latest news and convert any Timestamp to ISO strings
try:
    # Prefer using DataFrame.to_json with ISO dates then parse back to Python objects
    latest_news = json.loads(news_df.to_json(orient='records', date_format='iso'))
except Exception:
    # Fallback: convert datetime-like columns to string then to dict
    news_df_serializable = news_df.copy()
    for col in news_df_serializable.select_dtypes(include=['datetimetz', 'datetime']).columns:
        news_df_serializable[col] = news_df_serializable[col].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    latest_news = news_df_serializable.to_dict(orient='records')

# Optionally create a recent price chart and include it as a base64 PNG string
price_chart_base64 = None
try:
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # Try to get 90 days of history for charting
    hist = None
    try:
        hist = gold.history(period='90d')['Close']
    except Exception:
        # If the current `gold` object isn't the yf.Ticker for futures, try the symbol directly
        try:
            yf_ticker = yf.Ticker('GC=F')
            hist = yf_ticker.history(period='90d')['Close']
        except Exception:
            hist = None

    if hist is not None and len(hist) > 0:
        buf = BytesIO()
        plt.figure(figsize=(8, 4))
        hist.plot(title='Gold Close Price (90d)')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        price_chart_base64 = base64.b64encode(buf.read()).decode('ascii')
except Exception:
    price_chart_base64 = None

# Combine data into a dictionary (JSON-serializable)
combined_data = {
    'latest_price': latest_price,
    'latest_news': latest_news,
    'price_chart_base64': price_chart_base64
}

# Save combined data as JSON
with open('gold_data.json', 'w') as json_file:
    json.dump(combined_data, json_file, indent=4)

