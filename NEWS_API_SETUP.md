# News API Configuration

This trading dashboard now supports **14+ news sources** for comprehensive market coverage.

## 🚀 MASSIVE NEWS API (Primary)

### **Massive Finance News API** (Recommended)
- Professional financial news aggregator
- **Requires API Key**
- Real-time market news
- Sentiment analysis included
- Get key at: https://massive.com/landing/finance-news-api

## 📰 PREMIUM NEWS APIs

### 1. **NewsAPI** (newsapi.org)
- General news with financial focus
- **Requires API Key**
- Free tier: 100 requests/day
- Get key at: https://newsapi.org/register

### 2. **Alpha Vantage** (alphavantage.co)
- Financial news with sentiment analysis
- **Requires API Key**
- Free tier: 25 requests/day
- Get key at: https://www.alphavantage.co/support/#api-key

### 3. **GNews** (gnews.io)
- Global news aggregator
- **Requires API Key**
- Free tier: 100 requests/day
- Get key at: https://gnews.io/

### 4. **Currents API** (currentsapi.services)
- Real-time news API
- **Requires API Key**
- Get key at: https://currentsapi.services/

### 5. **New York Times API**
- Business and finance section
- **Requires API Key**
- Free tier available
- Get key at: https://developer.nytimes.com/

### 6. **The Guardian API**
- Business news
- **Requires API Key**
- Free tier: 12 requests/day
- Get key at: https://open-platform.theguardian.com/

### 7. **Benzinga API**
- Financial news specialist
- **Requires API Key**
- Professional-grade financial news
- Get key at: https://www.benzinga.com/apis

## 📡 RSS FEEDS (No API Key Required)

### 8. **Finviz** (Default)
- Market news and headlines
- Automatically enabled if `finvizfinance` package installed

### 9. **MarketWatch** (Dow Jones)
- Top market stories RSS feed
- No API key required

### 10. **Yahoo Finance**
- Market news feeds
- No API key required
- Requires: `pip install feedparser`

### 11. **Investing.com**
- Global financial news RSS
- No API key required
- Requires: `pip install feedparser`

## 💬 SOCIAL/COMMUNITY SOURCES

### 12. **StockTwits**
- Trader community ideas and sentiment
- No API key required
- Real-time trader discussions

### 13. **Reddit**
- r/wallstreetbets, r/investing, r/stocks, r/StockMarket
- No API key required
- Community sentiment and discussions

## ⚙️ Setup Instructions

### Option 1: Environment Variables (Recommended)

Create a `.env` file in your project root:

```bash
# MASSIVE Finance News API (Primary - Recommended)
# Get key at: https://massive.com/landing/finance-news-api
MASSIVE_KEY="your_massive_api_key_here"

# NewsAPI (newsapi.org) - 100 requests/day free
# Get key at: https://newsapi.org/register
NEWSAPI_KEY="your_newsapi_key_here"

# Alpha Vantage (alphavantage.co) - 25 requests/day free
# Get key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY="your_alpha_vantage_key_here"

# GNews (gnews.io) - 100 requests/day free
# Get key at: https://gnews.io/
GNEWS_KEY="your_gnews_key_here"

# Currents API (currentsapi.services)
# Get key at: https://currentsapi.services/
CURRENTS_KEY="your_currents_key_here"

# New York Times API
# Get key at: https://developer.nytimes.com/
NYT_KEY="your_nyt_key_here"

# The Guardian API - 12 requests/day free
# Get key at: https://open-platform.theguardian.com/
GUARDIAN_KEY="your_guardian_key_here"

# Benzinga API
# Get key at: https://www.benzinga.com/apis
BENZINGA_KEY="your_benzinga_key_here"
```

### Option 2: Export in Shell

```bash
export MASSIVE_KEY="your_key_here"
export NEWSAPI_KEY="your_key_here"
export ALPHA_VANTAGE_KEY="your_key_here"
export GNEWS_KEY="your_key_here"
export CURRENTS_KEY="your_key_here"
export NYT_KEY="your_key_here"
export GUARDIAN_KEY="your_key_here"
export BENZINGA_KEY="your_key_here"
```

### Option 3: Direct in Code (Not Recommended for Production)

Edit `services/news_service.py`:

```python
@dataclass
class NewsService:
    source: str = 'multi'
    MASSIVE_KEY: Optional[str] = "your_key_here"
    NEWSAPI_KEY: Optional[str] = "your_key_here"
    ALPHA_VANTAGE_KEY: Optional[str] = "your_key_here"
    GNEWS_KEY: Optional[str] = "your_key_here"
    CURRENTS_KEY: Optional[str] = "your_key_here"
    NYT_KEY: Optional[str] = "your_key_here"
    GUARDIAN_KEY: Optional[str] = "your_key_here"
    BENZINGA_KEY: Optional[str] = "your_key_here"
```

## Installing Additional Dependencies

For full functionality, install these packages:

```bash
# For MarketWatch RSS support
pip install feedparser

# For Finviz (already in requirements)
pip install finvizfinance

# For general HTTP requests (already in requirements)
pip install requests
```

## Rate Limiting

The service implements automatic rate limiting:
- **Finviz**: 1 second between requests
- **NewsAPI**: Respects API limits (100/day free)
- **Alpha Vantage**: 1 second between requests (25/day free)
- **MarketWatch**: 1 second between requests

## Troubleshooting

### No news appearing?
1. Check that at least one news source is configured
2. Verify API keys are valid (check logs for errors)
3. Ensure internet connection is active
4. Check that `db/news/` directory exists and is writable

### API rate limit errors?
- News sources will automatically skip if rate limited
- The service aggregates from multiple sources, so one failing won't stop others
- Consider upgrading to paid plans for higher limits

### Missing feedparser?
MarketWatch RSS will be skipped with a log message. Install with:
```bash
pip install feedparser
```

## News Data Structure

Each news item contains:
```json
{
  "timestamp": "2026-02-07T12:00:00",
  "headline": "Market reaches new highs",
  "source": "newsapi:Bloomberg",
  "mapped_currencies": ["USD", "EUR"],
  "url": "https://...",
  "description": "...",
  "sentiment": "positive"  // Alpha Vantage only
}
```

## Storage

All news is stored in: `db/news/news_YYYY-MM-DD.jsonl`
- One file per day
- JSON Lines format
- Automatically deduplicated
- Sorted by timestamp (most recent first)
