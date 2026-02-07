# News API Configuration

This trading dashboard now supports multiple news sources for comprehensive market coverage.

## Supported News Sources

### 1. **Finviz** (Default - No API Key Required)
- Market news and headlines
- Automatically enabled if `finvizfinance` package is installed

### 2. **NewsAPI** (newsapi.org)
- General news with financial focus
- **Requires API Key**
- Free tier: 100 requests/day
- Get key at: https://newsapi.org/register

### 3. **Alpha Vantage** (alphavantage.co)
- Financial news with sentiment analysis
- **Requires API Key**
- Free tier: 25 requests/day
- Get key at: https://www.alphavantage.co/support/#api-key

### 4. **MarketWatch** (Dow Jones)
- RSS feed for top market stories
- No API key required
- Automatically enabled

## Setup Instructions

### Option 1: Environment Variables (Recommended)

Add these to your `.env` file or export in your shell:

```bash
# NewsAPI (Get free key from newsapi.org)
export NEWSAPI_KEY="your_newsapi_key_here"

# Alpha Vantage (Get free key from alphavantage.co)
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key_here"
```

### Option 2: Direct in Code (Not Recommended for Production)

Edit `services/news_service.py`:

```python
@dataclass
class NewsService:
    source: str = 'multi'
    NEWSAPI_KEY: Optional[str] = "your_key_here"
    ALPHA_VANTAGE_KEY: Optional[str] = "your_key_here"
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
