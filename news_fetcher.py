"""
News Fetcher using yfinance (Yahoo Finance) - More Reliable
"""
import yfinance as yf
from datetime import datetime

class NewsFetcher:
    """Fetch and analyze news using Yahoo Finance API"""
    
    def __init__(self):
        # Market-specific sentiment keywords
        self.positive_keywords = [
            'surge', 'rally', 'gain', 'profit', 'growth', 'bullish', 'upside', 
            'record', 'high', 'breakthrough', 'strong', 'outperform', 'beat',
            'exceed', 'positive', 'rise', 'soar', 'jump', 'climb', 'advance',
            'rebound', 'recover', 'boost', 'upgrade', 'buy', 'optimistic'
        ]
        
        self.negative_keywords = [
            'fall', 'drop', 'loss', 'decline', 'crash', 'bearish', 'downside',
            'concern', 'worry', 'risk', 'weak', 'underperform', 'miss',
            'warning', 'negative', 'plunge', 'tumble', 'sink', 'slide', 'slump',
            'fear', 'recession', 'downgrade', 'sell', 'pessimistic', 'caution',
            'lower', 'cut', 'reduce'
        ]
        
        # Symbol mapping for yfinance (NSE indices)
        self.symbol_map = {
            'NIFTY': '^NSEI',
            'NIFTY 50': '^NSEI',
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'BANK NIFTY': '^NSEBANK',
            'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
            'SENSEX': '^BSESN',
            'MIDCPNIFTY': '^NSMIDCP'
        }
    
    def fetch_news(self, symbol: str, max_articles: int = 10) -> list:
        """
        Fetch news using yfinance
        
        Args:
            symbol: Stock/Index symbol (e.g., 'NIFTY', 'BANKNIFTY', 'RELIANCE')
            max_articles: Maximum number of articles
        
        Returns:
            List of news articles with sentiment
        """
        try:
            # Clean and map symbol
            clean_symbol = symbol.replace('NSE:', '').replace('BSE:', '').strip().upper()
            
            # Use mapped symbol if available, otherwise try as-is
            yf_symbol = self.symbol_map.get(clean_symbol, clean_symbol)
            
            print(f"📰 Fetching news for {clean_symbol} (yfinance symbol: {yf_symbol})")
            
            # Create ticker object
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch news
            news_data = ticker.news
            
            if not news_data or len(news_data) == 0:
                # Try alternative: Search for India market news
                return self._fetch_market_news_fallback(clean_symbol, max_articles)
            
            # Process news articles
            articles = []
            for item in news_data[:max_articles]:
                # Extract data
                title = item.get('title', 'No title')
                publisher = item.get('publisher', 'Unknown')
                
                # Convert timestamp to readable date
                timestamp = item.get('providerPublishTime', 0)
                if timestamp:
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                else:
                    date = 'Unknown'
                
                link = item.get('link', '#')
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(title)
                
                article = {
                    'title': title,
                    'source': publisher,
                    'date': date,
                    'sentiment': sentiment,
                    'link': link
                }
                articles.append(article)
            
            print(f"✅ Successfully fetched {len(articles)} articles")
            return articles
        
        except Exception as e:
            print(f"⚠️ yfinance error: {e}")
            # Return fallback news
            return self._fetch_market_news_fallback(symbol, max_articles)
    
    def _fetch_market_news_fallback(self, symbol: str, max_articles: int) -> list:
        """
        Fallback: Fetch general Indian market news
        Uses Nifty 50 as proxy for market news
        """
        try:
            print(f"📰 Fetching general market news as fallback...")
            
            # Use Nifty 50 index for general market news
            ticker = yf.Ticker('^NSEI')
            news_data = ticker.news
            
            if not news_data:
                raise Exception("No fallback news available")
            
            articles = []
            for item in news_data[:max_articles]:
                title = item.get('title', 'No title')
                publisher = item.get('publisher', 'Unknown')
                timestamp = item.get('providerPublishTime', 0)
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp else 'Unknown'
                link = item.get('link', '#')
                sentiment = self.analyze_sentiment(title)
                
                article = {
                    'title': title,
                    'source': publisher,
                    'date': date,
                    'sentiment': sentiment,
                    'link': link
                }
                articles.append(article)
            
            print(f"✅ Fetched {len(articles)} market news articles (fallback)")
            return articles
        
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            # Return placeholder
            return [{
                'title': f'Unable to fetch news for {symbol} at this time',
                'source': 'System Message',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'sentiment': 'Neutral',
                'link': '#'
            }]
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment using keyword matching
        
        Args:
            text: Text to analyze (title + description)
        
        Returns:
            'Positive', 'Negative', or 'Neutral'
        """
        if not text:
            return 'Neutral'
        
        text_lower = text.lower()
        
        # Count keyword matches
        positive_score = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_score = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Determine sentiment
        if positive_score > negative_score:
            return 'Positive'
        elif negative_score > positive_score:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_summary(self, articles: list) -> dict:
        """
        Calculate sentiment summary statistics
        
        Args:
            articles: List of news articles
        
        Returns:
            Dict with sentiment counts and percentages
        """
        if not articles:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0
            }
        
        positive = sum(1 for a in articles if a.get('sentiment') == 'Positive')
        negative = sum(1 for a in articles if a.get('sentiment') == 'Negative')
        neutral = sum(1 for a in articles if a.get('sentiment') == 'Neutral')
        total = len(articles)
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0,
            'neutral_pct': round((neutral / total) * 100, 1) if total > 0 else 0
        }
