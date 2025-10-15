"""
News Fetcher - Based on working old code with NewsAPI + Sentiment Analysis
"""
import requests
from typing import List, Dict
import os

class NewsFetcher:
    """Fetch and analyze news using NewsAPI.org"""
    
    def __init__(self, api_key: str = None):
        # Use provided API key, environment variable, or fallback to old working key
        self.api_key = api_key or os.getenv('NEWS_API_KEY', 'e205d77d7bc14acc8744d3ea10568f50')
        
        # Sentiment keywords (fallback if transformer not available)
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
            'fear', 'recession', 'downgrade', 'sell', 'pessimistic', 'caution'
        ]
    
    def fetch_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch news headlines using NewsAPI (same as old code)
        
        Args:
            symbol: Ticker symbol (e.g., 'NIFTY', 'BANKNIFTY')
            max_articles: Number of articles to fetch
        
        Returns:
            List of news articles with sentiment
        """
        try:
            # Clean ticker name (same as old code)
            search_query = symbol.replace('NSE:', '').replace('BSE:', '').replace('^', '').replace('.NS', '').strip()
            
            # Enhance query for Indian markets
            if 'NIFTY' in search_query.upper():
                search_query = "Nifty OR NSE India"
            elif 'SENSEX' in search_query.upper():
                search_query = "Sensex OR BSE India"
            
            # Build API URL (exactly like old code)
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={self.api_key}&pageSize={max_articles}"
            
            headers = {"User-Agent": "Mozilla/5.0"}
            
            print(f"📰 Fetching news for: {search_query}")
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            news_data = response.json()
            
            articles = []
            
            if news_data.get("status") == "ok" and news_data.get("articles"):
                for article in news_data["articles"]:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    source = article.get("source", {}).get("name", "Unknown")
                    published = article.get("publishedAt", "")
                    url = article.get("url", "#")
                    
                    # Only include articles with meaningful titles
                    if title and len(title) > 15:
                        # Format date
                        date = published[:10] if published else "Unknown"
                        
                        # Analyze sentiment
                        sentiment = self.analyze_sentiment(title + " " + (description or ""))
                        
                        articles.append({
                            'title': title,
                            'source': source,
                            'date': date,
                            'sentiment': sentiment,
                            'link': url
                        })
                    
                    if len(articles) >= max_articles:
                        break
                
                if articles:
                    print(f"✅ Fetched {len(articles)} articles successfully")
                    return articles
            
            print("⚠️ No articles found")
            return self._get_placeholder(symbol)
        
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return self._get_placeholder(symbol)
    
    def _get_placeholder(self, symbol: str) -> List[Dict]:
        """Return placeholder when fetch fails"""
        from datetime import datetime
        return [{
            'title': f'No recent news found for {symbol}',
            'source': 'System',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'sentiment': 'Neutral',
            'link': '#'
        }]
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment using keyword matching
        (Simplified version - old code used transformer model)
        
        Args:
            text: Text to analyze
        
        Returns:
            'Positive', 'Negative', or 'Neutral'
        """
        if not text or len(text) < 15:
            return 'Neutral'
        
        text_lower = text.lower()
        
        # Count keyword occurrences
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """
        Calculate sentiment statistics
        
        Args:
            articles: List of articles with sentiment
        
        Returns:
            Dictionary with sentiment counts and percentages
        """
        if not articles:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'neutral_pct': 0.0
            }
        
        total = len(articles)
        positive = sum(1 for a in articles if a.get('sentiment') == 'Positive')
        negative = sum(1 for a in articles if a.get('sentiment') == 'Negative')
        neutral = sum(1 for a in articles if a.get('sentiment') == 'Neutral')
        
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0.0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0.0,
            'neutral_pct': round((neutral / total) * 100, 1) if total > 0 else 0.0
        }
