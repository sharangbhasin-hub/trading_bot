"""
News Fetcher & Sentiment Analysis
Migrated from old codebase - adapted for dynamic news sources
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

class NewsFetcher:
    """Fetch and analyze market news"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch latest news for a symbol
        
        Args:
            symbol: Stock/Index symbol (e.g., 'NIFTY', 'BANKNIFTY', 'RELIANCE')
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of news articles with title, source, date, sentiment
        """
        news_articles = []
        
        try:
            # Method 1: Google News RSS (Most reliable, no API key needed)
            search_query = f"{symbol} stock market india"
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:max_articles]
                
                for item in items:
                    try:
                        title = item.find('title').text if item.find('title') else 'No Title'
                        source = item.find('source').text if item.find('source') else 'Unknown'
                        pub_date = item.find('pubDate').text if item.find('pubDate') else 'Unknown'
                        link = item.find('link').text if item.find('link') else '#'
                        
                        # Basic sentiment analysis
                        sentiment = self._analyze_sentiment(title)
                        
                        news_articles.append({
                            'title': title,
                            'source': source,
                            'date': pub_date,
                            'link': link,
                            'sentiment': sentiment
                        })
                    except Exception as e:
                        print(f"Error parsing article: {e}")
                        continue
            
            # Fallback: If no news found, add placeholder
            if not news_articles:
                news_articles.append({
                    'title': f'No recent news found for {symbol}',
                    'source': 'System',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'link': '#',
                    'sentiment': 'Neutral'
                })
        
        except Exception as e:
            print(f"❌ News fetch error: {e}")
            news_articles.append({
                'title': 'Unable to fetch news at this time',
                'source': 'System',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'link': '#',
                'sentiment': 'Neutral'
            })
        
        return news_articles
    
    def _analyze_sentiment(self, text: str) -> str:
        """
        Simple keyword-based sentiment analysis
        (Replaces transformer model for lightweight operation)
        """
        text_lower = text.lower()
        
        # Positive keywords
        positive_keywords = [
            'surge', 'gain', 'rise', 'up', 'rally', 'bullish', 'profit', 
            'growth', 'positive', 'strong', 'high', 'record', 'breakthrough',
            'success', 'advance', 'jump', 'soar', 'boost', 'upgrade'
        ]
        
        # Negative keywords
        negative_keywords = [
            'fall', 'drop', 'decline', 'down', 'crash', 'bearish', 'loss',
            'weak', 'negative', 'low', 'concern', 'risk', 'warning',
            'plunge', 'slide', 'tumble', 'slump', 'downgrade', 'fear'
        ]
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_summary(self, news_articles: List[Dict]) -> Dict:
        """
        Calculate sentiment breakdown from news articles
        
        Returns:
            Dict with sentiment counts and percentages
        """
        if not news_articles:
            return {
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0,
                'total': 0
            }
        
        sentiments = [article['sentiment'] for article in news_articles]
        total = len(sentiments)
        
        positive = sentiments.count('Positive')
        negative = sentiments.count('Negative')
        neutral = sentiments.count('Neutral')
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0,
            'neutral_pct': round((neutral / total) * 100, 1) if total > 0 else 0,
            'total': total
        }
