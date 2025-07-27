"""
News and sentiment analysis API interface.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from ..base.config import config
from ..base.exceptions import APIError, DataError


class NewsSentimentAPI:
    """
    API for fetching news data and performing sentiment analysis.
    """
    
    def __init__(self):
        self.alpha_vantage_key = config.alpha_vantage_api_key
        self.base_url = config.get("market_data.alpha_vantage.base_url")
        self.logger = logger.bind(service="news_sentiment_api")
        
    async def get_news_sentiment(self, tickers: List[str] = None, 
                               topics: List[str] = None,
                               time_from: datetime = None,
                               time_to: datetime = None,
                               limit: int = 50) -> Dict[str, Any]:
        """
        Get news sentiment data.
        
        Args:
            tickers: List of tickers to filter news
            topics: List of topics to filter news  
            time_from: Start time for news
            time_to: End time for news
            limit: Maximum number of articles
            
        Returns:
            Dictionary with news sentiment data
        """
        if not self.alpha_vantage_key:
            raise APIError("Alpha Vantage API key not configured")
            
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.alpha_vantage_key,
                'limit': min(limit, 1000)  # API limit
            }
            
            # Add optional filters
            if tickers:
                params['tickers'] = ','.join(tickers)
                
            if topics:
                params['topics'] = ','.join(topics)
                
            if time_from:
                params['time_from'] = time_from.strftime('%Y%m%dT%H%M')
                
            if time_to:
                params['time_to'] = time_to.strftime('%Y%m%dT%H%M')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'feed' not in data:
                raise DataError("No news sentiment data found")
            
            return data
            
        except Exception as e:
            raise APIError(f"News sentiment API error: {str(e)}")
    
    async def get_company_news(self, symbol: str, 
                             days_back: int = 7,
                             limit: int = 20) -> pd.DataFrame:
        """
        Get recent news for a specific company.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            limit: Maximum number of articles
            
        Returns:
            DataFrame with news articles
        """
        try:
            time_to = datetime.utcnow()
            time_from = time_to - timedelta(days=days_back)
            
            news_data = await self.get_news_sentiment(
                tickers=[symbol],
                time_from=time_from,
                time_to=time_to,
                limit=limit
            )
            
            if not news_data.get('feed'):
                return pd.DataFrame()
            
            # Convert to DataFrame
            articles = []
            for article in news_data['feed']:
                # Extract relevant ticker sentiment
                ticker_sentiments = {}
                if 'ticker_sentiment' in article:
                    for ticker_data in article['ticker_sentiment']:
                        if ticker_data['ticker'] == symbol:
                            ticker_sentiments = {
                                'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                'sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                'sentiment_label': ticker_data.get('ticker_sentiment_label', 'Neutral')
                            }
                            break
                
                article_data = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'authors': ', '.join(article.get('authors', [])),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'category_within_source': article.get('category_within_source', ''),
                    'source_domain': article.get('source_domain', ''),
                    'topics': ', '.join([topic['topic'] for topic in article.get('topics', [])]),
                    'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                    'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                    **ticker_sentiments
                }
                
                articles.append(article_data)
            
            df = pd.DataFrame(articles)
            if not df.empty:
                df['time_published'] = pd.to_datetime(df['time_published'])
                df = df.sort_values('time_published', ascending=False)
            
            return df
            
        except Exception as e:
            raise APIError(f"Company news API error for {symbol}: {str(e)}")
    
    async def get_market_sentiment_summary(self, symbols: List[str],
                                         days_back: int = 7) -> Dict[str, Any]:
        """
        Get sentiment summary for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sentiment summary
        """
        try:
            time_to = datetime.utcnow()
            time_from = time_to - timedelta(days=days_back)
            
            # Get news for all symbols
            news_data = await self.get_news_sentiment(
                tickers=symbols,
                time_from=time_from,
                time_to=time_to,
                limit=200
            )
            
            if not news_data.get('feed'):
                return {'symbols': symbols, 'sentiment_summary': {}}
            
            # Analyze sentiment by symbol
            symbol_sentiments = {}
            
            for symbol in symbols:
                symbol_articles = []
                
                for article in news_data['feed']:
                    if 'ticker_sentiment' in article:
                        for ticker_data in article['ticker_sentiment']:
                            if ticker_data['ticker'] == symbol:
                                symbol_articles.append({
                                    'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                    'sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                    'sentiment_label': ticker_data.get('ticker_sentiment_label', 'Neutral'),
                                    'time_published': article.get('time_published', '')
                                })
                
                if symbol_articles:
                    # Calculate aggregate metrics
                    sentiment_scores = [a['sentiment_score'] for a in symbol_articles]
                    relevance_scores = [a['relevance_score'] for a in symbol_articles]
                    
                    # Weight sentiment by relevance
                    if relevance_scores and sum(relevance_scores) > 0:
                        weighted_sentiment = sum(s * r for s, r in zip(sentiment_scores, relevance_scores)) / sum(relevance_scores)
                    else:
                        weighted_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                    
                    # Count sentiment labels
                    sentiment_labels = [a['sentiment_label'] for a in symbol_articles]
                    label_counts = {
                        'Bullish': sentiment_labels.count('Bullish'),
                        'Bearish': sentiment_labels.count('Bearish'),
                        'Neutral': sentiment_labels.count('Neutral')
                    }
                    
                    symbol_sentiments[symbol] = {
                        'article_count': len(symbol_articles),
                        'average_sentiment_score': sum(sentiment_scores) / len(sentiment_scores),
                        'weighted_sentiment_score': weighted_sentiment,
                        'average_relevance_score': sum(relevance_scores) / len(relevance_scores),
                        'sentiment_distribution': label_counts,
                        'dominant_sentiment': max(label_counts, key=label_counts.get),
                        'sentiment_strength': max(label_counts.values()) / sum(label_counts.values()) if sum(label_counts.values()) > 0 else 0
                    }
                else:
                    symbol_sentiments[symbol] = {
                        'article_count': 0,
                        'average_sentiment_score': 0,
                        'weighted_sentiment_score': 0,
                        'average_relevance_score': 0,
                        'sentiment_distribution': {'Bullish': 0, 'Bearish': 0, 'Neutral': 0},
                        'dominant_sentiment': 'Neutral',
                        'sentiment_strength': 0
                    }
            
            # Calculate market-wide sentiment
            all_scores = [s['weighted_sentiment_score'] for s in symbol_sentiments.values() if s['article_count'] > 0]
            market_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0
            
            return {
                'symbols': symbols,
                'analysis_period': f"{time_from.date()} to {time_to.date()}",
                'market_sentiment_score': market_sentiment,
                'total_articles': len(news_data['feed']),
                'symbol_sentiments': symbol_sentiments,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            raise APIError(f"Market sentiment summary error: {str(e)}")
    
    def classify_sentiment_signal(self, sentiment_score: float, 
                                strength: float = 0.5) -> str:
        """
        Classify sentiment score into trading signal.
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            strength: Minimum strength threshold
            
        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD')
        """
        if abs(sentiment_score) < 0.1:  # Near neutral
            return 'HOLD'
        elif sentiment_score > 0.2 and strength > 0.5:  # Strong positive sentiment
            return 'BUY'
        elif sentiment_score < -0.2 and strength > 0.5:  # Strong negative sentiment
            return 'SELL'
        else:
            return 'HOLD'
    
    async def get_sector_sentiment(self, sector_symbols: Dict[str, List[str]],
                                 days_back: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis by sector.
        
        Args:
            sector_symbols: Dictionary mapping sector names to symbol lists
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sector sentiment analysis
        """
        try:
            sector_sentiments = {}
            
            for sector, symbols in sector_symbols.items():
                self.logger.info(f"Analyzing sentiment for {sector} sector")
                
                sector_summary = await self.get_market_sentiment_summary(symbols, days_back)
                
                # Calculate sector-level metrics
                symbol_sentiments = sector_summary['symbol_sentiments']
                
                if symbol_sentiments:
                    sector_scores = [s['weighted_sentiment_score'] for s in symbol_sentiments.values() if s['article_count'] > 0]
                    sector_sentiment_score = sum(sector_scores) / len(sector_scores) if sector_scores else 0
                    
                    total_articles = sum(s['article_count'] for s in symbol_sentiments.values())
                    
                    # Count dominant sentiments
                    dominant_sentiments = [s['dominant_sentiment'] for s in symbol_sentiments.values() if s['article_count'] > 0]
                    sector_sentiment_distribution = {
                        'Bullish': dominant_sentiments.count('Bullish'),
                        'Bearish': dominant_sentiments.count('Bearish'),
                        'Neutral': dominant_sentiments.count('Neutral')
                    }
                    
                    sector_sentiments[sector] = {
                        'sector_sentiment_score': sector_sentiment_score,
                        'total_articles': total_articles,
                        'symbols_analyzed': len([s for s in symbol_sentiments.values() if s['article_count'] > 0]),
                        'sector_sentiment_distribution': sector_sentiment_distribution,
                        'symbol_details': symbol_sentiments
                    }
            
            return {
                'analysis_period': f"{datetime.utcnow() - timedelta(days=days_back)} to {datetime.utcnow()}",
                'sector_sentiments': sector_sentiments,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            raise APIError(f"Sector sentiment analysis error: {str(e)}")
    
    async def get_trending_topics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trending topics from recent news.
        
        Args:
            limit: Maximum number of articles to analyze
            
        Returns:
            List of trending topics with scores
        """
        try:
            # Get recent news without specific filters
            news_data = await self.get_news_sentiment(limit=limit)
            
            if not news_data.get('feed'):
                return []
            
            # Extract and count topics
            topic_counts = {}
            topic_sentiments = {}
            
            for article in news_data['feed']:
                overall_sentiment = float(article.get('overall_sentiment_score', 0))
                
                for topic_data in article.get('topics', []):
                    topic = topic_data['topic']
                    relevance = float(topic_data.get('relevance_score', 0))
                    
                    if topic not in topic_counts:
                        topic_counts[topic] = 0
                        topic_sentiments[topic] = []
                    
                    topic_counts[topic] += 1
                    topic_sentiments[topic].append(overall_sentiment * relevance)
            
            # Calculate topic scores
            trending_topics = []
            for topic, count in topic_counts.items():
                if count >= 2:  # Minimum mentions threshold
                    avg_sentiment = sum(topic_sentiments[topic]) / len(topic_sentiments[topic])
                    
                    trending_topics.append({
                        'topic': topic,
                        'mention_count': count,
                        'average_sentiment': avg_sentiment,
                        'trending_score': count * (1 + abs(avg_sentiment))  # Weight by mentions and sentiment strength
                    })
            
            # Sort by trending score
            trending_topics.sort(key=lambda x: x['trending_score'], reverse=True)
            
            return trending_topics[:20]  # Return top 20 trending topics
            
        except Exception as e:
            raise APIError(f"Trending topics analysis error: {str(e)}")