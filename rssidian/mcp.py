"""
Model Context Protocol (MCP) implementation for RSSidian.

This module provides a comprehensive interface for AI models to interact with
RSSidian's content, enabling natural language queries and responses about
RSS feed content.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .core import RSSProcessor
from .models import Feed, Article
from .config import Config


class ModelContextProtocol:
    """
    Model Context Protocol implementation for RSSidian.
    
    This class provides methods for AI models to interact with RSSidian's content,
    enabling natural language queries and responses about RSS feed content.
    """
    
    def __init__(self, config: Config, db_session: Session):
        """Initialize the MCP with configuration and database session."""
        self.config = config
        self.db_session = db_session
        self.processor = RSSProcessor(config, db_session)
    
    def get_subscriptions(self) -> List[Dict[str, Any]]:
        """
        Get a list of all feed subscriptions.
        
        Returns:
            List of feed subscription details
        """
        feeds = self.db_session.query(Feed).order_by(Feed.title).all()
        
        result = []
        for feed in feeds:
            result.append({
                "id": feed.id,
                "title": feed.title,
                "url": feed.url,
                "description": feed.description,
                "website_url": feed.website_url,
                "last_updated": feed.last_updated.isoformat() if feed.last_updated else None,
                "muted": feed.muted,
                "peer_through": feed.peer_through,
                "article_count": feed.article_count
            })
        
        return result
    
    def get_articles(self, 
                    limit: int = 50, 
                    offset: int = 0, 
                    feed_title: Optional[str] = None,
                    days_back: Optional[int] = None,
                    min_quality_tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of articles with optional filtering.
        
        Args:
            limit: Maximum number of articles to return
            offset: Number of articles to skip
            feed_title: Filter by feed title
            days_back: Only return articles from the last N days
            min_quality_tier: Minimum quality tier (S, A, B, C, D)
            
        Returns:
            List of article details
        """
        query = self.db_session.query(Article)
        
        # Apply filters
        if feed_title:
            feed = self.db_session.query(Feed).filter_by(title=feed_title).first()
            if feed:
                query = query.filter_by(feed_id=feed.id)
        
        if days_back:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(Article.published_at >= start_date)
        
        if min_quality_tier:
            query = query.filter(Article.quality_tier <= min_quality_tier)  # S < A < B < C < D
        
        # Order by published date (newest first)
        query = query.order_by(Article.published_at.desc())
        
        # Apply pagination
        articles = query.offset(offset).limit(limit).all()
        
        result = []
        for article in articles:
            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
            feed_title = feed.title if feed else "Unknown Feed"
            
            result.append({
                "id": article.id,
                "feed_id": article.feed_id,
                "feed_title": feed_title,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "summary": article.summary,
                "quality_tier": article.quality_tier,
                "quality_score": article.quality_score,
                "labels": article.labels,
                "word_count": article.word_count
            })
        
        return result
    
    def get_article_content(self, article_id: int) -> Dict[str, Any]:
        """
        Get the full content of an article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            Article details including full content
        """
        article = self.db_session.query(Article).filter_by(id=article_id).first()
        if not article:
            return {"error": "Article not found"}
        
        feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
        feed_title = feed.title if feed else "Unknown Feed"
        
        return {
            "id": article.id,
            "feed_id": article.feed_id,
            "feed_title": feed_title,
            "title": article.title,
            "url": article.url,
            "guid": article.guid,
            "description": article.description,
            "content": article.content,
            "author": article.author,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "fetched_at": article.fetched_at.isoformat() if article.fetched_at else None,
            "processed_at": article.processed_at.isoformat() if article.processed_at else None,
            "summary": article.summary,
            "quality_tier": article.quality_tier,
            "quality_score": article.quality_score,
            "labels": article.labels,
            "word_count": article.word_count,
            "jina_enhanced": article.jina_enhanced
        }
    
    def search_articles(self, 
                       query: str, 
                       relevance_threshold: float = 0.3, 
                       max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search for articles using semantic search.
        
        Args:
            query: Search query
            relevance_threshold: Minimum relevance score (0-1)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        return self.processor.search(query, relevance_threshold, max_results)
    
    def get_digest(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get a digest of high-value articles from the specified period.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Digest content
        """
        return self.processor.generate_digest(days_back)
    
    def get_feed_stats(self, feed_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for feeds.
        
        Args:
            feed_title: Optional feed title to get stats for a specific feed
            
        Returns:
            Feed statistics
        """
        if feed_title:
            feed = self.db_session.query(Feed).filter_by(title=feed_title).first()
            if not feed:
                return {"error": "Feed not found"}
            
            # Parse quality tier counts
            tier_counts = {}
            if feed.quality_tier_counts:
                try:
                    tier_counts = json.loads(feed.quality_tier_counts)
                except json.JSONDecodeError:
                    tier_counts = {}
            
            return {
                "id": feed.id,
                "title": feed.title,
                "article_count": feed.article_count,
                "avg_quality_score": feed.avg_quality_score,
                "quality_tier_counts": tier_counts,
                "last_updated": feed.last_updated.isoformat() if feed.last_updated else None
            }
        else:
            # Get stats for all feeds
            feeds = self.db_session.query(Feed).all()
            result = {
                "total_feeds": len(feeds),
                "total_articles": sum(feed.article_count or 0 for feed in feeds),
                "feeds": []
            }
            
            for feed in feeds:
                tier_counts = {}
                if feed.quality_tier_counts:
                    try:
                        tier_counts = json.loads(feed.quality_tier_counts)
                    except json.JSONDecodeError:
                        tier_counts = {}
                
                result["feeds"].append({
                    "id": feed.id,
                    "title": feed.title,
                    "article_count": feed.article_count,
                    "avg_quality_score": feed.avg_quality_score,
                    "quality_tier_counts": tier_counts
                })
            
            return result
