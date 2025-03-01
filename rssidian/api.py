import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import Config
from .models import get_db_session, Feed, Article
from .core import RSSProcessor


# Pydantic models for API
class FeedResponse(BaseModel):
    id: int
    title: str
    url: str
    description: Optional[str] = None
    website_url: Optional[str] = None
    last_updated: Optional[str] = None
    muted: bool


class ArticleResponse(BaseModel):
    id: int
    feed_id: int
    feed_title: str
    title: str
    url: str
    published_at: Optional[str] = None
    summary: Optional[str] = None
    quality_tier: Optional[str] = None
    quality_score: Optional[int] = None
    labels: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class SuccessResponse(BaseModel):
    success: bool
    message: str


def create_api(config: Config) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="RSSidian API",
        description="API for RSSidian RSS feed manager",
        version="0.1.0"
    )
    
    # Dependency to get database session
    def get_db():
        db = get_db_session(config.db_path)
        try:
            yield db
        finally:
            db.close()
    
    # Search endpoint
    @app.get("/api/v1/search", response_model=SearchResponse, tags=["Search"])
    def search(
        query: str,
        relevance: float = Query(0.6, ge=0, le=1),
        max_results: int = Query(20, ge=1, le=100),
        refresh: bool = False,
        db: Session = Depends(get_db)
    ):
        """
        Search articles using semantic search.
        
        - **query**: Search query
        - **relevance**: Minimum relevance threshold (0-1)
        - **max_results**: Maximum number of results to return
        - **refresh**: Whether to refresh the index before searching
        """
        processor = RSSProcessor(config, db)
        results = processor.search(query, relevance, max_results, refresh)
        return {"query": query, "results": results}
    
    # List all articles
    @app.get("/api/v1/articles", response_model=List[ArticleResponse], tags=["Articles"])
    def list_articles(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        db: Session = Depends(get_db)
    ):
        """
        List articles with pagination.
        
        - **limit**: Maximum number of articles to return
        - **offset**: Number of articles to skip
        """
        articles = db.query(Article).order_by(Article.published_at.desc()).offset(offset).limit(limit).all()
        
        response = []
        for article in articles:
            feed = db.query(Feed).filter_by(id=article.feed_id).first()
            feed_title = feed.title if feed else "Unknown Feed"
            
            response.append({
                "id": article.id,
                "feed_id": article.feed_id,
                "feed_title": feed_title,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "summary": article.summary,
                "quality_tier": article.quality_tier,
                "quality_score": article.quality_score,
                "labels": article.labels
            })
        
        return response
    
    # Get article by ID
    @app.get("/api/v1/articles/{article_id}", response_model=Dict[str, Any], tags=["Articles"])
    def get_article(article_id: int, db: Session = Depends(get_db)):
        """
        Get article by ID.
        
        - **article_id**: ID of the article
        """
        article = db.query(Article).filter_by(id=article_id).first()
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        feed = db.query(Feed).filter_by(id=article.feed_id).first()
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
            "word_count": article.word_count
        }
    
    # List subscriptions
    @app.get("/api/v1/subscriptions", response_model=List[FeedResponse], tags=["Subscriptions"])
    def list_subscriptions(db: Session = Depends(get_db)):
        """List all feed subscriptions."""
        feeds = db.query(Feed).order_by(Feed.title).all()
        
        response = []
        for feed in feeds:
            response.append({
                "id": feed.id,
                "title": feed.title,
                "url": feed.url,
                "description": feed.description,
                "website_url": feed.website_url,
                "last_updated": feed.last_updated.isoformat() if feed.last_updated else None,
                "muted": feed.muted
            })
        
        return response
    
    # Mute subscription
    @app.post("/api/v1/subscriptions/{feed_title}/mute", response_model=SuccessResponse, tags=["Subscriptions"])
    def mute_subscription(feed_title: str, db: Session = Depends(get_db)):
        """
        Mute a feed subscription.
        
        - **feed_title**: Title of the feed
        """
        feed = db.query(Feed).filter_by(title=feed_title).first()
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        feed.muted = True
        db.commit()
        
        return {"success": True, "message": f"Feed '{feed_title}' has been muted"}
    
    # Unmute subscription
    @app.post("/api/v1/subscriptions/{feed_title}/unmute", response_model=SuccessResponse, tags=["Subscriptions"])
    def unmute_subscription(feed_title: str, db: Session = Depends(get_db)):
        """
        Unmute a feed subscription.
        
        - **feed_title**: Title of the feed
        """
        feed = db.query(Feed).filter_by(title=feed_title).first()
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        feed.muted = False
        db.commit()
        
        return {"success": True, "message": f"Feed '{feed_title}' has been unmuted"}
    
    return app