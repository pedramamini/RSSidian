import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import Config
from .models import get_db_session, Feed, Article
from .core import RSSProcessor
from .mcp import ModelContextProtocol


# Pydantic models for API
class FeedResponse(BaseModel):
    id: int
    title: str
    url: str
    description: Optional[str] = None
    website_url: Optional[str] = None
    last_updated: Optional[str] = None
    muted: bool
    peer_through: bool


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
                "muted": feed.muted,
                "peer_through": feed.peer_through
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
    
    # Enable peer-through for a feed
    @app.post("/api/v1/subscriptions/{feed_title}/enable-peer-through", response_model=SuccessResponse, tags=["Subscriptions"])
    def enable_peer_through(feed_title: str, db: Session = Depends(get_db)):
        """
        Enable peer-through for an aggregator feed to fetch origin article content.
        
        - **feed_title**: Title of the feed
        """
        feed = db.query(Feed).filter_by(title=feed_title).first()
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        feed.peer_through = True
        db.commit()
        
        return {"success": True, "message": f"Peer-through enabled for feed '{feed_title}'"}
    
    # Disable peer-through for a feed
    @app.post("/api/v1/subscriptions/{feed_title}/disable-peer-through", response_model=SuccessResponse, tags=["Subscriptions"])
    def disable_peer_through(feed_title: str, db: Session = Depends(get_db)):
        """
        Disable peer-through for a feed.
        
        - **feed_title**: Title of the feed
        """
        feed = db.query(Feed).filter_by(title=feed_title).first()
        if not feed:
            raise HTTPException(status_code=404, detail="Feed not found")
        
        feed.peer_through = False
        db.commit()
        
        return {"success": True, "message": f"Peer-through disabled for feed '{feed_title}'"}
    
    # MCP Endpoints
    
    # MCP - Discovery endpoint
    @app.get("/api/v1/mcp", tags=["MCP"])
    def mcp_discovery():
        """
        Discovery endpoint for the Model Context Protocol service.
        
        Returns information about available endpoints and capabilities.
        """
        return {
            "name": "RSSidian MCP",
            "version": "1.0.0",
            "description": "Model Context Protocol for RSSidian RSS feed manager",
            "capabilities": [
                "list_subscriptions",
                "list_articles",
                "get_article_content",
                "search_articles",
                "get_digest",
                "get_feed_stats",
                "natural_language_query"
            ],
            "endpoints": {
                "discovery": "/api/v1/mcp",
                "subscriptions": "/api/v1/mcp/subscriptions",
                "articles": "/api/v1/mcp/articles",
                "article_content": "/api/v1/mcp/articles/{article_id}/content",
                "search": "/api/v1/mcp/search",
                "digest": "/api/v1/mcp/digest",
                "feed_stats": "/api/v1/mcp/feed-stats",
                "query": "/api/v1/mcp/query"
            }
        }
    
    # MCP - List subscriptions
    @app.get("/api/v1/mcp/subscriptions", tags=["MCP"])
    def mcp_list_subscriptions(db: Session = Depends(get_db)):
        """
        List all feed subscriptions with MCP format.
        """
        mcp = ModelContextProtocol(config, db)
        return mcp.get_subscriptions()
    
    # MCP - List articles
    @app.get("/api/v1/mcp/articles", tags=["MCP"])
    def mcp_list_articles(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        feed_title: Optional[str] = None,
        days_back: Optional[int] = None,
        min_quality_tier: Optional[str] = None,
        db: Session = Depends(get_db)
    ):
        """
        List articles with MCP format and advanced filtering.
        
        - **limit**: Maximum number of articles to return
        - **offset**: Number of articles to skip
        - **feed_title**: Filter by feed title
        - **days_back**: Only return articles from the last N days
        - **min_quality_tier**: Minimum quality tier (S, A, B, C, D)
        """
        mcp = ModelContextProtocol(config, db)
        return mcp.get_articles(limit, offset, feed_title, days_back, min_quality_tier)
    
    # MCP - Get article content
    @app.get("/api/v1/mcp/articles/{article_id}/content", tags=["MCP"])
    def mcp_get_article_content(article_id: int, db: Session = Depends(get_db)):
        """
        Get the full content of an article with MCP format.
        
        - **article_id**: ID of the article
        """
        mcp = ModelContextProtocol(config, db)
        result = mcp.get_article_content(article_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    
    # MCP - Search articles
    @app.get("/api/v1/mcp/search", tags=["MCP"])
    def mcp_search_articles(
        query: str,
        relevance: float = Query(0.3, ge=0, le=1),
        max_results: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db)
    ):
        """
        Search for articles using semantic search with MCP format.
        
        - **query**: Search query
        - **relevance**: Minimum relevance threshold (0-1)
        - **max_results**: Maximum number of results to return
        """
        mcp = ModelContextProtocol(config, db)
        return mcp.search_articles(query, relevance, max_results)
    
    # MCP - Get digest
    @app.get("/api/v1/mcp/digest", tags=["MCP"])
    def mcp_get_digest(
        days_back: int = Query(7, ge=1, le=90),
        db: Session = Depends(get_db)
    ):
        """
        Get a digest of high-value articles from the specified period.
        
        - **days_back**: Number of days to look back
        """
        mcp = ModelContextProtocol(config, db)
        return mcp.get_digest(days_back)
    
    # MCP - Get feed stats
    @app.get("/api/v1/mcp/feed-stats", tags=["MCP"])
    def mcp_get_feed_stats(
        feed_title: Optional[str] = None,
        db: Session = Depends(get_db)
    ):
        """
        Get statistics for feeds.
        
        - **feed_title**: Optional feed title to get stats for a specific feed
        """
        mcp = ModelContextProtocol(config, db)
        result = mcp.get_feed_stats(feed_title)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    
    # MCP - Query endpoint for AI-based interaction
    @app.post("/api/v1/mcp/query", tags=["MCP"])
    async def mcp_query(
        query: Dict[str, Any],
        db: Session = Depends(get_db)
    ):
        """
        Process a natural language query about RSS content.
        
        The query should be a JSON object with the following structure:
        ```json
        {
            "query": "What are the latest articles about AI?",
            "context": {
                "relevance_threshold": 0.3,
                "max_results": 20,
                "days_back": 30
            }
        }
        ```
        
        The context is optional and can contain parameters to customize the query.
        """
        # Extract query and context
        user_query = query.get("query", "")
        context = query.get("context", {})
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Extract context parameters with defaults
        relevance_threshold = context.get("relevance_threshold", 0.3)
        max_results = context.get("max_results", 20)
        days_back = context.get("days_back", 30)
        
        # Initialize MCP
        mcp = ModelContextProtocol(config, db)
        
        # Search for relevant articles
        search_results = mcp.search_articles(user_query, relevance_threshold, max_results)
        
        # Get feed statistics
        feed_stats = mcp.get_feed_stats()
        
        # Return comprehensive response
        return {
            "query": user_query,
            "search_results": search_results,
            "feed_stats": feed_stats,
            "context": {
                "relevance_threshold": relevance_threshold,
                "max_results": max_results,
                "days_back": days_back
            }
        }
    
    return app