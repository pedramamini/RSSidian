import feedparser
import requests
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from tqdm import tqdm
import logging

from .models import Feed, Article

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_html(html_content: str) -> str:
    """Remove HTML tags from content."""
    # Simple regex to remove HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', html_content)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def extract_content(entry: Dict[str, Any]) -> str:
    """Extract the main content from a feed entry."""
    # Try different content fields in order of preference
    if 'content' in entry and entry['content']:
        for content in entry['content']:
            if 'value' in content:
                return content['value']
    
    if 'summary_detail' in entry and 'value' in entry['summary_detail']:
        return entry['summary_detail']['value']
    
    if 'summary' in entry:
        return entry['summary']
    
    if 'description' in entry:
        return entry['description']
    
    return ""


def parse_published_date(entry: Dict[str, Any]) -> Optional[datetime]:
    """Parse the published date from a feed entry."""
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if date_field in entry and entry[date_field]:
            try:
                time_struct = entry[date_field]
                return datetime.fromtimestamp(time.mktime(time_struct))
            except (TypeError, ValueError):
                continue
    
    # Try string date fields if parsed fields not available
    for date_field in ['published', 'updated', 'created']:
        if date_field in entry and entry[date_field]:
            try:
                return datetime.fromisoformat(entry[date_field].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                try:
                    # Try feedparser's own date parser
                    time_struct = feedparser._parse_date(entry[date_field])
                    if time_struct:
                        return datetime.fromtimestamp(time.mktime(time_struct))
                except:
                    continue
    
    return None


def get_feed_entries(feed_url: str, lookback_days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch and parse a feed, returning entries within lookback period.
    
    Args:
        feed_url: URL of the RSS feed
        lookback_days: Number of days to look back for articles
        
    Returns:
        List of feed entries within the lookback period
    """
    try:
        feed_data = feedparser.parse(feed_url)
        
        if hasattr(feed_data, 'status') and feed_data.status >= 400:
            logger.error(f"Error fetching feed {feed_url}: HTTP {feed_data.status}")
            return []
        
        if not feed_data.entries:
            logger.warning(f"No entries found in feed {feed_url}")
            return []
        
        # Calculate lookback date
        lookback_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Filter entries by date if possible
        recent_entries = []
        for entry in feed_data.entries:
            published_date = parse_published_date(entry)
            
            # If no date available, include the entry
            if not published_date:
                recent_entries.append(entry)
                continue
            
            # Include entry if it's within lookback period
            if published_date >= lookback_date:
                recent_entries.append(entry)
        
        return recent_entries
    
    except Exception as e:
        logger.error(f"Failed to process feed {feed_url}: {str(e)}")
        return []


def process_feed(
    feed: Feed, 
    db_session: Session, 
    lookback_days: int = 7,
    max_articles: int = 25,
    progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    """
    Process a feed, fetch new articles and store them in the database.
    
    Args:
        feed: Feed object
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles: Maximum number of articles to process per feed
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Number of new articles added
    """
    if feed.muted:
        if progress_callback:
            progress_callback(f"Skipping muted feed: {feed.title}")
        return 0
    
    if progress_callback:
        progress_callback(f"Processing feed: {feed.title}")
    
    try:
        entries = get_feed_entries(feed.url, lookback_days)
        
        # Update feed last checked timestamp
        feed.last_checked = datetime.utcnow()
        
        # Sort entries by date if possible, newest first
        sorted_entries = []
        for entry in entries:
            published_date = parse_published_date(entry)
            sorted_entries.append((entry, published_date or datetime.min))
        
        sorted_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max_articles
        sorted_entries = sorted_entries[:max_articles]
        
        new_article_count = 0
        
        for entry, published_date in sorted_entries:
            guid = entry.get('id', entry.get('link', ''))
            if not guid:
                continue
            
            # Check if article already exists
            existing_article = db_session.query(Article).filter_by(guid=guid).first()
            if existing_article:
                continue
            
            # Extract content
            content = extract_content(entry)
            clean_content = clean_html(content)
            
            # Create new article
            new_article = Article(
                feed_id=feed.id,  # Ensure feed_id is set correctly
                title=entry.get('title', 'Untitled'),
                url=entry.get('link', ''),
                guid=guid,
                description=entry.get('summary', ''),
                content=clean_content,
                author=entry.get('author', ''),
                published_at=published_date,
                fetched_at=datetime.utcnow(),
                processed=False,
                word_count=len(clean_content.split())
            )
            
            # Double-check that feed_id is set correctly
            if not new_article.feed_id:
                logger.warning(f"feed_id not set for article {new_article.title}. Setting manually.")
                new_article.feed_id = feed.id
            
            db_session.add(new_article)
            new_article_count += 1
        
        # Update feed last updated timestamp if new articles were found
        if new_article_count > 0:
            feed.last_updated = datetime.utcnow()
        
        # Reset error count if successful
        feed.error_count = 0
        feed.last_error = None
        
        db_session.commit()
        return new_article_count
    
    except Exception as e:
        logger.error(f"Error processing feed {feed.title}: {str(e)}")
        # Update error information
        feed.error_count += 1
        feed.last_error = str(e)
        db_session.commit()
        return 0


def process_all_feeds(
    db_session: Session, 
    lookback_days: int = 7,
    max_articles_per_feed: int = 25
) -> Dict[str, int]:
    """
    Process all active feeds in the database.
    
    Args:
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles_per_feed: Maximum number of articles to process per feed
        
    Returns:
        Dictionary with processing statistics
    """
    feeds = db_session.query(Feed).filter_by(muted=False).all()
    
    if not feeds:
        logger.warning("No active feeds found in database")
        return {"feeds_processed": 0, "new_articles": 0}
    
    total_new_articles = 0
    progress_bar = tqdm(feeds, desc="Processing feeds")
    
    for feed in progress_bar:
        progress_bar.set_description(f"Processing {feed.title[:30]}")
        new_articles = process_feed(
            feed,
            db_session, 
            lookback_days=lookback_days,
            max_articles=max_articles_per_feed,
            progress_callback=lambda msg: progress_bar.set_description(msg[:30])
        )
        total_new_articles += new_articles
    
    return {
        "feeds_processed": len(feeds),
        "new_articles": total_new_articles
    }