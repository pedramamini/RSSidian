import feedparser
import requests
import time
import re
import email.utils
import dateutil.parser
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
    # First try the pre-parsed date fields from feedparser
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if date_field in entry and entry[date_field]:
            try:
                time_struct = entry[date_field]
                # Validate the time struct to avoid out of range errors
                if time_struct and len(time_struct) >= 9:
                    year = time_struct[0]
                    # Basic validation to avoid mktime errors
                    if 1970 <= year <= 2100:  # Reasonable year range
                        try:
                            return datetime.fromtimestamp(time.mktime(time_struct))
                        except (OverflowError, ValueError) as e:
                            logger.debug(f"Time struct conversion error: {str(e)} for {time_struct}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Error parsing {date_field}: {str(e)}")
                continue
    
    # Try string date fields if parsed fields not available or failed
    for date_field in ['published', 'updated', 'created']:
        if date_field in entry and entry[date_field] and isinstance(entry[date_field], str):
            date_str = entry[date_field].strip()
            
            # Try multiple date parsing approaches
            parsers = [
                # ISO format with Z timezone
                lambda d: datetime.fromisoformat(d.replace('Z', '+00:00')),
                # RFC 2822 format using email.utils
                lambda d: datetime.fromtimestamp(time.mktime(email.utils.parsedate(d))),
                # Use dateutil parser as a fallback
                lambda d: dateutil.parser.parse(d),
                # Try feedparser's own date parser
                lambda d: datetime.fromtimestamp(time.mktime(feedparser._parse_date(d)))
            ]
            
            for parser in parsers:
                try:
                    return parser(date_str)
                except (ValueError, TypeError, AttributeError, OverflowError):
                    continue
    
    # As a last resort, check if there's a date in the title or description
    for field in ['title', 'description', 'summary']:
        if field in entry and entry[field] and isinstance(entry[field], str):
            # Look for common date patterns in the text
            try:
                # Use dateutil's parser with fuzzy matching
                return dateutil.parser.parse(entry[field], fuzzy=True)
            except (ValueError, TypeError):
                pass
    
    return None


def get_feed_entries(feed_url: str, lookback_days: int = 7, max_retries: int = 3, retry_delay: int = 2) -> List[Dict[str, Any]]:
    """
    Fetch and parse a feed, returning entries within lookback period.
    
    Args:
        feed_url: URL of the RSS feed
        lookback_days: Number of days to look back for articles
        max_retries: Maximum number of retry attempts for transient errors
        retry_delay: Delay between retries in seconds
        
    Returns:
        List of feed entries within the lookback period
    """
    for attempt in range(max_retries):
        try:
            # Add a user agent to avoid 403 errors
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            # Use requests to get the content first to handle redirects and HTTP errors better
            response = None
            try:
                response = requests.get(feed_url, headers=headers, timeout=30)
                response.raise_for_status()  # Raise exception for 4XX/5XX status codes
                feed_data = feedparser.parse(response.content)
            except requests.exceptions.RequestException as req_err:
                # If requests fails, try direct parsing as fallback
                logger.warning(f"Request failed for {feed_url}, trying direct parsing: {str(req_err)}")
                feed_data = feedparser.parse(feed_url)
            
            # Check if the feed has a bozo exception (parsing error)
            if hasattr(feed_data, 'bozo') and feed_data.bozo and hasattr(feed_data, 'bozo_exception'):
                # Some bozo exceptions are acceptable, others indicate real problems
                if not isinstance(feed_data.bozo_exception, feedparser.CharacterEncodingOverride):
                    logger.warning(f"Feed parsing warning for {feed_url}: {str(feed_data.bozo_exception)}")
            
            # Check HTTP status
            if hasattr(feed_data, 'status') and feed_data.status >= 400:
                error_msg = f"Error fetching feed {feed_url}: HTTP {feed_data.status}"
                if attempt < max_retries - 1:
                    logger.warning(f"{error_msg}, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(error_msg)
                    return []
            
            # Check if feed has entries
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
            if attempt < max_retries - 1:
                logger.warning(f"Error processing feed {feed_url}: {str(e)}, retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to process feed {feed_url} after {max_retries} attempts: {str(e)}")
                return []


def process_feed(
    feed: Feed, 
    db_session: Session, 
    lookback_days: int = 7,
    max_articles: int = 25,
    progress_callback: Optional[Callable[[str], None]] = None,
    max_errors: int = 5,  # Maximum number of consecutive errors before marking feed as inactive
    retry_attempts: int = 3  # Number of retry attempts for transient errors
) -> int:
    """
    Process a feed, fetch new articles and store them in the database.
    
    Args:
        feed: Feed object
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles: Maximum number of articles to process per feed
        progress_callback: Optional callback function for progress updates
        max_errors: Maximum number of consecutive errors before marking feed as inactive
        retry_attempts: Number of retry attempts for transient errors
        
    Returns:
        Number of new articles added
    """
    if feed.muted:
        if progress_callback:
            progress_callback(f"Skipping muted feed: {feed.title}")
        return 0
    
    if progress_callback:
        progress_callback(f"Processing feed: {feed.title}")
    
    # Always update the last checked timestamp
    feed.last_checked = datetime.utcnow()
    
    try:
        # Get feed entries with retry logic
        entries = get_feed_entries(feed.url, lookback_days, max_retries=retry_attempts)
        
        # If no entries were found, this might be a temporary issue or a feed that rarely updates
        if not entries:
            # Only increment error count for HTTP errors (which would be logged as errors)
            # Don't increment for feeds that simply have no new entries
            feed.no_entries_count = getattr(feed, 'no_entries_count', 0) + 1
            
            # If we consistently find no entries for a long time, log a warning
            if feed.no_entries_count > 5:  # After 5 consecutive checks with no entries
                logger.warning(f"Feed {feed.title} has had no entries for {feed.no_entries_count} consecutive checks")
            
            db_session.commit()
            return 0
        
        # Reset no entries counter if we found entries
        feed.no_entries_count = 0
        
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
        feed.error_count = getattr(feed, 'error_count', 0) + 1
        feed.last_error = str(e)
        
        # Check if we should mark the feed as inactive due to too many errors
        if feed.error_count >= max_errors:
            logger.warning(f"Feed {feed.title} has had {feed.error_count} consecutive errors. Marking as muted.")
            feed.muted = True
            feed.muted_reason = f"Auto-muted after {feed.error_count} consecutive errors. Last error: {feed.last_error}"
        
        db_session.commit()
        return 0


def process_all_feeds(
    db_session: Session, 
    lookback_days: int = 7,
    max_articles_per_feed: int = 25,
    max_errors: int = 5,
    retry_attempts: int = 3
) -> Dict[str, int]:
    """
    Process all active feeds in the database.
    
    Args:
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles_per_feed: Maximum number of articles to process per feed
        max_errors: Maximum number of consecutive errors before marking feed as inactive
        retry_attempts: Number of retry attempts for transient errors
        
    Returns:
        Dictionary with processing statistics
    """
    feeds = db_session.query(Feed).filter_by(muted=False).all()
    
    if not feeds:
        logger.warning("No active feeds found in database")
        return {"feeds_processed": 0, "new_articles": 0, "auto_muted": 0}
    
    total_new_articles = 0
    auto_muted_count = 0
    progress_bar = tqdm(feeds, desc="Processing feeds")
    
    for feed in progress_bar:
        # Check if feed was muted before processing
        was_muted_before = feed.muted
        
        progress_bar.set_description(f"Processing {feed.title[:30]}")
        new_articles = process_feed(
            feed,
            db_session, 
            lookback_days=lookback_days,
            max_articles=max_articles_per_feed,
            progress_callback=lambda msg: progress_bar.set_description(msg[:30]),
            max_errors=max_errors,
            retry_attempts=retry_attempts
        )
        total_new_articles += new_articles
        
        # Check if feed was auto-muted during processing
        if not was_muted_before and feed.muted:
            auto_muted_count += 1
    
    # Log summary of auto-muted feeds
    if auto_muted_count > 0:
        logger.warning(f"Auto-muted {auto_muted_count} feeds due to persistent errors")
    
    return {
        "feeds_processed": len(feeds),
        "new_articles": total_new_articles,
        "auto_muted": auto_muted_count
    }