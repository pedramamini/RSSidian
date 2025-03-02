import feedparser
import requests
import time
import re
import json
import email.utils
import dateutil.parser
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from tqdm import tqdm
import logging

from .models import Feed, Article
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_openrouter_api(prompt: str, api_key: str, model: str) -> Optional[str]:
    """
    Call the OpenRouter API with a prompt.
    
    Args:
        prompt: Prompt to send to the API
        api_key: OpenRouter API key
        model: Model to use
        
    Returns:
        Response from the API or None if failed
    """
    if not api_key:
        logger.error("OpenRouter API key not configured")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            logger.error(f"Invalid API response: {response_data}")
            return None
    
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return None


def generate_summary(content: str, api_key: str, prompt_template: str, model: str) -> Optional[str]:
    """
    Generate a summary for an article.
    
    Args:
        content: Article content to summarize
        api_key: OpenRouter API key
        prompt_template: Prompt template for summarization
        model: Model to use
        
    Returns:
        Summary text or None if summarization failed
    """
    if not content:
        logger.warning("No content to summarize")
        return None
    
    prompt = prompt_template.format(content=content)
    return call_openrouter_api(prompt, api_key, model)


def analyze_value(content: str, api_key: str, prompt_template: str, model: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Analyze the value of an article.
    
    Args:
        content: Article content to analyze
        api_key: OpenRouter API key
        prompt_template: Prompt template for value analysis
        model: Model to use
        
    Returns:
        Tuple of (quality_tier, quality_score, labels)
    """
    if not content:
        return None, None, None
    
    prompt = prompt_template.format(content=content)
    response = call_openrouter_api(prompt, api_key, model)
    
    if not response:
        return None, None, None
    
    try:
        # Try to parse as JSON
        data = json.loads(response)
        
        # Extract quality tier from the rating field
        quality_tier = None
        rating = data.get("rating:")
        if rating and isinstance(rating, str):
            match = re.search(r"([SABCD]) Tier", rating)
            if match:
                quality_tier = match.group(1)
        
        # Extract quality score
        quality_score = data.get("quality-score")
        if isinstance(quality_score, str):
            try:
                quality_score = int(quality_score)
            except ValueError:
                quality_score = None
        
        # Extract labels
        labels = data.get("labels")
        
        return quality_tier, quality_score, labels
    
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse value analysis: {str(e)}")
        return None, None, None


def clean_html(content: str) -> str:
    """Clean HTML or markdown content for storage.
    
    Args:
        content: HTML or markdown content to clean
        
    Returns:
        Cleaned text content
    """
    # If content is empty, return empty string
    if not content:
        return ""
        
    # Check if content is likely markdown (from Jina.ai)
    if content.startswith('#') or '**' in content or '*' in content or '```' in content:
        # For markdown, we'll preserve it as is, but remove excessive whitespace
        clean_text = re.sub(r'\s{2,}', ' ', content).strip()
        return clean_text
    
    # For HTML content
    # Simple regex to remove HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', content)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def fetch_jina_content(url: str) -> Optional[str]:
    """Fetch content from a URL using Jina.ai's content extraction service.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Markdown content or None if fetching failed
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully fetched content from Jina.ai for URL: {url}")
            return response.text
        else:
            logger.warning(f"Failed to fetch content from Jina.ai for URL: {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching content from Jina.ai for URL: {url}. Error: {str(e)}")
        return None


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
    retry_attempts: int = 3,  # Number of retry attempts for transient errors
    config: Optional[Config] = None,  # Configuration for article analysis
    analyze_content: bool = True  # Whether to analyze content during ingestion
) -> Tuple[int, int, int, int]:
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
        config: Configuration for article analysis
        analyze_content: Whether to analyze content during ingestion
        
    Returns:
        Tuple of (new_articles, jina_enhanced, with_summary, with_value_analysis)
    """
    if feed.muted:
        if progress_callback:
            progress_callback(f"Skipping muted feed: {feed.title}")
        return 0, 0, 0, 0
    
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
            return 0, 0, 0, 0
        
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
        jina_enhanced_count = 0
        with_summary_count = 0
        with_value_count = 0
        
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
            
            # If content is empty but we have a URL in the GUID, try to fetch content from Jina.ai
            jina_enhanced = False
            if not content and guid.startswith('http'):
                logger.info(f"No content found for entry with GUID {guid}, attempting to fetch from Jina.ai")
                jina_content = fetch_jina_content(guid)
                if jina_content:
                    content = jina_content
                    jina_enhanced = True
                    jina_enhanced_count += 1
                    logger.info(f"Successfully fetched content from Jina.ai for {guid}")
            
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
                processed=False,  # Will be set to True if we analyze it now
                word_count=len(clean_content.split()),
                jina_enhanced=jina_enhanced  # Set flag if content was fetched from Jina.ai
            )
            
            # Double-check that feed_id is set correctly
            if not new_article.feed_id:
                logger.warning(f"feed_id not set for article {new_article.title}. Setting manually.")
                new_article.feed_id = feed.id
            
            # Analyze the article if requested and we have a configuration
            if analyze_content and config and clean_content:
                if progress_callback:
                    progress_callback(f"Analyzing article: {new_article.title[:30]}")
                
                # Generate summary
                if config.openrouter_api_key and config.openrouter_prompt:
                    summary = generate_summary(
                        clean_content,
                        config.openrouter_api_key,
                        config.openrouter_prompt,
                        config.openrouter_model
                    )
                    if summary:
                        new_article.summary = summary
                        with_summary_count += 1
                
                # Analyze value
                if config.openrouter_api_key and config.value_prompt_enabled and config.value_prompt:
                    quality_tier, quality_score, labels = analyze_value(
                        clean_content,
                        config.openrouter_api_key,
                        config.value_prompt,
                        config.openrouter_processing_model or config.openrouter_model
                    )
                    new_article.quality_tier = quality_tier
                    new_article.quality_score = quality_score
                    new_article.labels = labels
                    if quality_tier:
                        with_value_count += 1
                
                # Mark as processed
                new_article.processed = True
                new_article.processed_at = datetime.utcnow()
            
            db_session.add(new_article)
            new_article_count += 1
        
        # Update feed last updated timestamp if new articles were found
        if new_article_count > 0:
            feed.last_updated = datetime.utcnow()
        
        # Reset error count if successful
        feed.error_count = 0
        feed.last_error = None
        
        db_session.commit()
        return new_article_count, jina_enhanced_count, with_summary_count, with_value_count
    
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
        return 0, 0, 0, 0


def process_all_feeds(
    db_session: Session, 
    lookback_days: int = 7,
    max_articles_per_feed: int = 25,
    max_errors: int = 5,
    retry_attempts: int = 3,
    config: Optional[Config] = None,
    analyze_content: bool = True
) -> Dict[str, int]:
    """
    Process all active feeds in the database.
    
    Args:
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles_per_feed: Maximum number of articles to process per feed
        max_errors: Maximum number of consecutive errors before marking feed as inactive
        retry_attempts: Number of retry attempts for transient errors
        config: Configuration for article analysis
        analyze_content: Whether to analyze content during ingestion
        
    Returns:
        Dictionary with processing statistics
    """
    feeds = db_session.query(Feed).filter_by(muted=False).all()
    
    if not feeds:
        logger.warning("No active feeds found in database")
        return {
            "feeds_processed": 0, 
            "new_articles": 0, 
            "auto_muted": 0,
            "jina_enhanced": 0,
            "with_summary": 0,
            "with_value_analysis": 0
        }
    
    total_new_articles = 0
    auto_muted_count = 0
    total_jina_enhanced = 0
    total_with_summary = 0
    total_with_value = 0
    progress_bar = tqdm(feeds, desc="Processing feeds")
    
    for feed in progress_bar:
        # Check if feed was muted before processing
        was_muted_before = feed.muted
        
        progress_bar.set_description(f"Processing {feed.title[:30]}")
        new_articles, jina_enhanced, with_summary, with_value = process_feed(
            feed,
            db_session, 
            lookback_days=lookback_days,
            max_articles=max_articles_per_feed,
            progress_callback=lambda msg: progress_bar.set_description(msg[:30]),
            max_errors=max_errors,
            retry_attempts=retry_attempts,
            config=config,
            analyze_content=analyze_content
        )
        total_new_articles += new_articles
        total_jina_enhanced += jina_enhanced
        total_with_summary += with_summary
        total_with_value += with_value
        
        # Check if feed was auto-muted during processing
        if not was_muted_before and feed.muted:
            auto_muted_count += 1
    
    # Log summary of auto-muted feeds
    if auto_muted_count > 0:
        logger.warning(f"Auto-muted {auto_muted_count} feeds due to persistent errors")
    
    return {
        "feeds_processed": len(feeds),
        "new_articles": total_new_articles,
        "auto_muted": auto_muted_count,
        "jina_enhanced": total_jina_enhanced,
        "with_summary": total_with_summary,
        "with_value_analysis": total_with_value
    }