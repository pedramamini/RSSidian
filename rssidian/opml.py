import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from .models import Feed


def parse_opml(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse an OPML file and extract feed information.
    
    Args:
        file_path: Path to the OPML file
        
    Returns:
        List of dictionaries containing feed information
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OPML file not found: {file_path}")
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        feeds = []
        
        # Find all outline elements that have an xmlUrl attribute
        for outline in root.findall(".//outline[@xmlUrl]"):
            feed = {
                "title": outline.get("title") or outline.get("text", "Unknown Feed"),
                "url": outline.get("xmlUrl"),
                "description": outline.get("description", ""),
                "website_url": outline.get("htmlUrl", "")
            }
            feeds.append(feed)
        
        return feeds
    
    except ET.ParseError as e:
        raise ValueError(f"Invalid OPML file: {e}")


def import_feeds_from_opml(file_path: str, db_session: Session) -> Tuple[int, int]:
    """
    Import feeds from an OPML file into the database.
    
    Args:
        file_path: Path to the OPML file
        db_session: SQLAlchemy database session
        
    Returns:
        Tuple of (new_feeds_count, updated_feeds_count)
    """
    feeds_data = parse_opml(file_path)
    
    new_count = 0
    updated_count = 0
    
    for feed_data in feeds_data:
        # Check if feed already exists
        existing_feed = db_session.query(Feed).filter_by(url=feed_data["url"]).first()
        
        if existing_feed:
            # Update existing feed
            existing_feed.title = feed_data["title"]
            existing_feed.description = feed_data["description"]
            existing_feed.website_url = feed_data["website_url"]
            updated_count += 1
        else:
            # Create new feed
            new_feed = Feed(
                title=feed_data["title"],
                url=feed_data["url"],
                description=feed_data["description"],
                website_url=feed_data["website_url"],
                last_updated=datetime.utcnow(),
                last_checked=datetime.utcnow()
            )
            db_session.add(new_feed)
            new_count += 1
    
    db_session.commit()
    return new_count, updated_count