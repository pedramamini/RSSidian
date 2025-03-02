import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
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
    error_count = 0
    
    for feed_data in feeds_data:
        try:
            # Check if feed already exists - use get() to avoid potential session issues
            existing_feed = db_session.query(Feed).filter_by(url=feed_data["url"]).one_or_none()
            
            if existing_feed:
                # Update existing feed
                existing_feed.title = feed_data["title"]
                existing_feed.description = feed_data["description"]
                existing_feed.website_url = feed_data["website_url"]
                updated_count += 1
                # Commit each update individually to avoid transaction conflicts
                db_session.commit()
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
                # Commit each new feed individually to avoid transaction conflicts
                db_session.commit()
                new_count += 1
        except Exception as e:
            # Roll back the transaction if there's an error
            db_session.rollback()
            error_count += 1
            print(f"Error importing feed {feed_data['title']}: {e}")
            # Continue with the next feed
    
    return new_count, updated_count


def export_feeds_to_opml(db_session: Session, include_muted: bool = True) -> str:
    """
    Export feeds from the database to OPML format.
    
    Args:
        db_session: SQLAlchemy database session
        include_muted: Whether to include muted feeds in the export
        
    Returns:
        OPML content as a string
    """
    # Create the OPML structure
    root = ET.Element("opml")
    root.set("version", "2.0")
    
    # Add the head element
    head = ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    title.text = "RSSidian Feed Export"
    date_created = ET.SubElement(head, "dateCreated")
    date_created.text = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S %z")
    
    # Add the body element
    body = ET.SubElement(root, "body")
    
    # Query feeds from the database
    query = db_session.query(Feed)
    if not include_muted:
        query = query.filter_by(muted=False)
    
    feeds = query.order_by(Feed.title).all()
    
    # Add feeds to the OPML file
    for feed in feeds:
        outline = ET.SubElement(body, "outline")
        outline.set("text", feed.title)
        outline.set("title", feed.title)
        outline.set("type", "rss")
        outline.set("xmlUrl", feed.url)
        
        if feed.website_url:
            outline.set("htmlUrl", feed.website_url)
        
        if feed.description:
            outline.set("description", feed.description)
        
        # Add custom attributes for RSSidian-specific features
        if feed.muted:
            outline.set("muted", "true")
            if feed.muted_reason:
                outline.set("mutedReason", feed.muted_reason)
        
        if feed.peer_through:
            outline.set("peerThrough", "true")
    
    # Format the XML with proper indentation
    rough_string = ET.tostring(root, encoding="unicode", method="xml")
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")