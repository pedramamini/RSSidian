import os
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for macOS and other filesystems.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace characters that are problematic in filenames
    unsafe_chars = r'[<>:"/\\|?*]'
    filename = re.sub(unsafe_chars, '_', filename)
    
    # Limit filename length
    max_length = 200
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename


def write_digest_to_obsidian(digest: Dict[str, Any], config: Config) -> Optional[str]:
    """
    Write a digest to Obsidian as markdown.
    
    Args:
        digest: Digest content
        config: Configuration
        
    Returns:
        Path to the created markdown file or None if failed
    """
    # Get Obsidian vault path
    vault_path = config.obsidian_vault_path
    
    if not os.path.exists(vault_path):
        logger.error(f"Obsidian vault path does not exist: {vault_path}")
        return None
    
    # Create directory for digests if it doesn't exist
    digest_dir = os.path.join(vault_path, "RSS Digests")
    os.makedirs(digest_dir, exist_ok=True)
    
    # Create filename from date range
    date_range = digest["date_range"]
    filename = sanitize_filename(date_range) + ".md"
    file_path = os.path.join(digest_dir, filename)
    
    # Apply template
    template = config.obsidian_template
    content = template.format(
        date_range=digest["date_range"],
        summary_items=digest["summary_items"],
        feed_stats=digest["feed_stats"]
    )
    
    # Write to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Wrote digest to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to write digest to {file_path}: {str(e)}")
        return None