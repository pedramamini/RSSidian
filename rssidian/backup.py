import os
import shutil
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .config import Config, DEFAULT_DB_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    if not os.path.exists(file_path):
        return 0
    return os.path.getsize(file_path) / (1024 * 1024)


def get_backup_list(config: Config) -> List[Dict[str, Any]]:
    """
    Get a list of available backups.
    
    Args:
        config: Configuration
        
    Returns:
        List of backup information dictionaries
    """
    backup_dir = config.backup_dir
    
    if not os.path.exists(backup_dir):
        return []
    
    # Find all backup files
    backup_files = glob.glob(os.path.join(backup_dir, "*.db"))
    
    backups = []
    for backup_file in backup_files:
        try:
            # Extract date from filename
            filename = os.path.basename(backup_file)
            date_str = filename.replace(".db", "")
            
            # Handle multiple backups on same date with index
            if "-" in date_str and date_str.split("-")[-1].isdigit():
                base_date = "-".join(date_str.split("-")[:-1])
                index = int(date_str.split("-")[-1])
                display_date = f"{base_date} ({index})"
            else:
                display_date = date_str
            
            # Get file stats
            size_mb = get_file_size_mb(backup_file)
            
            backups.append({
                "date": display_date,
                "path": backup_file,
                "size_mb": size_mb,
                "filename": filename
            })
        except Exception as e:
            logger.warning(f"Error processing backup file {backup_file}: {str(e)}")
    
    # Sort by filename (which should be date-based)
    backups.sort(key=lambda x: x["filename"], reverse=True)
    
    return backups


def create_backup(config: Config) -> Optional[str]:
    """
    Create a backup of the database.
    
    Args:
        config: Configuration
        
    Returns:
        Path to the created backup or None if failed
    """
    db_path = config.db_path
    backup_dir = config.backup_dir
    
    if not os.path.exists(db_path):
        logger.error(f"Database file does not exist: {db_path}")
        return None
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup filename with today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check for existing backups with today's date
    existing_backups = glob.glob(os.path.join(backup_dir, f"{today}*.db"))
    
    if existing_backups:
        # Add index for multiple backups on same day
        index = len(existing_backups) + 1
        backup_filename = f"{today}-{index}.db"
    else:
        backup_filename = f"{today}.db"
    
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return None


def restore_backup(backup_date: str, config: Config, force: bool = False) -> bool:
    """
    Restore database from a backup.
    
    Args:
        backup_date: Date string to identify the backup
        config: Configuration
        force: Whether to force restore without confirmation
        
    Returns:
        True if restore was successful, False otherwise
    """
    backups = get_backup_list(config)
    
    # Find the backup by date
    target_backup = None
    for backup in backups:
        if backup_date in backup["date"]:
            target_backup = backup
            break
    
    if not target_backup:
        logger.error(f"No backup found for date: {backup_date}")
        return False
    
    backup_path = target_backup["path"]
    db_path = config.db_path
    
    if not os.path.exists(backup_path):
        logger.error(f"Backup file does not exist: {backup_path}")
        return False
    
    # Create a temporary backup of current database
    temp_backup = None
    if os.path.exists(db_path):
        temp_backup = f"{db_path}.temp_backup"
        try:
            shutil.copy2(db_path, temp_backup)
            logger.info(f"Created temporary backup at {temp_backup}")
        except Exception as e:
            logger.error(f"Failed to create temporary backup: {str(e)}")
            if not force:
                return False
    
    # Copy the backup to the database location
    try:
        shutil.copy2(backup_path, db_path)
        logger.info(f"Restored database from {backup_path}")
        
        # Remove temporary backup if restore was successful
        if temp_backup and os.path.exists(temp_backup):
            os.remove(temp_backup)
        
        return True
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        
        # Restore from temporary backup if available
        if temp_backup and os.path.exists(temp_backup):
            try:
                shutil.copy2(temp_backup, db_path)
                logger.info(f"Restored from temporary backup")
            except Exception as e2:
                logger.error(f"Failed to restore from temporary backup: {str(e2)}")
        
        return False