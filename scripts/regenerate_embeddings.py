#!/usr/bin/env python
"""
Script to regenerate embeddings for all articles in the RSSidian database.
This script rebuilds the vector index from scratch for all articles.
"""

import os
import sys
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rssidian.config import Config
from rssidian.models import get_db_session, Article
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_embedding(model, text: str) -> List[float]:
    """Generate an embedding vector for text."""
    # Truncate text if too long (most models have a max token limit)
    max_chars = 5000
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    return model.encode(truncated_text).tolist()

def main():
    """Main function to regenerate embeddings."""
    # Load configuration
    config = Config()
    db_session = get_db_session(config.db_path)
    
    try:
        # Count articles
        total_articles = db_session.query(Article).count()
        logger.info(f"Found {total_articles} articles in the database")
        
        # Delete existing Annoy index file if it exists
        index_path = os.path.expanduser(config.annoy_index_path)
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"Deleted existing Annoy index file: {index_path}")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Create a new Annoy index
        embedding_dim = 768  # all-mpnet-base-v2 dimension
        vector_index = AnnoyIndex(embedding_dim, config.annoy_metric)
        
        # Get all articles
        articles = db_session.query(Article).all()
        
        # Track statistics
        stats = {
            "articles_processed": 0,
            "with_embedding": 0
        }
        
        # Process articles with progress bar
        logger.info(f"Processing {len(articles)} articles...")
        for article in tqdm(articles, desc="Generating embeddings"):
            if article.content:
                # Generate text to embed
                text_to_embed = article.content
                if article.summary:
                    text_to_embed = article.summary + "\n\n" + text_to_embed
                
                # Generate embedding
                embedding = generate_embedding(embedding_model, text_to_embed)
                
                # Add to index
                vector_index.add_item(article.id, embedding)
                
                # Update article
                article.embedding_generated = True
                stats["with_embedding"] += 1
            
            stats["articles_processed"] += 1
        
        # Build and save the index
        logger.info(f"Building vector index with {config.annoy_n_trees} trees...")
        vector_index.build(config.annoy_n_trees)
        vector_index.save(index_path)
        logger.info(f"Saved vector index to {index_path}")
        
        # Commit changes to database
        db_session.commit()
        
        # Log results
        logger.info(f"Processed {stats['articles_processed']} articles:")
        logger.info(f"  With embeddings: {stats['with_embedding']}")
        
    except Exception as e:
        logger.error(f"Error during embedding regeneration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db_session.close()

if __name__ == "__main__":
    main()
