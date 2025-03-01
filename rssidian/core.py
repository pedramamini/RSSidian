import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
import logging
import requests
from sqlalchemy.orm import Session
from tqdm import tqdm
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

from .config import Config
from .models import Article, Feed
from .rss import process_all_feeds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quality tier ranking
QUALITY_TIERS = {
    "S": 5,
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}


class RSSProcessor:
    """Main processor for RSS feeds and articles."""
    
    def __init__(self, config: Config, db_session: Session):
        """Initialize the processor with configuration and database session."""
        self.config = config
        self.db_session = db_session
        self._embedding_model = None
        self._vector_index = None
        
    @property
    def embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
        return self._embedding_model
    
    def ingest_feeds(self, lookback_days: Optional[int] = None, debug: bool = False) -> Dict[str, int]:
        """
        Ingest articles from all active feeds.
        
        Args:
            lookback_days: Number of days to look back for articles (default: config value)
            debug: Enable debug logging
            
        Returns:
            Dictionary with ingestion statistics
        """
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        lookback = lookback_days or self.config.default_lookback
        logger.info(f"Ingesting feeds with {lookback} day lookback period")
        
        return process_all_feeds(
            self.db_session,
            lookback_days=lookback,
            max_articles_per_feed=self.config.max_articles_per_feed,
            max_errors=5,  # Maximum number of consecutive errors before marking feed as inactive
            retry_attempts=3  # Number of retry attempts for transient errors
        )
    
    def _call_openrouter_api(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Call the OpenRouter API with a prompt.
        
        Args:
            prompt: Prompt to send to the API
            model: Model to use (default: config value)
            
        Returns:
            Response from the API or None if failed
        """
        api_key = self.config.openrouter_api_key
        if not api_key:
            logger.error("OpenRouter API key not configured")
            return None
        
        use_model = model or self.config.openrouter_model
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": use_model,
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
    
    def _generate_summary(self, article: Article) -> Optional[str]:
        """
        Generate a summary for an article.
        
        Args:
            article: Article to summarize
            
        Returns:
            Summary text or None if summarization failed
        """
        if not article.content:
            logger.warning(f"Article {article.id} has no content to summarize")
            return None
        
        prompt = self.config.openrouter_prompt.format(content=article.content)
        return self._call_openrouter_api(prompt)
    
    def _analyze_value(self, article: Article) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Analyze the value of an article.
        
        Args:
            article: Article to analyze
            
        Returns:
            Tuple of (quality_tier, quality_score, labels)
        """
        if not self.config.value_prompt_enabled or not article.content:
            return None, None, None
        
        prompt = self.config.value_prompt.format(content=article.content)
        response = self._call_openrouter_api(prompt, self.config.openrouter_processing_model)
        
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
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Truncate text if too long (most models have a max token limit)
        max_chars = 5000
        truncated_text = text[:max_chars] if len(text) > max_chars else text
        
        return self.embedding_model.encode(truncated_text).tolist()
    
    def _store_embedding(self, article_id: int, embedding: List[float]) -> None:
        """
        Store an embedding vector in the Annoy index.
        
        Args:
            article_id: ID of the article
            embedding: Embedding vector
        """
        if self._vector_index is None:
            self._load_or_create_index()
        
        if self._vector_index:
            self._vector_index.add_item(article_id, embedding)
            self._vector_index.build(self.config.annoy_n_trees)
            self._vector_index.save(self.config.annoy_index_path)
    
    def _load_or_create_index(self) -> None:
        """Load the existing Annoy index or create a new one."""
        embedding_dim = 768  # all-mpnet-base-v2 dimension
        
        self._vector_index = AnnoyIndex(embedding_dim, self.config.annoy_metric)
        
        # Try to load existing index
        if os.path.exists(self.config.annoy_index_path):
            try:
                self._vector_index.load(self.config.annoy_index_path)
                logger.info(f"Loaded existing vector index from {self.config.annoy_index_path}")
            except Exception as e:
                logger.error(f"Failed to load vector index: {str(e)}")
                # Create a new index
                self._vector_index = AnnoyIndex(embedding_dim, self.config.annoy_metric)
    
    def process_articles(self, batch_size: int = 10) -> Dict[str, int]:
        """
        Process unprocessed articles (summarize, analyze value, generate embeddings).
        
        Args:
            batch_size: Number of articles to process at once
            
        Returns:
            Dictionary with processing statistics
        """
        # Get unprocessed articles
        unprocessed = self.db_session.query(Article).filter_by(processed=False).all()
        
        if not unprocessed:
            logger.info("No unprocessed articles found")
            return {"articles_processed": 0, "with_summary": 0, "with_value": 0, "with_embedding": 0}
        
        # Initialize counters
        stats = {
            "articles_processed": 0,
            "with_summary": 0,
            "with_value": 0,
            "with_embedding": 0
        }
        
        # Process articles in batches
        for i in range(0, len(unprocessed), batch_size):
            batch = unprocessed[i:i+batch_size]
            progress_bar = tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{(len(unprocessed)-1)//batch_size + 1}")
            
            for article in progress_bar:
                progress_bar.set_description(f"Processing {article.title[:30]}")
                
                # Generate summary
                summary = self._generate_summary(article)
                if summary:
                    article.summary = summary
                    stats["with_summary"] += 1
                
                # Analyze value
                if self.config.value_prompt_enabled:
                    quality_tier, quality_score, labels = self._analyze_value(article)
                    article.quality_tier = quality_tier
                    article.quality_score = quality_score
                    article.labels = labels
                    if quality_tier:
                        stats["with_value"] += 1
                
                # Generate embedding from content and summary
                if article.content:
                    text_to_embed = article.content
                    if article.summary:
                        text_to_embed = article.summary + "\n\n" + text_to_embed
                    
                    embedding = self._generate_embedding(text_to_embed)
                    self._store_embedding(article.id, embedding)
                    article.embedding_generated = True
                    stats["with_embedding"] += 1
                
                # Mark as processed
                article.processed = True
                article.processed_at = datetime.utcnow()
                stats["articles_processed"] += 1
                
                # Save after each article in case of errors
                self.db_session.commit()
        
        return stats
    
    def search(self, query: str, relevance_threshold: float = 0.6, max_results: int = 20, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Search for articles using semantic search.
        
        Args:
            query: Search query
            relevance_threshold: Minimum relevance score (0-1)
            max_results: Maximum number of results to return
            refresh: Whether to refresh the index before searching
            
        Returns:
            List of search results
        """
        # Refresh the index if requested
        if refresh or self._vector_index is None:
            self._load_or_create_index()
            
            # If still none, there are no embeddings yet
            if self._vector_index is None:
                logger.warning("No vector index available")
                return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search for similar articles
        similar_ids, distances = self._vector_index.get_nns_by_vector(
            query_embedding, 
            max_results * 2,  # Get more than we need to filter by relevance
            include_distances=True
        )
        
        # Convert distances to similarity scores (Annoy uses distance, we want similarity)
        # For angular distance, similarity = 1 - (distance^2 / 2)
        similarities = [1 - (dist**2 / 2) for dist in distances]
        
        # Filter by relevance threshold
        results = []
        for article_id, similarity in zip(similar_ids, similarities):
            if similarity >= relevance_threshold:
                article = self.db_session.query(Article).filter_by(id=article_id).first()
                if article:
                    feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                    
                    # Generate excerpt around the most relevant part
                    excerpt = self._generate_excerpt(article.content, query, self.config.excerpt_length)
                    
                    results.append({
                        "id": article.id,
                        "title": article.title,
                        "feed": feed.title if feed else "Unknown Feed",
                        "url": article.url,
                        "published_at": article.published_at,
                        "summary": article.summary,
                        "excerpt": excerpt,
                        "quality_tier": article.quality_tier,
                        "quality_score": article.quality_score,
                        "relevance": similarity
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Limit to max_results
        return results[:max_results]
    
    def _generate_excerpt(self, text: str, query: str, max_length: int = 300) -> str:
        """
        Generate an excerpt of text around the most relevant part to the query.
        
        Args:
            text: Text to excerpt
            query: Search query
            max_length: Maximum length of excerpt
            
        Returns:
            Excerpt of text
        """
        if not text:
            return ""
        
        # Simple approach: split text into sentences and find the one with the most query terms
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Count query terms in each sentence
        query_terms = re.findall(r'\w+', query.lower())
        scores = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            scores.append(score)
        
        # Find the sentence with the highest score
        if not scores:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        best_idx = scores.index(max(scores))
        
        # Build excerpt around the best sentence
        excerpt = sentences[best_idx]
        
        # Add context before and after
        left_idx, right_idx = best_idx - 1, best_idx + 1
        remaining_length = max_length - len(excerpt)
        
        while remaining_length > 0 and (left_idx >= 0 or right_idx < len(sentences)):
            # Alternate adding sentences from before and after
            if left_idx >= 0:
                left_sentence = sentences[left_idx]
                if len(left_sentence) <= remaining_length:
                    excerpt = left_sentence + " " + excerpt
                    remaining_length -= len(left_sentence) + 1
                    left_idx -= 1
                else:
                    break
            
            if right_idx < len(sentences) and remaining_length > 0:
                right_sentence = sentences[right_idx]
                if len(right_sentence) <= remaining_length:
                    excerpt = excerpt + " " + right_sentence
                    remaining_length -= len(right_sentence) + 1
                    right_idx += 1
                else:
                    break
        
        # Add ellipsis if we didn't include all context
        if left_idx >= 0:
            excerpt = "..." + excerpt
        if right_idx < len(sentences):
            excerpt = excerpt + "..."
        
        return excerpt
    
    def generate_digest(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Generate a digest of high-value articles from the lookback period.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with digest content
        """
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get processed articles in date range
        articles = self.db_session.query(Article).filter(
            Article.processed == True,
            Article.published_at >= start_date,
            Article.published_at <= end_date
        ).all()
        
        if not articles:
            logger.warning(f"No processed articles found in date range")
            return {
                "date_range": f"Feed Overview from {start_date.strftime('%Y-%m-%d')} through {end_date.strftime('%Y-%m-%d')}",
                "summary_items": "No articles processed in this period.",
                "feed_stats": "No data available.",
                "articles": []
            }
        
        # Filter by quality tier
        min_tier = self.config.minimum_quality_tier
        min_tier_rank = QUALITY_TIERS.get(min_tier, 0)
        
        high_value_articles = []
        for article in articles:
            if not article.quality_tier:
                continue
            
            tier_rank = QUALITY_TIERS.get(article.quality_tier, 0)
            if tier_rank >= min_tier_rank:
                high_value_articles.append(article)
        
        # Group by topic using labels
        topics = defaultdict(list)
        for article in high_value_articles:
            if not article.labels:
                topics["Uncategorized"].append(article)
                continue
            
            # Split labels and use the first one as primary topic
            labels = [label.strip() for label in article.labels.split(",")]
            if labels:
                primary_label = labels[0]
                topics[primary_label].append(article)
            else:
                topics["Uncategorized"].append(article)
        
        # Sort topics by quality
        def get_topic_quality(topic_articles):
            return sum(QUALITY_TIERS.get(a.quality_tier, 0) for a in topic_articles)
        
        sorted_topics = sorted(topics.items(), key=lambda x: get_topic_quality(x[1]), reverse=True)
        
        # Build summary items
        summary_items = []
        for topic, topic_articles in sorted_topics:
            summary_items.append(f"## {topic}")
            
            # Sort articles by quality tier and then by date
            sorted_articles = sorted(
                topic_articles,
                key=lambda a: (QUALITY_TIERS.get(a.quality_tier, 0), a.published_at or datetime.min),
                reverse=True
            )
            
            for article in sorted_articles:
                feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                feed_name = feed.title if feed else "Unknown Feed"
                
                tier_display = f"{article.quality_tier}-Tier" if article.quality_tier else ""
                date_display = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown date"
                
                summary_items.append(f"### [{article.title}]({article.url})")
                summary_items.append(f"*{feed_name} | {date_display} | {tier_display}*")
                
                if article.summary:
                    summary_items.append(f"\n{article.summary}\n")
        
        # Build feed stats
        feed_counts = defaultdict(int)
        for article in articles:
            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
            if feed:
                feed_counts[feed.title] += 1
        
        sorted_feeds = sorted(feed_counts.items(), key=lambda x: x[1], reverse=True)
        
        feed_stats = [
            f"Total articles processed: {len(articles)}",
            f"High-value articles (>= {min_tier}-Tier): {len(high_value_articles)}",
            f"Number of topics: {len(topics)}",
            "\nArticles per feed:"
        ]
        
        for feed_name, count in sorted_feeds:
            feed_stats.append(f"- {feed_name}: {count}")
        
        digest = {
            "date_range": f"Feed Overview from {start_date.strftime('%Y-%m-%d')} through {end_date.strftime('%Y-%m-%d')}",
            "summary_items": "\n\n".join(summary_items),
            "feed_stats": "\n".join(feed_stats),
            "articles": high_value_articles
        }
        
        return digest