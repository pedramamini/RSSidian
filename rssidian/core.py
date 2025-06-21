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
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn

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

        try:
            return process_all_feeds(
                self.db_session,
                lookback_days=lookback,
                max_articles_per_feed=self.config.max_articles_per_feed,
                max_errors=5,  # Maximum number of consecutive errors before marking feed as inactive
                retry_attempts=3,  # Number of retry attempts for transient errors
                config=self.config,  # Pass the config for article analysis
                analyze_content=self.config.analyze_during_ingestion  # Whether to analyze during ingestion
            )
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            # Return empty stats in case of complete failure
            return {
                "feeds_processed": 0,
                "new_articles": 0,
                "auto_muted": 0,
                "jina_enhanced": 0,
                "with_summary": 0,
                "with_value_analysis": 0
            }

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

        # Import cost tracker here to avoid circular imports
        from .cost_tracker import track_api_call

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

            # Track the cost of this API call if enabled
            if self.config.cost_tracking_enabled:
                track_api_call(response_data, use_model)

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

    def find_similar_articles(self, article: Article, similarity_threshold: float = None) -> List[Article]:
        """
        Find similar articles to the given article using embeddings.

        Args:
            article: Article to find similar articles for
            similarity_threshold: Minimum similarity score (default: config value)

        Returns:
            List of similar articles
        """
        if not article.content:
            return []

        threshold = similarity_threshold or self.config.similarity_threshold

        # Generate embedding for the article
        text_to_embed = article.content
        if article.summary:
            text_to_embed = article.summary + "\n\n" + text_to_embed

        query_embedding = self._generate_embedding(text_to_embed)

        # Load index if needed
        if self._vector_index is None:
            self._load_or_create_index()

        if self._vector_index is None:
            return []

        try:
            # Search for similar articles
            similar_ids, distances = self._vector_index.get_nns_by_vector(
                query_embedding,
                20,  # Get up to 20 candidates
                include_distances=True
            )

            # Convert distances to similarity scores
            similarities = [1 - (dist**2 / 2) for dist in distances]

            # Find articles that meet the similarity threshold
            similar_articles = []
            for article_id, similarity in zip(similar_ids, similarities):
                if similarity >= threshold and article_id != article.id:
                    similar_article = self.db_session.query(Article).filter_by(id=article_id).first()
                    if similar_article and not similar_article.processed:
                        similar_articles.append(similar_article)

            return similar_articles

        except Exception as e:
            logger.warning(f"Error finding similar articles for {article.title}: {str(e)}")
            return []

    def _group_similar_articles(self, articles: List[Article]) -> List[List[Article]]:
        """
        Group articles by similarity.

        Args:
            articles: List of articles to group

        Returns:
            List of article groups (each group is a list of similar articles)
        """
        if not articles:
            return []

        groups = []
        processed_ids = set()

        for article in articles:
            if article.id in processed_ids:
                continue

            # Start a new group with this article
            group = [article]
            processed_ids.add(article.id)

            # Find similar articles
            similar_articles = self.find_similar_articles(article)
            for similar_article in similar_articles:
                if similar_article.id not in processed_ids:
                    group.append(similar_article)
                    processed_ids.add(similar_article.id)

            groups.append(group)

        logger.info(f"Grouped {len(articles)} articles into {len(groups)} groups")
        for i, group in enumerate(groups):
            if len(group) > 1:
                logger.info(f"Group {i+1}: {len(group)} similar articles")
                for article in group:
                    logger.info(f"  - {article.title[:60]}")

        return groups

    def _generate_grouped_summary(self, articles: List[Article]) -> Optional[str]:
        """
        Generate a summary for a group of similar articles.

        Args:
            articles: List of similar articles to summarize together

        Returns:
            Combined summary text or None if generation failed
        """
        if not articles:
            return None

        if len(articles) == 1:
            # Single article, use regular summary
            return self._generate_summary(articles[0])

        # Multiple similar articles - create combined summary
        logger.info(f"Generating grouped summary for {len(articles)} similar articles")

        # Prepare content from all articles
        combined_content = []
        sources = []
        
        for i, article in enumerate(articles, 1):
            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
            feed_name = feed.title if feed else "Unknown Feed"
            
            # Add source information
            sources.append(f"Source {i}: {article.title} ({feed_name}) - {article.url}")
            
            # Add content with source label
            if article.content:
                combined_content.append(f"--- Source {i} Content ---\n{article.content}")

        if not combined_content:
            logger.warning("No content found in article group")
            return None

        # Create prompt for grouped summarization
        all_content = "\n\n".join(combined_content)
        all_sources = "\n".join(sources)

        grouped_prompt = f"""You are summarizing multiple articles about the same topic or event. 

Create a comprehensive summary that:
1. Identifies the main topic/event being covered
2. Combines information from all sources into a coherent narrative
3. Notes any different perspectives or additional details from different sources
4. Provides key insights and takeaways

Sources:
{all_sources}

Combined Content:
{all_content}

Provide a comprehensive summary that synthesizes information from all sources."""

        return self._call_openrouter_api(grouped_prompt)

    def process_articles(self, batch_size: int = 10) -> Dict[str, int]:
        """
        Process unprocessed articles (summarize, analyze value, generate embeddings).
        Groups similar articles and creates combined summaries.

        Args:
            batch_size: Number of articles to process at once

        Returns:
            Dictionary with processing statistics
        """
        # Get unprocessed articles
        unprocessed = self.db_session.query(Article).filter_by(processed=False).all()

        if not unprocessed:
            logger.info("No unprocessed articles found")
            return {"articles_processed": 0, "with_summary": 0, "with_value": 0, "with_embedding": 0, "groups_processed": 0}

        # Initialize counters
        stats = {
            "articles_processed": 0,
            "with_summary": 0,
            "with_value": 0,
            "with_embedding": 0,
            "groups_processed": 0
        }

        # Track feed statistics updates
        feed_stats = defaultdict(lambda: {
            "new_articles": 0,
            "quality_tiers": defaultdict(int),
            "quality_scores": []
        })

        # Process article groups using Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            # Step 1: Generate embeddings for all articles first (needed for similarity detection)
            logger.info(f"Generating embeddings for {len(unprocessed)} articles...")
            embedding_task = progress.add_task(f"[bold]Generating embeddings", total=len(unprocessed))
            
            for article in unprocessed:
                try:
                    if article.content:
                        embedding = self._generate_embedding(article.content)
                        self._store_embedding(article.id, embedding)
                        article.embedding_generated = True
                        stats["with_embedding"] += 1
                except Exception as e:
                    logger.error(f"Error generating embedding for article {article.id}: {str(e)}")
                progress.update(embedding_task, advance=1)
            
            # Step 2: Group similar articles after embeddings are generated
            logger.info(f"Grouping {len(unprocessed)} articles by similarity...")
            article_groups = self._group_similar_articles(unprocessed)
            stats["groups_processed"] = len(article_groups)

            # Step 3: Process article groups for summaries and value analysis
            summary_task = progress.add_task(f"[bold]Processing {len(article_groups)} groups", total=len(unprocessed))

            for group_idx, group in enumerate(article_groups):
                try:
                    # Update progress description
                    if len(group) == 1:
                        progress.update(summary_task, description=f"[bold blue]Processing: [cyan]{group[0].title[:40]}")
                    else:
                        progress.update(summary_task, description=f"[bold blue]Processing group of {len(group)} similar articles")

                    # Generate grouped summary
                    grouped_summary = self._generate_grouped_summary(group)
                    
                    # Process each article in the group
                    for article in group:
                        try:
                            # Set the grouped summary for all articles in the group
                            if grouped_summary:
                                article.summary = grouped_summary
                                stats["with_summary"] += 1

                            # Analyze value (done individually for each article)
                            if self.config.value_prompt_enabled:
                                quality_tier, quality_score, labels = self._analyze_value(article)
                                article.quality_tier = quality_tier
                                article.quality_score = quality_score
                                article.labels = labels
                                if quality_tier:
                                    stats["with_value"] += 1

                            # Mark as processed
                            article.processed = True
                            article.processed_at = datetime.utcnow()
                            stats["articles_processed"] += 1

                            # Track statistics for this feed
                            feed_id = article.feed_id
                            feed_stats[feed_id]["new_articles"] += 1
                            if article.quality_tier:
                                feed_stats[feed_id]["quality_tiers"][article.quality_tier] += 1
                            if article.quality_score:
                                feed_stats[feed_id]["quality_scores"].append(article.quality_score)

                            # Update progress
                            progress.update(summary_task, advance=1)
                        except Exception as e:
                            logger.error(f"Error processing article {article.id} in group {group_idx}: {str(e)}")
                            # Still mark as processed to avoid reprocessing
                            article.processed = True
                            article.processed_at = datetime.utcnow()
                            stats["articles_processed"] += 1
                            progress.update(summary_task, advance=1)

                    # Save after each group in case of errors
                    self.db_session.commit()
                except Exception as e:
                    logger.error(f"Error processing group {group_idx}: {str(e)}")
                    # Mark all articles in the group as processed to avoid reprocessing
                    for article in group:
                        if not article.processed:
                            article.processed = True
                            article.processed_at = datetime.utcnow()
                            stats["articles_processed"] += 1
                            progress.update(summary_task, advance=1)
                    self.db_session.commit()

        # Update feed statistics
        self._update_feed_statistics(feed_stats)

        return stats
    
    except Exception as e:
        logger.error(f"Error during article processing: {str(e)}", exc_info=True)
        # Return partial stats if any processing was done
        return stats

    def _update_feed_statistics(self, feed_stats: Dict[int, Dict[str, Any]]):
        """
        Update feed statistics with processed article data.

        Args:
            feed_stats: Dictionary of feed statistics keyed by feed_id
        """
        for feed_id, stats in feed_stats.items():
            feed = self.db_session.query(Feed).filter_by(id=feed_id).first()
            if not feed:
                continue

            # Update article count
            feed.article_count = (feed.article_count or 0) + stats["new_articles"]

            # Update quality tier counts
            current_tier_counts = {}
            if feed.quality_tier_counts:
                try:
                    current_tier_counts = json.loads(feed.quality_tier_counts)
                except json.JSONDecodeError:
                    current_tier_counts = {}

            # Merge current and new tier counts
            for tier, count in stats["quality_tiers"].items():
                current_tier_counts[tier] = current_tier_counts.get(tier, 0) + count
            feed.quality_tier_counts = json.dumps(current_tier_counts)

            # Update average quality score
            if stats["quality_scores"]:
                current_total = (feed.avg_quality_score or 0) * (feed.article_count - stats["new_articles"] or 0)
                new_total = current_total + sum(stats["quality_scores"])
                feed.avg_quality_score = new_total / feed.article_count if feed.article_count > 0 else None

        # Commit changes
        self.db_session.commit()

    def search(self, query: str, relevance_threshold: float = 0.3, max_results: int = 20, refresh: bool = False) -> List[Dict[str, Any]]:
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

    def _generate_aggregated_summary(self, articles: List[Article], filename: str = None) -> Optional[str]:
        """
        Generate an aggregated summary of articles categorized by subject matter.

        Args:
            articles: List of articles to summarize
            filename: Name of the digest file (without extension)

        Returns:
            Aggregated summary text or None if generation failed
        """
        if not articles:
            logger.warning("No articles to generate aggregated summary")
            return None

        # Prepare article summaries for the prompt
        article_summaries = []
        for article in articles:
            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
            feed_name = feed.title if feed else "Unknown Feed"

            # Format: Title (Feed) - Summary with internal link
            # Store the article title for later reference in Obsidian internal link format
            article_title = article.title

            # Use Obsidian internal link format for the LLM to reference
            summary_text = f"## **{article.title}** ({feed_name})\n"

            # Add URL as plain text
            summary_text += f"Source: {article.url}\n"

            # Add quality tier if available
            if article.quality_tier:
                summary_text += f"Quality: {article.quality_tier}-Tier\n"

            # Add labels if available
            if article.labels:
                summary_text += f"Labels: {article.labels}\n"

            # Add the article summary
            if article.summary:
                summary_text += f"\n{article.summary}\n"

            article_summaries.append(summary_text)

        # Join all summaries with separators
        all_summaries = "\n---\n".join(article_summaries)

        # Create the prompt with the summaries and filename
        prompt = self.config.aggregator_prompt.format(
            summaries=all_summaries,
            filename=filename or "RSS Digest"
        )

        # Call the API to generate the aggregated summary
        llm_output = self._call_openrouter_api(prompt, self.config.openrouter_processing_model)

        if llm_output and filename:
            # Process the output to replace external links with Obsidian internal links
            return self._process_aggregated_summary(llm_output, filename, articles)

        return llm_output

    def _process_aggregated_summary(self, summary: str, filename: str, articles: List[Article]) -> str:
        """
        Process the aggregated summary to replace external links with Obsidian internal links.

        Args:
            summary: The aggregated summary text
            filename: The filename of the digest
            articles: List of articles

        Returns:
            Processed summary text
        """
        # Create a mapping of article titles to use for replacements
        article_titles = {article.title: article for article in articles}

        # Regular expression to find markdown links: [Title](URL)
        link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

        def replace_link(match):
            title = match.group(1)
            url = match.group(2)

            # Clean up the title (remove ** or other markdown formatting)
            clean_title = re.sub(r'[\*_`]', '', title)

            # Check if this title exists in our articles
            for article_title, article in article_titles.items():
                # Check if the clean title is similar to an article title
                if clean_title in article_title or article_title in clean_title:
                    # If the URL is an external link, replace with Obsidian internal link
                    if url.startswith('http'):
                        return f'[[{filename}#{article_title}|{title}]]'

            # If no match or not an external link, return the original
            return match.group(0)

        # Replace links in the summary
        processed_summary = re.sub(link_pattern, replace_link, summary)

        return processed_summary

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
            from_date_str = start_date.strftime('%Y-%m-%d')
            to_date_str = end_date.strftime('%Y-%m-%d')
            return {
                "date_range": f"Feed Overview from {from_date_str} through {to_date_str}",
                "from_date": from_date_str,
                "to_date": to_date_str,
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

        # Format dates for the digest
        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = end_date.strftime('%Y-%m-%d')

        # Create filename for the digest
        filename_template = self.config.obsidian_filename_template
        formatted_filename = filename_template.format(
            date_range=f"Feed Overview from {from_date_str} through {to_date_str}",
            from_date=from_date_str,
            to_date=to_date_str,
            date=datetime.now().strftime("%Y-%m-%d"),
            datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        # Remove .md extension if present
        if formatted_filename.endswith(".md"):
            formatted_filename = formatted_filename[:-3]

        # Generate aggregated summary with filename
        aggregated_summary = self._generate_aggregated_summary(high_value_articles, formatted_filename)

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

            # Group articles by summary to detect and display similar articles together
            summary_groups = defaultdict(list)
            for article in sorted_articles:
                summary_key = article.summary if article.summary else f"no_summary_{article.id}"
                summary_groups[summary_key].append(article)

            for summary_key, articles_in_group in summary_groups.items():
                if len(articles_in_group) == 1:
                    # Single article - display normally
                    article = articles_in_group[0]
                    feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                    feed_name = feed.title if feed else "Unknown Feed"

                    tier_display = f"{article.quality_tier}-Tier" if article.quality_tier else ""
                    date_display = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown date"

                    summary_items.append(f"### {article.title}")
                    summary_items.append(f"*{feed_name} | {date_display} | {tier_display}*")
                    summary_items.append(f"\n{article.url}\n")

                    if article.summary:
                        summary_items.append(f"{article.summary}\n")
                else:
                    # Multiple articles with same summary - display as grouped
                    logger.info(f"Displaying {len(articles_in_group)} similar articles as a group")
                    
                    # Create a title that represents the group
                    first_article = articles_in_group[0]
                    summary_items.append(f"### {first_article.title}")
                    
                    # Show all sources
                    source_lines = []
                    for article in articles_in_group:
                        feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                        feed_name = feed.title if feed else "Unknown Feed"
                        tier_display = f"{article.quality_tier}-Tier" if article.quality_tier else ""
                        date_display = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown date"
                        source_lines.append(f"{feed_name} | {date_display} | {tier_display}")
                        source_lines.append(article.url)
                    
                    summary_items.append("*" + "\n".join(source_lines) + "*")
                    summary_items.append("")

                    if first_article.summary:
                        summary_items.append(f"{first_article.summary}\n")

        # Build feed stats
        feed_stats_dict = defaultdict(lambda: {"total": 0, "accepted": 0})
        for article in articles:
            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
            if feed:
                feed_stats_dict[feed.title]["total"] += 1
                if article.quality_tier and QUALITY_TIERS.get(article.quality_tier, 0) >= min_tier_rank:
                    feed_stats_dict[feed.title]["accepted"] += 1

        sorted_feeds = sorted(feed_stats_dict.items(), key=lambda x: x[1]["total"], reverse=True)

        feed_stats = [
            f"Total articles processed: {len(articles)}",
            f"High-value articles (>= {min_tier}-Tier): {len(high_value_articles)}",
            f"Number of topics: {len(topics)}",
            "\nArticles per feed:"
        ]

        for feed_name, stats in sorted_feeds:
            acceptance_rate = (stats["accepted"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            feed_stats.append(f"- {feed_name}: {stats['total']}, {stats['accepted']} accepted, {acceptance_rate:.0f}%")

        digest = {
            "date_range": f"Feed Overview from {from_date_str} through {to_date_str}",
            "from_date": from_date_str,
            "to_date": to_date_str,
            "summary_items": "\n\n".join(summary_items),
            "feed_stats": "\n".join(feed_stats),
            "articles": high_value_articles,
            "aggregated_summary": aggregated_summary,
            "filename": formatted_filename
        }

        return digest