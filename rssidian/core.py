import os
import json
import time
import secrets
import hashlib
import base64
import urllib.parse
import webbrowser
import http.server
import socketserver
import threading
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


class AnthropicOAuthClient:
    """OAuth client for Anthropic API."""
    
    def __init__(self, config: Config):
        """Initialize the OAuth client with configuration."""
        self.config = config
        self.client_id = config.anthropic_client_id
        self.client_secret = config.anthropic_client_secret
        self.redirect_uri = config.anthropic_redirect_uri
        self.token_file = config.anthropic_token_file
        self.auth_url = "https://console.anthropic.com/oauth/authorize"
        self.token_url = "https://api.anthropic.com/oauth/token"
        self.api_base = "https://api.anthropic.com"
        self._access_token = None
        self._refresh_token = None
        self._load_tokens()
    
    def _generate_pkce_challenge(self) -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        return code_verifier, code_challenge
    
    def _load_tokens(self):
        """Load tokens from file or config."""
        # First try config/environment
        if self.config.anthropic_access_token:
            self._access_token = self.config.anthropic_access_token
            self._refresh_token = self.config.anthropic_refresh_token
            return
        
        # Then try token file
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    tokens = json.load(f)
                    self._access_token = tokens.get('access_token')
                    self._refresh_token = tokens.get('refresh_token')
                    logger.info("Loaded Anthropic tokens from file")
            except Exception as e:
                logger.error(f"Failed to load tokens from file: {str(e)}")
    
    def _save_tokens(self, access_token: str, refresh_token: str):
        """Save tokens to file."""
        self._access_token = access_token
        self._refresh_token = refresh_token
        
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'updated_at': datetime.utcnow().isoformat()
                }, f, indent=2)
            logger.info("Saved Anthropic tokens to file")
        except Exception as e:
            logger.error(f"Failed to save tokens to file: {str(e)}")
    
    def get_authorization_url(self) -> Tuple[str, str]:
        """Get OAuth authorization URL and code verifier."""
        if not self.client_id:
            raise ValueError("Anthropic client_id not configured")
        
        code_verifier, code_challenge = self._generate_pkce_challenge()
        state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'read write',
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(params)}"
        return auth_url, code_verifier
    
    def exchange_code_for_tokens(self, authorization_code: str, code_verifier: str) -> bool:
        """Exchange authorization code for access tokens."""
        if not self.client_id or not self.client_secret:
            raise ValueError("Anthropic client_id and client_secret must be configured")
        
        data = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'code_verifier': code_verifier
        }
        
        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data['access_token']
            refresh_token = token_data.get('refresh_token')
            
            self._save_tokens(access_token, refresh_token)
            return True
            
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {str(e)}")
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self._refresh_token:
            logger.error("No refresh token available")
            return False
        
        data = {
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self._refresh_token
        }
        
        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data['access_token']
            refresh_token = token_data.get('refresh_token', self._refresh_token)
            
            self._save_tokens(access_token, refresh_token)
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {str(e)}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if we have a valid access token."""
        return bool(self._access_token)
    
    def start_oauth_flow(self) -> bool:
        """Start the OAuth flow in a browser."""
        if not self.client_id or not self.client_secret:
            logger.error("Anthropic OAuth client_id and client_secret must be configured")
            return False
        
        auth_url, code_verifier = self.get_authorization_url()
        
        # Start a local server to handle the callback
        callback_received = threading.Event()
        authorization_code = None
        
        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal authorization_code
                
                if self.path.startswith('/callback'):
                    query = urllib.parse.urlparse(self.path).query
                    params = urllib.parse.parse_qs(query)
                    
                    if 'code' in params:
                        authorization_code = params['code'][0]
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b'''
                        <html><body>
                        <h1>Authorization successful!</h1>
                        <p>You can close this window and return to the CLI.</p>
                        </body></html>
                        ''')
                        callback_received.set()
                    else:
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b'<html><body><h1>Authorization failed!</h1></body></html>')
                        callback_received.set()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        # Extract port from redirect URI
        parsed_uri = urllib.parse.urlparse(self.redirect_uri)
        port = parsed_uri.port or 8080
        
        # Start the callback server
        with socketserver.TCPServer(("", port), CallbackHandler) as httpd:
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            
            logger.info(f"Starting OAuth flow. Opening browser to: {auth_url}")
            webbrowser.open(auth_url)
            
            # Wait for callback
            if callback_received.wait(timeout=120):  # 2 minutes timeout
                httpd.shutdown()
                
                if authorization_code:
                    logger.info("Authorization code received, exchanging for tokens...")
                    return self.exchange_code_for_tokens(authorization_code, code_verifier)
                else:
                    logger.error("Authorization failed")
                    return False
            else:
                httpd.shutdown()
                logger.error("OAuth flow timed out")
                return False
    
    def make_api_request(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Make an API request to Anthropic."""
        if not self._access_token:
            logger.error("No access token available. Please authenticate first.")
            return None
        
        use_model = model or self.config.anthropic_model
        
        headers = {
            'Authorization': f'Bearer {self._access_token}',
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': use_model,
            'max_tokens': 4000,
            'system': self.config.anthropic_system_prompt,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/messages",
                headers=headers,
                json=data
            )
            
            if response.status_code == 401:
                # Token might be expired, try to refresh
                logger.info("Access token expired, attempting to refresh...")
                if self.refresh_access_token():
                    # Update the headers with new token
                    headers['Authorization'] = f'Bearer {self._access_token}'
                    response = requests.post(
                        f"{self.api_base}/v1/messages",
                        headers=headers,
                        json=data
                    )
                else:
                    logger.error("Failed to refresh access token")
                    return None
            
            response.raise_for_status()
            response_data = response.json()
            
            # Track the cost if enabled
            if self.config.cost_tracking_enabled:
                from .cost_tracker import track_anthropic_api_call
                track_anthropic_api_call(response_data, use_model)
            
            if 'content' in response_data and len(response_data['content']) > 0:
                return response_data['content'][0]['text']
            else:
                logger.error(f"Invalid API response: {response_data}")
                return None
                
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            return None


class RSSProcessor:
    """Main processor for RSS feeds and articles."""

    def __init__(self, config: Config, db_session: Session):
        """Initialize the processor with configuration and database session."""
        self.config = config
        self.db_session = db_session
        self._embedding_model = None
        self._vector_index = None
        self._anthropic_client = None
        
        # Initialize AI provider client
        if config.ai_provider == "anthropic":
            self._anthropic_client = AnthropicOAuthClient(config)

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

    def _call_ai_api(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Call the configured AI API with a prompt.

        Args:
            prompt: Prompt to send to the API
            model: Model to use (default: config value)

        Returns:
            Response from the API or None if failed
        """
        if self.config.ai_provider == "anthropic":
            return self._call_anthropic_api(prompt, model)
        elif self.config.ai_provider == "openrouter":
            return self._call_openrouter_api(prompt, model)
        else:
            logger.error(f"Unknown AI provider: {self.config.ai_provider}")
            return None
    
    def _call_anthropic_api(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Call the Anthropic API with a prompt using OAuth.

        Args:
            prompt: Prompt to send to the API
            model: Model to use (default: config value)

        Returns:
            Response from the API or None if failed
        """
        if not self._anthropic_client:
            logger.error("Anthropic client not initialized")
            return None
        
        if not self._anthropic_client.is_authenticated():
            logger.error("Not authenticated with Anthropic. Please run OAuth flow first.")
            return None
        
        return self._anthropic_client.make_api_request(prompt, model)

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
    
    def authenticate_anthropic(self) -> bool:
        """Authenticate with Anthropic using OAuth."""
        if self.config.ai_provider != "anthropic":
            logger.error("Anthropic provider not selected")
            return False
        
        if not self._anthropic_client:
            self._anthropic_client = AnthropicOAuthClient(self.config)
        
        if self._anthropic_client.is_authenticated():
            logger.info("Already authenticated with Anthropic")
            return True
        
        logger.info("Starting Anthropic OAuth authentication...")
        return self._anthropic_client.start_oauth_flow()
    
    def _get_processing_model(self) -> Optional[str]:
        """Get the processing model for the configured AI provider."""
        if self.config.ai_provider == "anthropic":
            return self.config.anthropic_processing_model
        elif self.config.ai_provider == "openrouter":
            return self.config.openrouter_processing_model
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
        return self._call_ai_api(prompt)

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
        response = self._call_ai_api(prompt, self._get_processing_model())

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

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two article titles using enhanced string similarity.
        
        Args:
            title1: First article title
            title2: Second article title
            
        Returns:
            Similarity score between 0 and 1
        """
        if not title1 or not title2:
            return 0.0
            
        # Normalize titles for comparison
        def normalize_title(title):
            # Convert to lowercase and remove common words/punctuation
            title = title.lower()
            # Remove common prefixes and suffixes
            title = re.sub(r'^(breaking|exclusive|update|news|report|analysis):\s*', '', title)
            title = re.sub(r'\s*\(.*?\)\s*', '', title)  # Remove parenthetical content
            # Remove common news suffixes like source attribution
            title = re.sub(r'\s*[-–—]\s*[^-–—]*$', '', title)  # Remove "- Source Name" type suffixes
            title = re.sub(r'[^\w\s]', ' ', title)  # Remove punctuation
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
            return title
            
        norm_title1 = normalize_title(title1)
        norm_title2 = normalize_title(title2)
        
        # Calculate word overlap
        words1 = set(norm_title1.split())
        words2 = set(norm_title2.split())
        
        if not words1 or not words2:
            return 0.0
            
        # Remove common stop words that don't add semantic meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'its', 'it', 'that', 'this', 'these', 'those', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        # Jaccard similarity (intersection over union)
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Enhanced entity overlap detection
        # Look for company names, proper nouns, dollar amounts, percentages, etc.
        entities1 = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b|\$[\d,]+[BMK]?|\d+%|\b\d+\b', title1))
        entities2 = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b|\$[\d,]+[BMK]?|\d+%|\b\d+\b', title2))
        
        entity_overlap = 0.0
        if entities1 and entities2:
            entity_intersection = len(entities1 & entities2)
            entity_union = len(entities1 | entities2)
            entity_overlap = entity_intersection / entity_union if entity_union > 0 else 0.0
        
        # Calculate semantic similarity using key phrases
        def extract_key_phrases(title):
            # Extract multi-word phrases that might be semantically important
            phrases = []
            words = title.split()
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i + 1]) > 2:  # Skip short words
                    phrases.append(f"{words[i]} {words[i + 1]}")
                if i < len(words) - 2 and len(words[i]) > 2 and len(words[i + 1]) > 2 and len(words[i + 2]) > 2:
                    phrases.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
            return set(phrases)
        
        phrases1 = extract_key_phrases(norm_title1)
        phrases2 = extract_key_phrases(norm_title2)
        
        phrase_overlap = 0.0
        if phrases1 and phrases2:
            phrase_intersection = len(phrases1 & phrases2)
            phrase_union = len(phrases1 | phrases2)
            phrase_overlap = phrase_intersection / phrase_union if phrase_union > 0 else 0.0
        
        # Check for synonym/variation patterns common in news
        synonym_boost = 0.0
        synonyms = [
            {'delay', 'postpone', 'push back', 'reschedule'},
            {'announce', 'reveal', 'unveil', 'disclose'},
            {'launch', 'release', 'debut', 'introduce'},
            {'again', 'once more', 'another'},
            {'model', 'system', 'ai'},
            {'open', 'open-source', 'open-weight'}
        ]
        
        for synonym_set in synonyms:
            if any(word in norm_title1 for word in synonym_set) and any(word in norm_title2 for word in synonym_set):
                synonym_boost += 0.1
        
        # Weighted combination: prioritize word overlap, then entities, then phrases
        base_similarity = 0.5 * jaccard + 0.3 * entity_overlap + 0.2 * phrase_overlap
        
        # Apply synonym boost
        final_similarity = min(1.0, base_similarity + synonym_boost)
        
        return final_similarity

    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """
        Calculate similarity between two URLs based on domain and path.
        
        Args:
            url1: First article URL
            url2: Second article URL
            
        Returns:
            Similarity score between 0 and 1
        """
        if not url1 or not url2:
            return 0.0
            
        try:
            from urllib.parse import urlparse
            
            parsed1 = urlparse(url1)
            parsed2 = urlparse(url2)
            
            # Same domain gets a boost
            domain_similarity = 0.0
            if parsed1.netloc == parsed2.netloc:
                domain_similarity = 1.0
            elif parsed1.netloc and parsed2.netloc:
                # Check for subdomain similarity (e.g., techcrunch.com vs www.techcrunch.com)
                domain1 = parsed1.netloc.replace('www.', '').lower()
                domain2 = parsed2.netloc.replace('www.', '').lower()
                if domain1 == domain2:
                    domain_similarity = 1.0
                elif domain1 in domain2 or domain2 in domain1:
                    domain_similarity = 0.8
            
            # Path similarity (for articles on same domain with similar paths)
            path_similarity = 0.0
            if parsed1.path and parsed2.path and domain_similarity > 0:
                path1_parts = set(parsed1.path.strip('/').split('/'))
                path2_parts = set(parsed2.path.strip('/').split('/'))
                if path1_parts and path2_parts:
                    path_intersection = len(path1_parts & path2_parts)
                    path_union = len(path1_parts | path2_parts)
                    path_similarity = path_intersection / path_union if path_union > 0 else 0.0
            
            # Weight domain similarity heavily, path similarity lightly
            return 0.8 * domain_similarity + 0.2 * path_similarity
            
        except Exception:
            # If URL parsing fails, return 0
            return 0.0

    def _store_embedding(self, article_id: int, embedding: List[float]) -> None:
        """
        Store an embedding vector in the Annoy index.

        Args:
            article_id: ID of the article
            embedding: Embedding vector
        """
        # Buffer embeddings for batch rebuild (Annoy indices are immutable after build)
        if not hasattr(self, '_pending_embeddings'):
            self._pending_embeddings = {}
        self._pending_embeddings[article_id] = embedding

    def _flush_embeddings(self) -> int:
        """
        Rebuild the Annoy index from all embedding vectors stored in the database.

        Annoy indices are immutable after build(), so we must rebuild from scratch
        whenever new embeddings are added. Embedding vectors are persisted in the
        Article.embedding_vector column so rebuilds don't require re-computation.

        Returns:
            Number of new embeddings added in this batch
        """
        pending = getattr(self, '_pending_embeddings', {})
        if not pending:
            return 0

        # Persist pending vectors to the database
        for article_id, embedding in pending.items():
            article = self.db_session.query(Article).filter_by(id=article_id).first()
            if article:
                article.embedding_vector = json.dumps(embedding)
        self.db_session.commit()

        # Rebuild the full index from all stored vectors
        embedding_dim = 768  # all-mpnet-base-v2 dimension
        new_index = AnnoyIndex(embedding_dim, self.config.annoy_metric)

        articles_with_vectors = self.db_session.query(Article).filter(
            Article.embedding_generated == True,
            Article.embedding_vector.isnot(None)
        ).all()

        indexed = 0
        for article in articles_with_vectors:
            try:
                vector = json.loads(article.embedding_vector)
                new_index.add_item(article.id, vector)
                indexed += 1
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Skipping article {article.id}: bad embedding vector: {e}")

        if indexed > 0:
            new_index.build(self.config.annoy_n_trees)
            new_index.save(self.config.annoy_index_path)
            logger.info(f"Rebuilt vector index with {indexed} embeddings ({len(pending)} new)")

        # Clear state for next batch
        self._vector_index = None  # Lazy-reload on next search
        self._pending_embeddings = {}

        return len(pending)

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
        Find similar articles to the given article using both content embeddings and title similarity.

        Args:
            article: Article to find similar articles for
            similarity_threshold: Minimum similarity score (default: config value)

        Returns:
            List of similar articles
        """
        if not article.content:
            return []

        threshold = similarity_threshold or self.config.similarity_threshold
        similar_articles = []

        # Method 1: Content-based similarity using embeddings
        content_similar = self._find_content_similar_articles(article, threshold)
        similar_articles.extend(content_similar)

        # Method 2: Title-based similarity (catches articles with similar titles but different content)
        title_similar = self._find_title_similar_articles(article, threshold)
        
        # Combine and deduplicate results
        all_similar = {}
        for similar_article in similar_articles + title_similar:
            if similar_article.id != article.id and not similar_article.processed:
                all_similar[similar_article.id] = similar_article

        return list(all_similar.values())

    def _find_content_similar_articles(self, article: Article, threshold: float) -> List[Article]:
        """Find articles similar by content using embeddings."""
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
            logger.warning(f"Error finding content-similar articles for {article.title}: {str(e)}")
            return []

    def _find_title_similar_articles(self, article: Article, threshold: float) -> List[Article]:
        """Find articles similar by title and URL using enhanced similarity detection."""
        if not article.title:
            return []

        try:
            # Get recent unprocessed articles to compare against
            recent_articles = self.db_session.query(Article).filter(
                Article.processed == False,
                Article.id != article.id
            ).limit(100).all()  # Limit to avoid performance issues

            similar_articles = []
            for other_article in recent_articles:
                if other_article.title:
                    # Calculate title similarity
                    title_similarity = self._calculate_title_similarity(article.title, other_article.title)
                    
                    # Calculate URL similarity
                    url_similarity = self._calculate_url_similarity(article.url, other_article.url)
                    
                    # Combined similarity score - weight title more heavily
                    combined_similarity = 0.8 * title_similarity + 0.2 * url_similarity
                    
                    # Use adaptive threshold: lower for title similarity since it's enhanced
                    # Also give boost if URLs are from same domain
                    effective_threshold = threshold - 0.15  # More lenient than before
                    if url_similarity > 0.8:  # Same domain
                        effective_threshold -= 0.1  # Even more lenient for same-domain articles
                    
                    if combined_similarity >= effective_threshold:
                        logger.info(f"Combined similarity {combined_similarity:.2f} (title: {title_similarity:.2f}, url: {url_similarity:.2f}) between '{article.title}' and '{other_article.title}'")
                        similar_articles.append(other_article)

            return similar_articles

        except Exception as e:
            logger.warning(f"Error finding title-similar articles for {article.title}: {str(e)}")
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

        return self._call_ai_api(grouped_prompt)

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
            
            # Flush all pending embeddings to the Annoy index in one batch rebuild
            logger.info("Building vector index with new embeddings...")
            self._flush_embeddings()
            self.db_session.commit()

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
        llm_output = self._call_ai_api(prompt, self._get_processing_model())

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

    def _group_articles_for_digest(self, articles: List[Article]) -> List[List[Article]]:
        """
        Group articles by similarity for digest display.
        This is a post-processing step that catches articles missed during initial grouping.
        
        Args:
            articles: List of articles to group
            
        Returns:
            List of article groups
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
            
            # Find similar articles based on title similarity
            for other_article in articles:
                if (other_article.id != article.id and 
                    other_article.id not in processed_ids and
                    other_article.title and article.title):
                    
                    title_similarity = self._calculate_title_similarity(article.title, other_article.title)
                    
                    # Use a lower threshold for digest grouping to catch more similar articles
                    if title_similarity >= 0.6:  # 60% title similarity
                        logger.info(f"Grouping for digest: {title_similarity:.2f} similarity between '{article.title}' and '{other_article.title}'")
                        group.append(other_article)
                        processed_ids.add(other_article.id)
            
            groups.append(group)
        
        # Log grouping results
        grouped_count = sum(1 for group in groups if len(group) > 1)
        if grouped_count > 0:
            logger.info(f"Digest grouping: Created {grouped_count} groups from {len(articles)} articles")
            for i, group in enumerate(groups):
                if len(group) > 1:
                    logger.info(f"  Group {i+1}: {len(group)} articles")
                    for article in group:
                        logger.info(f"    - {article.title[:60]}...")
        
        return groups

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

            # First, deduplicate by URL - keep the highest quality version of each URL
            url_to_best_article = {}
            for article in sorted_articles:
                if article.url not in url_to_best_article:
                    url_to_best_article[article.url] = article
                else:
                    # Keep the article with higher quality tier (already sorted by quality)
                    # Since articles are sorted by quality, the first one is already the best
                    continue
            
            deduplicated_articles = list(url_to_best_article.values())
            
            # Group similar articles for display - use improved similarity detection
            article_groups = self._group_articles_for_digest(deduplicated_articles)
            
            # Convert groups to summary groups format for compatibility
            summary_groups = {}
            for i, group in enumerate(article_groups):
                if len(group) == 1:
                    # Single article
                    article = group[0]
                    summary_key = article.summary if article.summary else f"no_summary_{article.id}"
                    summary_groups[summary_key] = group
                else:
                    # Multiple similar articles
                    summary_key = f"grouped_{i}"
                    summary_groups[summary_key] = group

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
                    # Multiple articles with same summary - display as merged story
                    logger.info(f"Displaying {len(articles_in_group)} similar articles as a merged story")
                    
                    # Create a comprehensive title by finding the most detailed/informative one
                    titles = [a.title for a in articles_in_group if a.title]
                    if titles:
                        # Sort by length (descending) and informativeness to pick the most detailed title
                        detailed_title = max(titles, key=lambda t: (len(t), t.count(' ')))
                        # Extract the primary source from the first (highest quality) article
                        first_article = articles_in_group[0]
                        primary_feed = self.db_session.query(Feed).filter_by(id=first_article.feed_id).first()
                        primary_source = primary_feed.title if primary_feed else "Unknown Feed"
                        
                        # Format like user's example: "Title (Primary Source/Author)"
                        summary_items.append(f"### {detailed_title}")
                    else:
                        summary_items.append(f"### Similar Articles ({len(articles_in_group)} sources)")
                    
                    # Create source metadata line combining all sources
                    source_info = []
                    all_dates = []
                    all_tiers = []
                    
                    for article in articles_in_group:
                        feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                        feed_name = feed.title if feed else "Unknown Feed"
                        source_info.append(feed_name)
                        
                        if article.published_at:
                            all_dates.append(article.published_at)
                        if article.quality_tier:
                            all_tiers.append(article.quality_tier)
                    
                    # Use most recent date and best quality tier
                    best_date = max(all_dates).strftime("%Y-%m-%d") if all_dates else "Unknown date"
                    best_tier = min(all_tiers, key=lambda t: ["S", "A", "B", "C", "D"].index(t)) if all_tiers else ""
                    tier_display = f"{best_tier}-Tier" if best_tier else ""
                    
                    # Create sources line showing multiple sources
                    if len(source_info) > 1:
                        sources_display = f"Multiple sources | {best_date} | {tier_display}"
                    else:
                        sources_display = f"{source_info[0]} | {best_date} | {tier_display}"
                    
                    summary_items.append(f"*{sources_display}*")
                    summary_items.append("")
                    
                    # List all article URLs
                    for article in articles_in_group:
                        summary_items.append(article.url)
                        summary_items.append("")
                    
                    # Use the summary from the best/first article, or combined summary if available
                    best_article = articles_in_group[0]  # Already sorted by quality
                    if best_article.summary:
                        summary_items.append(f"{best_article.summary}\n")
                    
                    # Add source attribution at the end for merged stories
                    if len(articles_in_group) > 1:
                        source_attribution = []
                        for i, article in enumerate(articles_in_group, 1):
                            feed = self.db_session.query(Feed).filter_by(id=article.feed_id).first()
                            feed_name = feed.title if feed else "Unknown Feed"
                            source_attribution.append(f"Source {i}: {feed_name}")
                        summary_items.append(f"*Sources: {', '.join(source_attribution)}*\n")

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