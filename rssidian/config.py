import os
import tomli
import shutil
import pathlib
from typing import Dict, Any, Optional, List, Union

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.config/rssidian")
DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/rssidian/rssidian.db")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.toml")
DEFAULT_BACKUP_DIR = os.path.expanduser("~/.local/share/rssidian/backups")


class Config:
    """Configuration manager for RSSidian."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file."""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_dir = os.path.dirname(self.config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not os.path.exists(self.config_path):
            return {}
        
        with open(self.config_path, "rb") as f:
            return tomli.load(f)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if section not in self.config:
            return default
        if key not in self.config[section]:
            return default
        return self.config[section][key]
    
    @property
    def obsidian_vault_path(self) -> str:
        """Get Obsidian vault path."""
        path = self.get("obsidian", "vault_path", "~/Documents/Obsidian")
        return os.path.expanduser(path)
    
    @property
    def obsidian_template(self) -> str:
        """Get Obsidian note template."""
        return self.get("obsidian", "template", "{date_range}\n\n# Feed Digest\n\n## Top Stories\n{summary_items}\n\n## Feed Statistics\n{feed_stats}")
    
    @property
    def obsidian_filename_template(self) -> str:
        """Get Obsidian filename template.
        
        Available variables:
        - {date_range}: The date range of the digest (e.g., '2025-03-01 to 2025-03-02')
        - {date}: Current date in YYYY-MM-DD format
        - {datetime}: Current datetime in YYYY-MM-DD_HH-MM-SS format
        """
        return self.get("obsidian", "filename_template", "{date_range}")
    
    @property
    def db_path(self) -> str:
        """Get database path."""
        os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)
        return DEFAULT_DB_PATH
    
    @property
    def openrouter_api_key(self) -> Optional[str]:
        """Get OpenRouter API key."""
        env_key = os.environ.get("RSSIDIAN_OPENROUTER_API_KEY")
        if env_key:
            return env_key
        return self.get("openrouter", "api_key")
    
    @property
    def cost_tracking_enabled(self) -> bool:
        """Check if cost tracking is enabled."""
        return self.get("openrouter", "cost_tracking_enabled", True)
    
    @property
    def openrouter_model(self) -> str:
        """Get OpenRouter model."""
        return self.get("openrouter", "model", "openai/gpt-4o")
    
    @property
    def openrouter_processing_model(self) -> str:
        """Get OpenRouter processing model."""
        return self.get("openrouter", "processing_model", "openai/gpt-4o")
    
    @property
    def openrouter_prompt(self) -> str:
        """Get OpenRouter prompt template."""
        default_prompt = """You are a helpful article summarizer.
Given the following article content, provide:
1. A concise 1-2 sentence summary of the key points
2. The most important takeaway or insight

Content:
{content}
"""
        return self.get("openrouter", "prompt", default_prompt)
    
    @property
    def value_prompt_enabled(self) -> bool:
        """Check if value analysis is enabled."""
        return self.get("openrouter", "value_prompt_enabled", True)
    
    @property
    def minimum_quality_tier(self) -> str:
        """Get minimum quality tier for articles."""
        return self.get("openrouter", "minimum_quality_tier", "B")
    
    @property
    def value_prompt(self) -> str:
        """Get value analysis prompt template."""
        # This is a simplified version, the actual one would be longer as in the example
        return self.get("openrouter", "value_prompt", "")
    
    @property
    def topic_sample_size(self) -> int:
        """Get topic sample size."""
        return self.get("openrouter", "topic_sample_size", 4096)
    
    @property
    def annoy_index_path(self) -> str:
        """Get Annoy index path."""
        path = self.get("annoy", "index_path", "~/.config/rssidian/annoy.idx")
        expanded_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        return expanded_path
    
    @property
    def annoy_n_trees(self) -> int:
        """Get Annoy n_trees parameter."""
        return self.get("annoy", "n_trees", 10)
    
    @property
    def annoy_metric(self) -> str:
        """Get Annoy metric."""
        return self.get("annoy", "metric", "angular")
    
    @property
    def default_lookback(self) -> int:
        """Get default lookback period for RSS ingestion."""
        return self.get("feeds", "default_lookback", 7)
    
    @property
    def max_articles_per_feed(self) -> int:
        """Get maximum articles to process per feed."""
        return self.get("feeds", "max_articles_per_feed", 25)
    
    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold for duplicate detection and story merging."""
        return self.get("feeds", "similarity_threshold", 0.65)
    
    @property
    def excerpt_length(self) -> int:
        """Get search excerpt length."""
        return self.get("search", "excerpt_length", 300)
    
    @property
    def backup_dir(self) -> str:
        """Get backup directory."""
        os.makedirs(DEFAULT_BACKUP_DIR, exist_ok=True)
        return DEFAULT_BACKUP_DIR
        
    @property
    def analyze_during_ingestion(self) -> bool:
        """Check if content should be analyzed during ingestion."""
        return self.get("feeds", "analyze_during_ingestion", True)
        
    @property
    def aggregator_prompt(self) -> str:
        """Get aggregator prompt template for generating a categorized overview of articles."""
        default_prompt = """You are an expert content curator and analyst.

Your task is to organize and summarize the following article summaries into a cohesive overview, categorized by subject matter.

Group the articles into clear categories such as Politics, Science, Technology, AI/GenAI, Programming, Business, Startups, Health, etc. based on their content.

For each category:
1. Provide a brief overview of the key themes or trends
2. List the most important articles with their titles as internal links and a 1-sentence description
3. Highlight any connections or contradictions between articles in the same category

IMPORTANT: When creating links to articles, use Obsidian's internal linking format. The format should be [[{filename}#article-{{id}}|**Article Title**]] where {filename} is the name of the current digest file (without the .md extension) and {{id}} is the article ID from the anchor. DO NOT link directly to external URLs. This allows users to first jump to the article summary within the document before deciding to visit the external site.

Finally, provide a brief "Big Picture" section that identifies any cross-category trends or important developments.

Article Summaries:
{summaries}
"""
        return self.get("openrouter", "aggregator_prompt", default_prompt)
    
    # AI Provider Selection
    @property
    def ai_provider(self) -> str:
        """Get AI provider (openrouter or anthropic)."""
        return self.get("ai", "provider", "openrouter")
    
    # Anthropic OAuth Configuration
    @property
    def anthropic_client_id(self) -> Optional[str]:
        """Get Anthropic OAuth client ID."""
        env_client_id = os.environ.get("RSSIDIAN_ANTHROPIC_CLIENT_ID")
        if env_client_id:
            return env_client_id
        return self.get("anthropic", "client_id")
    
    @property
    def anthropic_client_secret(self) -> Optional[str]:
        """Get Anthropic OAuth client secret."""
        env_client_secret = os.environ.get("RSSIDIAN_ANTHROPIC_CLIENT_SECRET")
        if env_client_secret:
            return env_client_secret
        return self.get("anthropic", "client_secret")
    
    @property
    def anthropic_redirect_uri(self) -> str:
        """Get Anthropic OAuth redirect URI."""
        return self.get("anthropic", "redirect_uri", "http://localhost:8080/callback")
    
    @property
    def anthropic_model(self) -> str:
        """Get Anthropic model."""
        return self.get("anthropic", "model", "claude-3-5-sonnet-20241022")
    
    @property
    def anthropic_processing_model(self) -> str:
        """Get Anthropic processing model."""
        return self.get("anthropic", "processing_model", "claude-3-5-sonnet-20241022")
    
    @property
    def anthropic_system_prompt(self) -> str:
        """Get Anthropic system prompt."""
        return self.get("anthropic", "system_prompt", "You are Claude Code, Anthropic's official CLI for Claude.")
    
    @property
    def anthropic_access_token(self) -> Optional[str]:
        """Get Anthropic OAuth access token."""
        env_token = os.environ.get("RSSIDIAN_ANTHROPIC_ACCESS_TOKEN")
        if env_token:
            return env_token
        return self.get("anthropic", "access_token")
    
    @property
    def anthropic_refresh_token(self) -> Optional[str]:
        """Get Anthropic OAuth refresh token."""
        env_token = os.environ.get("RSSIDIAN_ANTHROPIC_REFRESH_TOKEN")
        if env_token:
            return env_token
        return self.get("anthropic", "refresh_token")
    
    @property
    def anthropic_token_file(self) -> str:
        """Get path to Anthropic token file."""
        path = self.get("anthropic", "token_file", "~/.config/rssidian/anthropic_tokens.json")
        expanded_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
        return expanded_path


def init_config() -> None:
    """Initialize configuration."""
    # Create config directories
    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)
    
    # Create example config file if it doesn't exist
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        # Get the location of the example config file
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        example_config = os.path.join(root_dir, "config.toml.example")
        
        # Copy the example config to the user's config directory
        if os.path.exists(example_config):
            shutil.copy(example_config, DEFAULT_CONFIG_PATH)
            print(f"Created configuration file at {DEFAULT_CONFIG_PATH}")
        else:
            # If example config not found, create a minimal one
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                f.write("# RSSidian Configuration\n\n")
                f.write("[obsidian]\n")
                f.write("vault_path = \"~/Documents/Obsidian\"\n\n")
                f.write("[openrouter]\n")
                f.write("api_key = \"\"\n")
            print(f"Created minimal configuration file at {DEFAULT_CONFIG_PATH}")
    else:
        print(f"Configuration file already exists at {DEFAULT_CONFIG_PATH}")
    
    print(f"Database will be stored at {DEFAULT_DB_PATH}")
    print(f"Edit {DEFAULT_CONFIG_PATH} to configure RSSidian")