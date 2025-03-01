# RSSidian

RSSidian is a powerful tool that bridges your RSS feed subscriptions with Obsidian, creating an automated pipeline for article content analysis and knowledge management.

## Features

- **OPML Integration**:
  - Automatically imports RSS subscriptions from OPML files
  - Smart feed management with mute/unmute capabilities
  - Easy subscription organization and article listing
- **RSS Feed Processing**:
  - Retrieves and parses RSS feeds to discover new articles
  - Defaults to processing only recent articles (last 7 days)
  - Configurable lookback period for older content
- **Smart Storage**:
  - SQLite3 database for article metadata and full content
  - Annoy vector index for fast semantic search
  - Vector embeddings for efficient content discovery
  - Configurable Obsidian markdown export
- **AI-Powered Analysis**: 
  - Uses OpenRouter to generate customized summaries and insights
  - Smart content quality assessment (S/A/B/C/D tier system)
  - Value analysis with numerical scoring (1-100)
  - Configurable quality thresholds for content filtering
- **Digest Generation**:
  - Creates comprehensive digest notes in Obsidian
  - Collapses similar/overlapping stories for concise presentation
  - Organizes content by quality tier and relevance
  - Includes feed statistics and processing metrics
- **Natural Language Search**:
  - Fast semantic search powered by Annoy library
  - Intelligent search that understands the meaning of your queries
  - Finds relevant content even when exact words don't match
  - Configurable relevance threshold for fine-tuning results
  - Results grouped by feed with relevant excerpts
- **Obsidian Integration**: Generates markdown notes with customizable templates
- **MCP Service API**: Exposes a Message Control Program service for AI agents

## Installation

```bash
# Clone the repository
git clone https://github.com/pedramamini/rssidian.git
cd rssidian

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install hatch
uv pip install -e .

# Or if you prefer using regular pip
python -m venv .venv
source .venv/bin/activate
pip install hatch
pip install -e .
```

Note: We use hatch as our build system. The -e flag installs the package in editable mode, which is recommended for development.

## OSX XCode Issues
If you have build issues, try:

```bash
sudo xcode-select --reset
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

sudo xcodebuild -license

export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
export CFLAGS="-isysroot $SDKROOT -I$SDKROOT/usr/include"
export CXXFLAGS="$CFLAGS"
export LDFLAGS="-L$SDKROOT/usr/lib"
export PATH="$SDKROOT/usr/bin:$PATH"

rm -rf ~/.cache/uv/builds-v0
uv pip install -e .
```

## Configuration

Initialize configuration:

```bash
rssidian init
```

This creates a config file at `~/.config/rssidian/config.toml`

Configure settings:

```toml
[obsidian]
# Path to your Obsidian vault
vault_path = "~/Documents/Obsidian"

# Template for generated digest note
# Available variables: {date_range}, {summary_items}, {value_analysis}, {feed_stats}
template = """{date_range}

# Feed Digest

## Top Stories
{summary_items}

## Feed Statistics
{feed_stats}
"""

[search]
# Search configuration options
# Length of context to include in search excerpts (in characters)
excerpt_length = 300

[openrouter]
# OpenRouter API configuration
# API key can also be set via RSSIDIAN_OPENROUTER_API_KEY environment variable
api_key = ""

# Model to use for topic detection and article processing
processing_model = "openai/gpt-4o"

# Sample size in characters for topic detection
topic_sample_size = 4096

# Model to use for summarization
# See https://openrouter.ai/docs for available models
model = "openai/gpt-4o"

# Prompt template for processing articles
# Available variables: {content}
prompt = """Your custom prompt template here.
Available variable: {content}"""

# Enable value analysis in output
# When enabled, each article will include a Value assessment
value_prompt_enabled = true

# Minimum quality tier to include in digest (S, A, B, C, D)
# Articles with lower quality tiers will be discarded
minimum_quality_tier = "B"

[annoy]
# Path to vector index file
index_path = "~/.config/rssidian/annoy.idx"

# Number of trees (more = better accuracy but slower build)
n_trees = 10

# Distance metric (angular = cosine similarity)
metric = "angular"
```

## Usage

```bash
# Initialize configuration
rssidian init

# Show configuration and system status
rssidian show-config    # Displays config, vector index status, and article stats

# Import OPML file with subscriptions
rssidian import path/to/subscriptions.opml

# Manage feed subscriptions
rssidian subscriptions list              # List all subscriptions (sorted alphabetically)
rssidian subscriptions list --sort=articles  # List all subscriptions (sorted by article count)
rssidian subscriptions mute "Feed Title"    # Mute a feed (skip during ingestion)
rssidian subscriptions unmute "Feed Title"  # Unmute a feed

# Process new articles (last 7 days by default)
rssidian ingest

# Process articles from last 30 days with debug output
rssidian ingest --lookback 30 --debug

# Search through article content using natural language (default relevance)
rssidian search "impact of blockchain on cybersecurity"

# Search with custom relevance threshold
rssidian search "meditation techniques for beginners" --relevance 75

# Force refresh of search index before searching
rssidian search "blockchain" --refresh

# Start the MCP service
rssidian mcp --port 8080
```

## Database Backup

RSSidian includes a robust backup system to help you safeguard your article database:

- Automatic Timestamping: Backups are automatically named with YYYY-MM-DD format
- Multiple Daily Backups: System automatically handles multiple backups on the same day by adding an index
- Safe Restore Process: Creates temporary backup before restore in case of failures

Commands:

```bash
# Create a new backup
rssidian backup create

# List all backups with sizes and dates
rssidian backup list

# Restore from a specific date
rssidian backup restore 2025-02-24
```

When restoring a backup, RSSidian will:

- Show size difference between current and backup database
- Display time difference between current and backup
- Require explicit confirmation before proceeding
- Create a temporary backup of your current database as a safety measure

## System Status

Use the show-config command to view the current state of your RSSidian installation:

```bash
rssidian show-config
```

This will display:
- Vector index location and size
- Number of total articles
- Number of articles with embeddings
- Other configuration settings

## How It Works

### Feed Management:
- Imports feed URLs from OPML file
- Stores subscription data in SQLite database
- Enables muting/unmuting of specific feeds

### Content Processing:
- Fetches articles from RSS feeds
- Generates vector embeddings
- Updates Annoy vector index
- Stores in SQLite database

### AI Processing:
- Generates summaries via OpenRouter
- Analyzes content value (S/A/B/C/D tiers)
- Assigns quality scores (1-100)
- Filters out low-value content (configurable threshold)

### Knowledge Integration:
- Creates digest notes in Obsidian
- Organizes content by quality and relevance
- Includes feed statistics and processing metrics
- Enables semantic search across all content

## MCP Service API

RESTful API for AI agent integration:

```
# Base URL
http://localhost:8080/api/v1

# Endpoints
GET  /search                       # Natural language search across articles
GET  /articles                     # List all processed articles
GET  /articles/:id                 # Get article details and content
GET  /subscriptions                # List all subscriptions with mute state
POST /subscriptions/:title/mute    # Mute a feed subscription
POST /subscriptions/:title/unmute  # Unmute a feed subscription
```

## Requirements

- Python 3.9+
- OpenRouter API access
- OPML file with RSS subscriptions
- Obsidian vault (optional)

## Development

```bash
# Setup development environment
./scripts/setup_dev.sh

# Activate environment
source .venv/bin/activate
```

## License

This project is open source and available under the MIT License.