# RSSidian

RSSidian is a powerful tool that bridges your RSS feed subscriptions with Obsidian, creating an automated pipeline for article content analysis and knowledge management.

## Features

- **OPML Integration**: 
  - Import RSS subscriptions from OPML files
  - Manage feed subscriptions easily
  - Mute/unmute specific feeds
- **RSS Feed Processing**: 
  - Retrieves and parses RSS feeds to discover new articles
  - Smart filtering with configurable lookback period
  - Duplicate/similar content detection
- **Smart Storage**:
  - SQLite3 database for article metadata and content
  - Annoy vector index for fast semantic search
  - Vector embeddings for efficient content discovery
- **AI-Powered Analysis**: 
  - Uses OpenRouter to generate customized summaries and insights
  - Value analysis of content quality
  - Intelligent filtering based on quality thresholds
- **Digest Generation**:
  - Creates a single, comprehensive digest note
  - Collapses overlapping stories
  - Configurable quality thresholds (S/A/B/C/D tiers)
- **Natural Language Search**: 
  - Fast semantic search powered by Annoy library
  - Intelligent search that understands the meaning of your queries
  - Finds relevant content even when exact words don't match
- **Obsidian Integration**: Generates markdown notes with customizable templates

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rssidian.git
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

## Configuration

1. Initialize configuration:
```bash
rssidian init
```
This creates a config file at `~/.config/rssidian/config.toml`

2. Edit the configuration file to set up OpenRouter API, Obsidian vault path, and other settings.

## Usage

```bash
# Initialize configuration
rssidian init

# Import OPML file with subscriptions
rssidian import path/to/subscriptions.opml

# List all subscriptions
rssidian subscriptions list

# Mute/unmute specific feeds
rssidian subscriptions mute "Feed Title"
rssidian subscriptions unmute "Feed Title"

# Process new articles (last 7 days by default)
rssidian ingest

# Process articles from last 30 days
rssidian ingest --lookback 30

# Search through article content using natural language
rssidian search "impact of blockchain on cybersecurity"

# Start the MCP service
rssidian mcp --port 8080
```

## How It Works

1. **Feed Management**:
   - Imports feed URLs from OPML file
   - Stores subscription data in SQLite database
   - Enables muting/unmuting of specific feeds

2. **Content Processing**:
   - Fetches articles from RSS feeds
   - Generates vector embeddings
   - Updates Annoy vector index
   - Stores in SQLite database

3. **AI Processing**:
   - Generates summaries via OpenRouter
   - Analyzes content value (S/A/B/C/D tiers)
   - Filters out low-value content (C/D tiers by default)
   - Collapses similar/overlapping stories

4. **Digest Generation**:
   - Creates a single Obsidian note with digest
   - Titles include date range of content
   - Organized by relevance and quality
   - Includes statistics about processed content

## Requirements

- Python 3.9+
- OpenRouter API access
- OPML file with RSS subscriptions
- Obsidian vault (optional)

## License

This project is open source and available under the MIT License.