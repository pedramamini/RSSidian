# RSSidian

RSSidian is a powerful tool that bridges your RSS feed subscriptions with Obsidian, creating an automated pipeline for article content analysis and knowledge management.

## Features

- **OPML Integration**:
  - Automatically imports RSS subscriptions from OPML files
  - Direct feed addition via command line (no OPML required)
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
  - Feed statistics tracking (article count, quality ratings)
  - Configurable Obsidian markdown export
- **AI-Powered Analysis**:
  - Supports multiple AI providers: OpenRouter and Anthropic (with OAuth 2.0)
  - Smart content quality assessment (S/A/B/C/D tier system)
  - Value analysis with numerical scoring (1-100)
  - Configurable quality thresholds for content filtering
  - Cost tracking for all AI API calls
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
- **Obsidian Integration**:
  - Generates markdown notes with customizable templates
- **MCP Service API**:
  -Exposes a Model Context Protocol service for AI agents

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
[ai]
# AI provider selection - choose between "openrouter" or "anthropic"
provider = "openrouter"

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

[anthropic]
# Anthropic OAuth configuration (when using ai.provider = "anthropic")
# Get these credentials from https://console.anthropic.com/

# OAuth client credentials (required for OAuth)
client_id = ""
client_secret = ""

# OAuth redirect URI (should match what you configure in Anthropic console)
redirect_uri = "http://localhost:8080/callback"

# Model configuration
model = "claude-3-5-sonnet-20241022"
processing_model = "claude-3-5-sonnet-20241022"

# System prompt for Claude (this identifies the application as Claude Code)
system_prompt = "You are Claude Code, Anthropic's official CLI for Claude."

# Token storage file (tokens will be saved here after OAuth authentication)
token_file = "~/.config/rssidian/anthropic_tokens.json"

[annoy]
# Path to vector index file
index_path = "~/.config/rssidian/annoy.idx"

# Number of trees (more = better accuracy but slower build)
n_trees = 10

# Distance metric (angular = cosine similarity)
metric = "angular"
```

## Authentication

RSSidian supports two AI providers for content analysis: **OpenRouter** and **Anthropic**. Choose the provider that best fits your needs:

### OpenRouter (API Key)

OpenRouter uses simple API key authentication:

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Set it in your config file or environment variable:
   ```bash
   # In config.toml
   [ai]
   provider = "openrouter"
   
   [openrouter]
   api_key = "your-openrouter-key"
   
   # Or via environment variable
   export RSSIDIAN_OPENROUTER_API_KEY="your-openrouter-key"
   ```

### Anthropic (OAuth 2.0) - Recommended for Server Deployment

Anthropic uses OAuth 2.0 authentication, which provides **long-lasting tokens** perfect for unattended server deployment. Once authenticated, RSSidian will automatically refresh tokens as needed, allowing it to run continuously without manual intervention.

#### Step-by-Step OAuth Setup

**Prerequisites:**
- An Anthropic account with API access
- Access to [Anthropic Console](https://console.anthropic.com/)

**Step 1: Create OAuth Application**

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Navigate to your API settings or OAuth applications section
3. Create a new OAuth application with these settings:
   - **Application Name**: `RSSidian` (or your preferred name)
   - **Redirect URI**: `http://localhost:8080/callback`
   - **Scopes**: Select `read` and `write` permissions

4. Save your application and note down:
   - **Client ID**
   - **Client Secret**

**Step 2: Configure RSSidian**

1. Edit your configuration file (`~/.config/rssidian/config.toml`):
   ```toml
   [ai]
   provider = "anthropic"
   
   [anthropic]
   client_id = "your-client-id-here"
   client_secret = "your-client-secret-here"
   redirect_uri = "http://localhost:8080/callback"
   model = "claude-3-5-sonnet-20241022"
   processing_model = "claude-3-5-sonnet-20241022"
   token_file = "~/.config/rssidian/anthropic_tokens.json"
   ```

   **Security Note:** You can also set credentials via environment variables:
   ```bash
   export RSSIDIAN_ANTHROPIC_CLIENT_ID="your-client-id-here"
   export RSSIDIAN_ANTHROPIC_CLIENT_SECRET="your-client-secret-here"
   ```

**Step 3: Authenticate**

Run the authentication command:
```bash
rssidian auth
```

This will:
1. Open your browser to Anthropic's OAuth consent page
2. Ask you to authorize RSSidian to access your account
3. Redirect back to RSSidian with an authorization code
4. Exchange the code for long-lasting access and refresh tokens
5. Save tokens securely to `~/.config/rssidian/anthropic_tokens.json`

**Step 4: Verify Authentication**

Test that authentication worked:
```bash
rssidian ingest --debug
```

You should see successful API calls to Anthropic in the debug output.

#### Token Management for Server Deployment

**Long-Lasting Tokens**: Anthropic OAuth tokens are designed for long-term use:
- **Access tokens** are valid for extended periods
- **Refresh tokens** allow automatic token renewal
- RSSidian automatically refreshes expired tokens without user intervention

**Server Deployment Best Practices**:

1. **Authenticate Once**: Run `rssidian auth` during initial server setup
2. **Secure Token Storage**: The token file (`anthropic_tokens.json`) contains sensitive credentials:
   ```bash
   # Set restrictive permissions
   chmod 600 ~/.config/rssidian/anthropic_tokens.json
   ```

3. **Automated Refresh**: RSSidian handles token refresh automatically:
   - Detects expired tokens during API calls
   - Uses refresh token to get new access token
   - Updates token file with new credentials
   - Continues operation seamlessly

4. **Monitoring**: Monitor your logs for authentication issues:
   ```bash
   rssidian ingest --debug 2>&1 | grep -i "anthropic\|auth\|token"
   ```

5. **Backup Configuration**: Keep a secure backup of your token file for disaster recovery

**Troubleshooting OAuth**:

- **Port 8080 in use**: If port 8080 is unavailable, update your OAuth application's redirect URI to use a different port and update the config accordingly
- **Browser doesn't open**: Manually copy the authorization URL from the console and open it in your browser
- **Token refresh fails**: Re-run `rssidian auth` to get fresh tokens

#### Environment Variables Reference

For production deployments, you can set all credentials via environment variables:

```bash
# OAuth credentials
export RSSIDIAN_ANTHROPIC_CLIENT_ID="your-client-id"
export RSSIDIAN_ANTHROPIC_CLIENT_SECRET="your-client-secret"

# Optional: Direct token setting (if you have tokens from elsewhere)
export RSSIDIAN_ANTHROPIC_ACCESS_TOKEN="your-access-token"
export RSSIDIAN_ANTHROPIC_REFRESH_TOKEN="your-refresh-token"
```

## Usage

```bash
# Initialize configuration
rssidian init

# Show configuration and system status
rssidian show-config    # Displays config, vector index status, and article stats

# Authenticate with AI providers (required for Anthropic)
rssidian auth           # Start OAuth flow for Anthropic, or show info for OpenRouter

# Import and export feeds in OPML format
rssidian opml import path/to/subscriptions.opml           # Import feeds from OPML file
rssidian opml export path/to/export.opml                  # Export all feeds
rssidian opml export path/to/export.opml --exclude-muted  # Export only active feeds (exclude muted)

# Manage feed subscriptions
rssidian subscriptions add <feed_url>                      # Add a new RSS feed subscription
rssidian subscriptions add <feed_url> --title "My Feed"    # Add feed with custom title
rssidian subscriptions add <feed_url> --force              # Update existing feed metadata
rssidian subscriptions list                                # List all subscriptions (sorted alphabetically)
rssidian subscriptions list --sort=articles                # List all subscriptions (sorted by article count)
rssidian subscriptions list --sort=rating                  # List all subscriptions (sorted by quality rating)
rssidian subscriptions list --sort=updated                 # List all subscriptions (sorted by last update date)
rssidian subscriptions mute "Feed Title"                   # Mute a feed (skip during ingestion)
rssidian subscriptions unmute "Feed Title"                 # Unmute a feed

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

# Start the MCP (Model Context Protocol) service
rssidian mcp --port 8080                # Start in HTTP mode on port 8080
rssidian mcp --stdio                   # Start in STDIO mode for Claude Desktop integration
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
- Exports feeds to OPML file (with option to include/exclude muted feeds)
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
- Customizable filename templates with date variables (`{from_date}`, `{to_date}`, `{date_range}`, `{date}`, `{datetime}`)
- Enables semantic search across all content

## MCP (Model Context Protocol) Service API

RSSidian provides a Model Context Protocol (MCP) service that enables AI agents like Claude to interact with your RSS content. The MCP service exposes a RESTful API that allows for comprehensive access to your RSS-ingested content.

### Starting the MCP Service

```bash
# Start the MCP service on the default port (8080)
rssidian mcp

# Start the MCP service on a custom port
rssidian mcp --port 9000
```

### API Endpoints

```
# Base URL
http://localhost:8080/api/v1

# Standard API Endpoints
GET  /search                       # Natural language search across articles
GET  /articles                     # List all processed articles
GET  /articles/:id                 # Get article details and content
GET  /subscriptions                # List all subscriptions with mute state
POST /subscriptions/:title/mute    # Mute a feed subscription
POST /subscriptions/:title/unmute  # Unmute a feed subscription

# MCP-specific Endpoints
GET  /mcp                          # Discovery endpoint with capabilities and endpoints
GET  /mcp/subscriptions            # List all feed subscriptions
GET  /mcp/articles                 # List articles with advanced filtering
GET  /mcp/articles/:id/content     # Get full article content
GET  /mcp/search                   # Semantic search with relevance control
GET  /mcp/digest                   # Get digest of high-value articles
GET  /mcp/feed-stats               # Get feed statistics
POST /mcp/query                    # Process natural language queries
```

### Configuring Claude Desktop to Use RSSidian MCP

RSSidian MCP can be used with Claude Desktop in two ways: HTTP mode or STDIO mode.

#### Option 1: STDIO Mode (Recommended)

STDIO mode allows Claude Desktop to directly spawn and communicate with the RSSidian MCP server, without requiring a separate HTTP server to be running.

1. Edit Claude Desktop's configuration file by clicking on the Claude icon in the menu bar, selecting "Settings", and then clicking "Edit Configuration File"

2. Add a new entry to the `mcpServers` section of the JSON configuration:
   ```json
   {
     "globalShortcut": "Shift+Space",
     "mcpServers": {
       "rssidian": {
         "command": "/path/to/your/python",
         "args": [
           "-m",
           "rssidian",
           "mcp",
           "--stdio"
         ]
       }
       // other MCP servers...
     }
   }
   ```

   Replace `/path/to/your/python` with the actual path to your Python executable where RSSidian is installed. You can find this by running `which python` in the terminal where you normally run RSSidian.

3. Save the configuration file

#### Option 2: HTTP Mode

In HTTP mode, you need to start the RSSidian MCP service separately and configure Claude Desktop to connect to it.

1. Start the RSSidian MCP service manually:
   ```bash
   rssidian mcp --port 8080
   ```

2. Edit Claude Desktop's configuration file by clicking on the Claude icon in the menu bar, selecting "Settings", and then clicking "Edit Configuration File"

3. Add a new entry to the `mcpServers` section of the JSON configuration:
   ```json
   {
     "globalShortcut": "Shift+Space",
     "mcpServers": {
       "rssidian-http": {
         "command": "/path/to/your/python",
         "args": [
           "-m",
           "rssidian",
           "mcp",
           "--port",
           "8080"
         ]
       }
       // other MCP servers...
     }
   }
   ```

4. Save the configuration file

5. When chatting with Claude, you can now ask questions about your RSS content such as:
   - "What are the latest articles about AI?"
   - "Find articles about blockchain from the last week"
   - "Summarize the top quality articles from my tech feeds"
   - "What are the trending topics across my subscriptions?"

Claude will automatically use the RSSidian MCP service to access your RSS content and provide informed responses based on your feeds.

## Requirements

- Python 3.9+
- AI Provider access (choose one):
  - **OpenRouter API** - Simple API key authentication
  - **Anthropic API** - OAuth 2.0 authentication (recommended for server deployment)
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