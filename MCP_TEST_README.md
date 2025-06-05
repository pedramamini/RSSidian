# RSSidian MCP Test Client

This directory contains a comprehensive test client for the RSSidian MCP (Model Context Protocol) service.

## Overview

The RSSidian MCP service exposes an HTTP API that allows AI models and other clients to interact with RSS feed content through a standardized protocol. This test client validates all the available endpoints and functionality.

## Prerequisites

1. **RSSidian MCP Service Running**: Make sure the RSSidian MCP service is running:
   ```bash
   rssidian mcp
   ```
   This should start the service on `http://127.0.0.1:8080`

2. **Python Dependencies**: Install the required dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

## Usage

### Run All Tests

```bash
python test_mcp_client.py
```

### Run Specific Tests

```bash
# Test service discovery
python test_mcp_client.py --test discovery

# Test subscription listing
python test_mcp_client.py --test subscriptions

# Test article listing
python test_mcp_client.py --test articles

# Test article content retrieval
python test_mcp_client.py --test content

# Test semantic search
python test_mcp_client.py --test search

# Test digest generation
python test_mcp_client.py --test digest

# Test feed statistics
python test_mcp_client.py --test stats

# Test natural language queries
python test_mcp_client.py --test query

# Test error handling
python test_mcp_client.py --test errors
```

### Custom Service URL

If your MCP service is running on a different host/port:

```bash
python test_mcp_client.py --url http://localhost:8080
```

## Available MCP Endpoints

The test client validates the following MCP endpoints:

### 1. Service Discovery (`/api/v1/mcp`)
- **Method**: GET
- **Purpose**: Discover available capabilities and endpoints
- **Response**: Service metadata, capabilities list, and endpoint URLs

### 2. List Subscriptions (`/api/v1/mcp/subscriptions`)
- **Method**: GET
- **Purpose**: Get all RSS feed subscriptions
- **Response**: Array of feed objects with metadata

### 3. List Articles (`/api/v1/mcp/articles`)
- **Method**: GET
- **Purpose**: Get articles with filtering options
- **Parameters**:
  - `limit`: Maximum number of articles (1-1000)
  - `offset`: Number of articles to skip
  - `feed_title`: Filter by specific feed
  - `days_back`: Only articles from last N days
  - `min_quality_tier`: Minimum quality tier (S, A, B, C, D)

### 4. Get Article Content (`/api/v1/mcp/articles/{article_id}/content`)
- **Method**: GET
- **Purpose**: Get full content of a specific article
- **Response**: Complete article data including content, summary, and metadata

### 5. Search Articles (`/api/v1/mcp/search`)
- **Method**: GET
- **Purpose**: Semantic search through article content
- **Parameters**:
  - `query`: Search query string
  - `relevance`: Minimum relevance threshold (0-1)
  - `max_results`: Maximum number of results (1-100)

### 6. Get Digest (`/api/v1/mcp/digest`)
- **Method**: GET
- **Purpose**: Generate a digest of high-value articles
- **Parameters**:
  - `days_back`: Number of days to look back (1-90)

### 7. Get Feed Statistics (`/api/v1/mcp/feed-stats`)
- **Method**: GET
- **Purpose**: Get statistics for feeds
- **Parameters**:
  - `feed_title`: Optional specific feed title

### 8. Natural Language Query (`/api/v1/mcp/query`)
- **Method**: POST
- **Purpose**: Process natural language queries about RSS content
- **Body**: JSON with query and optional context parameters

## Test Output

The test client provides detailed output for each test:

- ✅ **Success indicators** for passed tests
- ❌ **Error indicators** for failed tests
- 📊 **Summary statistics** at the end
- 🔍 **Detailed response data** for verification

## Example Output

```
🚀 Starting RSSidian MCP Service Tests
==================================================
Running: Service Discovery
🔍 Testing MCP Discovery...
✅ Service: RSSidian MCP
   Version: 1.0.0
   Description: Model Context Protocol for RSSidian RSS feed manager
   Capabilities: list_subscriptions, list_articles, get_article_content, search_articles, get_digest, get_feed_stats, natural_language_query

Running: List Subscriptions
📋 Testing List Subscriptions...
✅ Found 15 subscriptions
   1. TechCrunch (https://techcrunch.com/feed/)
      Muted: False, Articles: 245
   2. Hacker News (https://hnrss.org/frontpage)
      Muted: False, Articles: 1203
   ... and 13 more

📊 Test Results: 9/9 tests passed
🎉 All tests passed!
```

## Quality Tiers

The RSSidian system uses a quality tier system for articles:

- **S-Tier**: Exceptional quality (highest value)
- **A-Tier**: High quality
- **B-Tier**: Good quality
- **C-Tier**: Average quality
- **D-Tier**: Lower quality

## Troubleshooting

### Service Not Available
```
❌ MCP service not available at http://127.0.0.1:8080
   Make sure the service is running with: rssidian mcp
```

**Solution**: Start the RSSidian MCP service:
```bash
rssidian mcp
```

### Connection Refused
```
❌ Cannot connect to MCP service at http://127.0.0.1:8080
```

**Solutions**:
1. Check if the service is running
2. Verify the correct port (default: 8080)
3. Check firewall settings
4. Try a different URL with `--url` parameter

### No Data Available
If tests pass but return empty results, it might mean:
1. No RSS feeds are configured
2. No articles have been ingested yet
3. The database is empty

**Solution**: Run the ingestion process:
```bash
rssidian ingest
```

## Integration with AI Models

This MCP service is designed to work with AI models that support the Model Context Protocol. The endpoints provide structured access to RSS content that can be used for:

- Content summarization
- Trend analysis
- Research assistance
- Content discovery
- Feed management

## Contributing

To extend the test client:

1. Add new test methods to the `RSSidianMCPClient` class
2. Follow the naming convention: `test_<functionality>()`
3. Return `True` for success, `False` for failure
4. Add appropriate error handling and user feedback
5. Update the test choices in the argument parser

## License

This test client follows the same license as the main RSSidian project.