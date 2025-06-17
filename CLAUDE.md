# CLAUDE.md - RSSidian Development Guide

## Project Overview

**RSSidian** is a Python CLI application that bridges RSS feeds with Obsidian, providing AI-powered content analysis, summarization, and semantic search capabilities. It serves as both a standalone tool and an MCP (Model Context Protocol) service for AI agents.

### Core Purpose
- Ingest and process RSS feeds with AI-powered analysis
- Generate quality-rated summaries and insights
- Provide semantic search across content
- Export digests to Obsidian in markdown format
- Expose content via MCP API for AI agent interaction

## Project Architecture

### Technology Stack
- **Language**: Python 3.9+
- **Package Management**: pyproject.toml with hatch build system
- **Database**: SQLite with SQLAlchemy ORM
- **CLI Framework**: Click with Rich for enhanced output
- **Web Framework**: FastAPI for MCP API service
- **AI/ML**: OpenRouter API, sentence-transformers, Annoy vector search
- **Testing**: Custom test suite with MCP compliance focus

### Directory Structure
```
rssidian/
├── rssidian/           # Main package
│   ├── __init__.py     # Package initialization
│   ├── cli.py          # Click-based CLI interface
│   ├── core.py         # Main RSS processing logic
│   ├── models.py       # SQLAlchemy database models
│   ├── config.py       # Configuration management
│   ├── api.py          # FastAPI web service
│   ├── mcp.py          # Model Context Protocol implementation
│   ├── mcp_stdio.py    # STDIO mode for MCP
│   ├── rss.py          # RSS feed processing
│   ├── markdown.py     # Obsidian markdown generation
│   ├── opml.py         # OPML import/export
│   ├── backup.py       # Database backup functionality
│   └── cost_tracker.py # AI API cost tracking
├── scripts/            # Development and utility scripts
├── testing/            # Test files (custom testing approach)
└── pyproject.toml      # Project configuration and dependencies
```

## Development Guidelines

### Code Style and Standards
- **Line Length**: 100 characters (configured in pyproject.toml)
- **Python Version**: 3.9+ compatibility required
- **Linting**: Uses ruff for code quality (pycodestyle, pyflakes, isort)
- **Type Hints**: Use type hints throughout the codebase
- **Documentation**: Comprehensive docstrings for all public methods

### Key Architectural Patterns

#### 1. Lazy Loading Pattern
The codebase uses lazy loading for heavy dependencies to improve startup time:
```python
@property
def embedding_model(self):
    """Lazy load the embedding model."""
    if self._embedding_model is None:
        logger.info("Loading embedding model...")
        self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return self._embedding_model
```

#### 2. Configuration Management
- Central configuration via `Config` class
- TOML-based configuration files
- Environment variable support for API keys
- Example configuration provided in `config.toml.example`

#### 3. Database Design
- SQLAlchemy ORM with declarative base
- Two main models: `Feed` and `Article`
- Relationship management with cascading deletes
- Statistics tracking built into models

#### 4. Modular CLI Design
- Command separation using Click groups
- Rich console output for enhanced UX
- Progress bars and status indicators
- Deferred imports for faster startup

### Development Commands

#### Setup Development Environment
```bash
# Use the provided setup script
./scripts/setup_dev.sh

# Or manually:
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### Running the Application
```bash
# Initialize configuration
rssidian init

# Import RSS feeds from OPML
rssidian opml import feeds.opml

# Process feeds and generate content
rssidian ingest

# Search content semantically
rssidian search "AI impact on society"

# Start MCP service
rssidian mcp --port 8080
```

## API and Integration Points

### MCP (Model Context Protocol) Service
The application implements the MCP specification for AI agent integration:

#### Key Endpoints
- `GET /mcp/subscriptions` - List feed subscriptions
- `GET /mcp/articles` - List articles with filtering
- `GET /mcp/search` - Semantic search functionality
- `GET /mcp/digest` - Generate content digests
- `POST /mcp/query` - Process natural language queries

#### STDIO Mode Support
The MCP service supports both HTTP and STDIO modes for different integration scenarios:
```bash
# HTTP mode
rssidian mcp --port 8080

# STDIO mode (for Claude Desktop integration)
rssidian mcp --stdio
```

### Configuration Integration
The application uses a comprehensive configuration system with multiple sections:
- `[obsidian]` - Obsidian vault integration settings
- `[openrouter]` - AI API configuration and prompts
- `[search]` - Semantic search parameters
- `[annoy]` - Vector index configuration
- `[feeds]` - RSS processing settings

## Testing Strategy

### Test Organization
Tests are located in the `/testing/` directory with focus on:
- MCP protocol compliance (`test_mcp_stdio_compliance.py`)
- Basic functionality (`test_mcp_basic.py`)
- Client integration (`test_mcp_client.py`)
- Service functionality (`test_mcp_service.sh`)

### Running Tests
```bash
# Run MCP compliance tests
python testing/test_mcp_stdio_compliance.py

# Run service tests
./testing/test_mcp_service.sh
```

## Key Dependencies and Their Roles

### Core Dependencies
- **feedparser**: RSS/Atom feed parsing
- **sqlalchemy**: Database ORM and management
- **click**: CLI framework
- **fastapi**: Web API framework
- **sentence-transformers**: Text embeddings
- **annoy**: Approximate nearest neighbor search
- **requests**: HTTP client for feed fetching
- **rich**: Enhanced console output

### AI/ML Stack
- **OpenRouter API**: Content analysis and summarization
- **sentence-transformers**: Semantic embeddings (all-mpnet-base-v2 model)
- **Annoy**: Vector similarity search
- **Custom prompting**: Sophisticated prompts for content quality assessment

## Development Best Practices

### Error Handling
- Comprehensive error handling with proper logging
- Rich error messages for user-facing operations
- Graceful degradation for network failures
- Database transaction management

### Performance Considerations
- Lazy loading of ML models
- Batch processing for feed ingestion
- Vector index optimization
- Database query optimization
- Progress indicators for long-running operations

### Configuration Management
- Environment variable support for sensitive data
- Comprehensive example configuration
- Validation of configuration parameters
- Sensible defaults for all options

### Database Management
- Migration-ready schema design
- Built-in backup and restore functionality
- Statistics tracking for feeds and articles
- Efficient indexing for search operations

## Quality Assurance

### AI Content Quality System
The application implements a sophisticated content quality assessment:
- **Tier System**: S/A/B/C/D quality tiers
- **Numerical Scoring**: 1-100 quality scores
- **Thematic Analysis**: Focus on human meaning and flourishing
- **Configurable Thresholds**: Filter content by minimum quality

### Cost Tracking
- Comprehensive API cost tracking
- Per-request cost monitoring
- Configurable cost reporting
- Budget awareness features

## Integration Guidelines

### Obsidian Integration
- Markdown template system
- Customizable filename patterns
- Rich content formatting
- Link generation and management

### AI Agent Integration
- Full MCP protocol compliance
- RESTful API endpoints
- Natural language query processing
- Contextual content retrieval

This guide provides the foundation for understanding and contributing to the RSSidian project. The codebase emphasizes modularity, performance, and comprehensive AI-powered content analysis while maintaining a clean, maintainable architecture.