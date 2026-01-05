# RSSidian Constitution

This document codifies the core principles, values, and design philosophy embedded in the RSSidian codebase. It serves as a guide for contributors, maintainers, and AI agents working with this project.

---

## Preamble

RSSidian exists to transform the overwhelming flood of RSS content into meaningful, curated knowledge. In an age of information abundance, we believe the challenge is not access to information but rather discerning what truly matters for human flourishing.

---

## Article I: Core Values

### Section 1.1 - Human Flourishing Over Engagement

Content quality is measured by its contribution to human meaning, not by virality, clicks, or entertainment value. The quality tier system (S/A/B/C/D) explicitly prioritizes:

- Human meaning and flourishing
- The future of human meaning in a post-AI world
- AI's impact on humanity
- Continuous human improvement
- Enhancing human creative output
- The role of art and reading in enhancing human flourishing

Content that is "interesting and high quality but not directly related to human aspects" is intentionally scored lower.

### Section 1.2 - User Time is Sacred

Every design decision respects that user attention is finite and precious:

- **Default 7-day lookback**: Recent content, not the entire archive
- **25 articles per feed**: Quality over comprehensiveness
- **Quality filtering**: Not all articles are equal; filter by default
- **Similarity merging**: Collapse duplicate stories; show news once
- **Cost tracking**: Transparency in AI API spending

### Section 1.3 - Substance Over Style

The system values depth of thought over brevity or polish:

- Ideas are counted (11-20 good, 25+ excellent)
- Political extremism is penalized regardless of quality
- A short article with 15 insights outranks a long article with 8

---

## Article II: Architectural Principles

### Section 2.1 - Separation of Concerns

The codebase maintains clear boundaries:

- **Config Layer** (`config.py`): Centralized configuration with sensible defaults
- **Data Layer** (`models.py`): SQLAlchemy ORM for persistent storage
- **Processing Layer** (`core.py`, `rss.py`): Business logic for content analysis
- **Interface Layer** (`cli.py`, `api.py`, `mcp.py`): User-facing access points
- **Utility Modules** (`markdown.py`, `opml.py`, `backup.py`): Specialized functions

### Section 2.2 - Lazy Loading for Responsiveness

Heavy dependencies (embedding models, vector indexes) are loaded only when needed:

```python
@property
def embedding_model(self):
    """Lazy load the embedding model."""
    if self._embedding_model is None:
        self._embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return self._embedding_model
```

CLI startup must remain fast. Users should not wait for ML models when running simple commands.

### Section 2.3 - Convention Over Configuration

Follow established conventions with escape hatches:

- XDG-compliant paths (`~/.config/rssidian/`, `~/.local/share/rssidian/`)
- Environment variables override config files
- Example configuration documents all options
- Sensible defaults work out of the box

---

## Article III: Error Handling Philosophy

### Section 3.1 - Graceful Degradation Over Fail-Fast

The system prefers resilience:

- **Network failures**: Retry with exponential backoff (3 attempts)
- **Missing content**: Try multiple extraction strategies before giving up
- **Date parsing**: 5+ parsing strategies in sequence
- **API failures**: Return None, log warning, continue processing
- **One bad article**: Does not stop the entire ingest

### Section 3.2 - Partial Success is Success

Process items individually:

```python
for article in batch:
    try:
        process_article(article)
    except Exception:
        logger.error(...)
        continue  # Process next article
```

A feed with 24/25 processed articles is better than 0/25 due to one failure.

### Section 3.3 - Content Extraction Fallbacks

Never give up on content extraction prematurely:

1. Try `content` field
2. Try `summary_detail`
3. Try `summary`
4. Try `description`
5. Try `content_encoded`
6. Try Jina.ai enrichment
7. Fall back to direct URL fetch

---

## Article IV: User Experience Principles

### Section 4.1 - Progressive Disclosure

Commands reveal complexity only when needed:

```
rssidian/
├── init                    # Simple: initialize
├── ingest                  # Simple: process feeds
├── search                  # Simple: find content
├── subscriptions/          # Advanced: feed management
│   ├── add
│   ├── list
│   ├── mute/unmute
│   └── enable/disable-peer-through
└── mcp                     # Advanced: API service
```

### Section 4.2 - Rich Feedback

Use visual feedback to reduce user uncertainty:

- Progress bars for long operations
- Color-coded status messages (green=success, red=error, yellow=warning)
- Tables for structured output
- Clear error messages with context

### Section 4.3 - Helpful Error Messages

When something fails:

1. State what failed
2. Provide context (which feed, which step)
3. Suggest remediation when possible

---

## Article V: Integration Principles

### Section 5.1 - Standards-Based Interoperability

Embrace established standards:

- **OPML**: Import/export feed subscriptions
- **MCP**: Model Context Protocol for AI agents
- **JSON-RPC 2.0**: Protocol compliance for MCP STDIO mode
- **Markdown**: Universal content format
- **SQLite**: Portable, zero-configuration database

### Section 5.2 - Ecosystem Citizenship

RSSidian is designed to be part of a larger ecosystem:

- Export to Obsidian, not lock users in
- Provide MCP API for AI agents, not just CLI
- Use XDG paths, not proprietary locations
- Support environment variables for containerization

### Section 5.3 - AI-First Design

MCP tools are designed for AI agent consumption:

- Numbered lists (LLMs reference by number)
- Summaries before full content (token efficiency)
- Enum constraints (prevent invalid inputs)
- Pagination guidance (LLM knows how to get more)

---

## Article VI: Quality Assessment Philosophy

### Section 6.1 - The Quality Tier System

| Tier | Score Range | Meaning | Action |
|------|-------------|---------|--------|
| S | 80-100 | Must Consume | Within a week |
| A | 60-79 | Should Consume | This month |
| B | 40-59 | Worth Reading | When time allows |
| C | 20-39 | Maybe Skip | Low priority |
| D | 1-19 | Definitely Skip | Filter out |

### Section 6.2 - What Increases Quality

- High idea density (18+ ideas = S tier potential)
- Strong alignment with human flourishing themes
- Original thinking and insights
- Actionable wisdom

### Section 6.3 - What Decreases Quality

- Low idea density regardless of length
- Pure entertainment without substance
- Extreme or populist political advocacy
- Technical content without human implications
- Derivative or redundant information

---

## Article VII: Similarity Detection Philosophy

### Section 7.1 - Same Story, Different Words

News articles are often reported identically by multiple outlets. The system recognizes "same story, different words" through:

1. **Word-level similarity**: Jaccard coefficient of significant words
2. **Entity overlap**: Company names, numbers, percentages
3. **Key phrase matching**: 2-3 word phrase comparison
4. **Synonym recognition**: "launch" = "release" = "debut"
5. **URL analysis**: Same domain implies related content

### Section 7.2 - Prefer False Merges Over False Splits

When uncertain, combine similar articles rather than show duplicates. Users prefer seeing a story once over seeing it three times with slight variations.

### Section 7.3 - Domain Reputation

Articles from the same domain about the same topic are likely the same story. Apply more lenient thresholds for same-domain comparisons.

---

## Article VIII: Data Principles

### Section 8.1 - Minimal but Complete Schema

Store what's needed, nothing more:

- **Feed**: Identity, health metrics, statistics
- **Article**: Content, metadata, processing state, quality scores
- **Relationships**: Cascading deletes maintain integrity

### Section 8.2 - Audit Trail

Track processing stages independently:

- `processed`: Core processing complete
- `embedding_generated`: Vector index updated
- `jina_enhanced`: Content enrichment attempted

This enables selective reprocessing without full reingestion.

### Section 8.3 - Statistics Denormalization

Pre-compute aggregate statistics on Feed model (`avg_quality_score`, `quality_tier_counts`) to avoid expensive queries during common operations.

---

## Article IX: Transparency Principles

### Section 9.1 - Show What's Happening

- Comprehensive logging (DEBUG, INFO, WARNING, ERROR)
- Cost tracking for all AI API calls
- Progress indicators for long operations
- Statistics in digest output

### Section 9.2 - Configuration Visibility

The `show-config` command reveals:

- Vector index status and size
- Total articles in database
- Articles with embeddings
- All configuration locations

### Section 9.3 - No Hidden Decisions

When the system makes automatic decisions (quality filtering, similarity merging), include them in output statistics.

---

## Article X: Contributor Guidelines

### Section 10.1 - Code Style

- **Line Length**: 100 characters
- **Naming**: `snake_case` for functions, `PascalCase` for classes
- **Private Methods**: Prefix with underscore
- **Type Hints**: Required for all public interfaces
- **Documentation**: Docstrings for public methods

### Section 10.2 - Design Patterns

- Early return over deep nesting
- Property-based configuration access
- Defensive programming with fallbacks
- Individual commits per logical change

### Section 10.3 - Testing Philosophy

Focus on:

- Protocol compliance (MCP, JSON-RPC)
- End-to-end workflows
- Real agent interaction patterns
- Integration scenarios

Unit tests are less important than behavioral tests that verify real-world usage.

---

## Article XI: What RSSidian Is Not

### Section 11.1 - Not a Content Aggregator

RSSidian filters content, not aggregates it. The goal is less content of higher quality, not more content.

### Section 11.2 - Not a Replacement for Reading

AI analysis helps prioritize, not replace human engagement. Full content is always preserved and accessible.

### Section 11.3 - Not a Social Platform

There are no likes, shares, comments, or algorithmic feeds. Content quality is assessed objectively, not socially.

### Section 11.4 - Not Optimized for Engagement

The system explicitly penalizes content that is "interesting" but not meaningful. Engagement metrics are irrelevant.

---

## Article XII: Amendment Process

This constitution may be amended when:

1. Core values need clarification
2. New architectural patterns emerge
3. User feedback reveals gaps in principles
4. Technology evolution requires adaptation

Amendments should be proposed via pull request with clear rationale.

---

## Epilogue

RSSidian embodies a belief that technology should serve human flourishing, not attention capture. In curating what we read, we shape who we become. This constitution ensures that every design decision, from quality tiers to similarity thresholds, serves the goal of helping users engage with content that truly matters.

*"The goal is not to read everything, but to read what makes us more human."*
