#!/usr/bin/env python
"""
MCP-compliant STDIO server for RSSidian.

This server implements the Model Context Protocol (MCP) specification using
JSON-RPC 2.0 protocol over STDIO transport. It provides tools for accessing
RSSidian RSS feed functionality.

The server exposes the following tools:
- list_subscriptions: Get all RSS feed subscriptions
- list_articles: Get articles with optional filtering
- get_article_content: Get full content of a specific article
- search_articles: Search articles using semantic search
- get_digest: Get a digest of high-value articles
- get_feed_stats: Get statistics about feeds
"""

import asyncio
import json
import sys
import logging
import argparse
import datetime
from typing import Any, Dict, List, Optional

try:
    from .config import Config
    from .models import get_db_session
    from .mcp import ModelContextProtocol
except ImportError:
    # Handle case when running as standalone script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import Config
    from models import get_db_session
    from mcp import ModelContextProtocol

# Logger will be configured in main() after parsing arguments
logger = logging.getLogger("mcp_stdio")

# Global MCP instance
mcp: Optional[ModelContextProtocol] = None


def ensure_mcp_connection(config: Config) -> ModelContextProtocol:
    """Ensure MCP connection is established."""
    global mcp
    if mcp is None:
        try:
            db_session = get_db_session(config.db_path)
            mcp = ModelContextProtocol(config, db_session)
            logger.info("Connected to RSSidian database")
        except Exception as e:
            logger.error(f"Failed to connect to RSSidian database: {e}")
            raise Exception(f"Database connection error: {str(e)}")
    return mcp


def format_articles_response(articles: List[Dict]) -> str:
    """Format articles data for MCP response."""
    if not articles:
        return "No articles found."

    result = f"Found {len(articles)} articles:\n\n"
    for i, article in enumerate(articles[:10]):  # Show first 10
        result += f"{i+1}. {article['title']}\n"
        result += f"   Feed: {article['feed_title']}\n"
        result += f"   Published: {article['published_at']}\n"
        result += f"   Quality: {article['quality_tier']} (score: {article['quality_score']})\n"
        if article.get('summary'):
            result += f"   Summary: {article['summary'][:200]}{'...' if len(article['summary']) > 200 else ''}\n"
        result += f"   URL: {article['url']}\n\n"

    if len(articles) > 10:
        result += f"... and {len(articles) - 10} more articles\n"

    return result


def format_subscriptions_response(subscriptions: List[Dict]) -> str:
    """Format subscriptions data for MCP response."""
    if not subscriptions:
        return "No feed subscriptions found."

    result = f"Found {len(subscriptions)} feed subscriptions:\n\n"
    for i, feed in enumerate(subscriptions):
        result += f"{i+1}. {feed['title']}\n"
        result += f"   URL: {feed['url']}\n"
        if feed.get('description'):
            result += f"   Description: {feed['description'][:150]}{'...' if len(feed['description']) > 150 else ''}\n"
        result += f"   Articles: {feed['article_count']}\n"
        result += f"   Last Updated: {feed['last_updated']}\n"
        result += f"   Muted: {feed['muted']}\n\n"

    return result


def format_search_results(results: List[Dict]) -> str:
    """Format search results for MCP response."""
    if not results:
        return "No search results found."

    result = f"Found {len(results)} search results:\n\n"
    for i, item in enumerate(results[:10]):  # Show first 10
        result += f"{i+1}. {item['title']}\n"
        result += f"   Feed: {item['feed_title']}\n"
        result += f"   Relevance: {item['relevance_score']:.2f}\n"
        result += f"   Published: {item['published_at']}\n"
        if item.get('summary'):
            result += f"   Summary: {item['summary'][:200]}{'...' if len(item['summary']) > 200 else ''}\n"
        result += f"   URL: {item['url']}\n\n"

    if len(results) > 10:
        result += f"... and {len(results) - 10} more results\n"

    return result


class MCPServer:
    """MCP Server implementation using JSON-RPC 2.0 over STDIO."""

    def __init__(self, config: Config):
        self.config = config
        self.tools = {
            "list_subscriptions": {
                "description": "Get a list of all RSS feed subscriptions with their details including title, URL, article count, and last update time.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            "list_articles": {
                "description": "Get a list of articles from RSS feeds with optional filtering by feed, date range, and quality tier. Returns article summaries with metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of articles to return",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 200
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of articles to skip for pagination",
                            "default": 0,
                            "minimum": 0
                        },
                        "feed_title": {
                            "type": "string",
                            "description": "Filter articles by specific feed title"
                        },
                        "days_back": {
                            "type": "integer",
                            "description": "Only return articles from the last N days",
                            "minimum": 1,
                            "maximum": 365
                        },
                        "min_quality_tier": {
                            "type": "string",
                            "description": "Minimum quality tier (S=highest, A, B, C, D=lowest)",
                            "enum": ["S", "A", "B", "C", "D"]
                        }
                    },
                    "additionalProperties": False
                }
            },
            "get_article_content": {
                "description": "Get the full content of a specific article by its ID, including complete text, metadata, and processing information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "article_id": {
                            "type": "integer",
                            "description": "The unique ID of the article to retrieve"
                        }
                    },
                    "required": ["article_id"],
                    "additionalProperties": False
                }
            },
            "search_articles": {
                "description": "Search for articles using semantic search with natural language queries. Returns relevant articles ranked by similarity score.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "relevance_threshold": {
                            "type": "number",
                            "description": "Minimum relevance score (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            },
            "get_digest": {
                "description": "Get a curated digest of high-value articles from a specified time period, useful for staying updated on important content.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days_back": {
                            "type": "integer",
                            "description": "Number of days to look back for the digest",
                            "default": 7,
                            "minimum": 1,
                            "maximum": 30
                        }
                    },
                    "additionalProperties": False
                }
            },
            "get_feed_stats": {
                "description": "Get statistics about RSS feeds, including article counts, quality distributions, and feed performance metrics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feed_title": {
                            "type": "string",
                            "description": "Optional specific feed title to get stats for. If not provided, returns stats for all feeds."
                        }
                    },
                    "additionalProperties": False
                }
            }
        }

        self.resources = {
            "rssidian://subscriptions": {
                "name": "RSS Feed Subscriptions",
                "description": "Access to RSS feed subscription data",
                "mimeType": "application/json"
            },
            "rssidian://articles": {
                "name": "RSS Articles",
                "description": "Access to RSS article data",
                "mimeType": "application/json"
            },
            "rssidian://stats": {
                "name": "Feed Statistics",
                "description": "Access to feed statistics and metrics",
                "mimeType": "application/json"
            }
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC 2.0 request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            logger.info(f"Handling request: {method}")

            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {}
                        },
                        "serverInfo": {
                            "name": "rssidian-mcp",
                            "version": "1.0.0"
                        }
                    }
                }

            elif method == "notifications/initialized":
                # No response needed for notification
                return {}

            elif method == "tools/list":
                tools_list = []
                for name, tool_def in self.tools.items():
                    tools_list.append({
                        "name": name,
                        "description": tool_def["description"],
                        "inputSchema": tool_def["inputSchema"]
                    })

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools_list
                    }
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                result = await self.call_tool(tool_name, arguments)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }

            elif method == "resources/list":
                resources_list = []
                for uri, resource_def in self.resources.items():
                    resources_list.append({
                        "uri": uri,
                        "name": resource_def["name"],
                        "description": resource_def["description"],
                        "mimeType": resource_def["mimeType"]
                    })

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": resources_list
                    }
                }

            elif method == "resources/read":
                uri = params.get("uri")
                result = await self.get_resource(uri)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": result
                            }
                        ]
                    }
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Handle tool calls."""
        mcp_instance = ensure_mcp_connection(self.config)

        if name == "list_subscriptions":
            subscriptions = mcp_instance.get_subscriptions()
            return format_subscriptions_response(subscriptions)

        elif name == "list_articles":
            limit = arguments.get("limit", 50)
            offset = arguments.get("offset", 0)
            feed_title = arguments.get("feed_title")
            days_back = arguments.get("days_back")
            min_quality_tier = arguments.get("min_quality_tier")

            articles = mcp_instance.get_articles(
                limit=limit,
                offset=offset,
                feed_title=feed_title,
                days_back=days_back,
                min_quality_tier=min_quality_tier
            )
            return format_articles_response(articles)

        elif name == "get_article_content":
            article_id = arguments["article_id"]
            content = mcp_instance.get_article_content(article_id)

            if "error" in content:
                return f"Error: {content['error']}"

            result = f"Article: {content['title']}\n\n"
            result += f"Feed: {content['feed_title']}\n"
            result += f"Author: {content.get('author', 'Unknown')}\n"
            result += f"Published: {content['published_at']}\n"
            result += f"URL: {content['url']}\n"
            result += f"Quality: {content['quality_tier']} (score: {content['quality_score']})\n"
            if content.get('labels'):
                result += f"Labels: {', '.join(content['labels'])}\n"
            result += f"Word Count: {content['word_count']}\n\n"

            if content.get('summary'):
                result += f"Summary:\n{content['summary']}\n\n"

            if content.get('content'):
                result += f"Full Content:\n{content['content']}\n"
            elif content.get('description'):
                result += f"Description:\n{content['description']}\n"

            return result

        elif name == "search_articles":
            query = arguments["query"]
            relevance_threshold = arguments.get("relevance_threshold", 0.3)
            max_results = arguments.get("max_results", 20)

            results = mcp_instance.search_articles(
                query=query,
                relevance_threshold=relevance_threshold,
                max_results=max_results
            )
            return format_search_results(results)

        elif name == "get_digest":
            days_back = arguments.get("days_back", 7)
            digest = mcp_instance.get_digest(days_back=days_back)

            if isinstance(digest, dict) and "error" in digest:
                return f"Error: {digest['error']}"

            # Format digest response
            if isinstance(digest, str):
                return digest
            elif isinstance(digest, dict):
                result = f"Digest for the last {days_back} days:\n\n"
                if "summary" in digest:
                    result += f"Summary: {digest['summary']}\n\n"
                if "articles" in digest:
                    result += f"Featured Articles ({len(digest['articles'])}):\n\n"
                    for i, article in enumerate(digest['articles'][:10]):
                        result += f"{i+1}. {article.get('title', 'Untitled')}\n"
                        if article.get('summary'):
                            result += f"   {article['summary'][:150]}{'...' if len(article['summary']) > 150 else ''}\n"
                        result += f"   Quality: {article.get('quality_tier', 'N/A')}\n\n"
                return result
            else:
                return str(digest)

        elif name == "get_feed_stats":
            feed_title = arguments.get("feed_title")
            stats = mcp_instance.get_feed_stats(feed_title=feed_title)

            if isinstance(stats, dict) and "error" in stats:
                return f"Error: {stats['error']}"

            if feed_title:
                # Single feed stats
                result = f"Statistics for feed: {stats['title']}\n\n"
                result += f"Total Articles: {stats['article_count']}\n"
                result += f"Average Quality Score: {stats.get('avg_quality_score', 'N/A')}\n"
                result += f"Last Updated: {stats['last_updated']}\n\n"

                if stats.get('quality_tier_counts'):
                    result += "Quality Distribution:\n"
                    for tier, count in stats['quality_tier_counts'].items():
                        result += f"  {tier}: {count} articles\n"
            else:
                # All feeds stats
                result = f"Overall Statistics:\n\n"
                result += f"Total Feeds: {stats['total_feeds']}\n"
                result += f"Total Articles: {stats['total_articles']}\n\n"

                result += "Feed Details:\n"
                for feed in stats['feeds'][:20]:  # Show first 20 feeds
                    result += f"- {feed['title']}: {feed['article_count']} articles"
                    if feed.get('avg_quality_score'):
                        result += f" (avg quality: {feed['avg_quality_score']:.2f})"
                    result += "\n"

                if len(stats['feeds']) > 20:
                    result += f"... and {len(stats['feeds']) - 20} more feeds\n"

            return result

        else:
            raise ValueError(f"Unknown tool: {name}")

    async def get_resource(self, uri: str) -> str:
        """Get a resource by URI."""
        mcp_instance = ensure_mcp_connection(self.config)

        if uri == "rssidian://subscriptions":
            subscriptions = mcp_instance.get_subscriptions()
            return json.dumps(subscriptions, indent=2)

        elif uri == "rssidian://articles":
            # Get recent articles (last 7 days)
            articles = mcp_instance.get_articles(days_back=7, limit=100)
            return json.dumps(articles, indent=2)

        elif uri == "rssidian://stats":
            stats = mcp_instance.get_feed_stats()
            return json.dumps(stats, indent=2)

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP STDIO server")

        # Initialize MCP connection
        ensure_mcp_connection(self.config)

        try:
            while True:
                # Read JSON-RPC message from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                    logger.debug(f"Received request: {request}")

                    response = await self.handle_request(request)

                    if response is not None:
                        response_json = json.dumps(response)
                        print(response_json)
                        sys.stdout.flush()
                        logger.debug(f"Sent response: {response_json}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()

        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("MCP STDIO server stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP STDIO server for RSSidian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log", metavar="FILE", default="/tmp/rssidian_mcp_stdio.log",
                       help="Path to log file for debug outputs")
    parser.add_argument("--config", metavar="FILE", default="config.toml",
                       help="Path to configuration file")

    args = parser.parse_args()

    # Configure logging with the specified log file
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log)],
        force=True  # Override any existing configuration
    )

    # Set log level for our logger
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("MCP server starting")

    # Load configuration
    try:
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create and run server
    server = MCPServer(config)
    await server.run()


def run_stdio_server(config: Config):
    """Run the STDIO server (legacy function for compatibility)."""
    # Create a simple wrapper to run the async server
    async def run_async():
        server = MCPServer(config)
        await server.run()

    asyncio.run(run_async())


if __name__ == "__main__":
    asyncio.run(main())
