"""
STDIO interface for the Model Context Protocol.

This module provides a STDIO-based interface for the Model Context Protocol,
allowing AI models to interact with RSSidian through standard input/output
rather than HTTP requests.
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional, Callable
import traceback

from .config import Config
from .models import get_db_session
from .mcp import ModelContextProtocol

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   filename='/tmp/rssidian_mcp_stdio.log')
logger = logging.getLogger(__name__)

class MCPStdioServer:
    """
    STDIO server for the Model Context Protocol.
    
    This class handles communication with Claude Desktop through standard input/output,
    parsing JSON requests and returning JSON responses.
    """
    
    def __init__(self, config: Config):
        """Initialize the STDIO server with configuration."""
        self.config = config
        self.db_session = get_db_session(config.db_path)
        self.mcp = ModelContextProtocol(config, self.db_session)
        self.handlers = self._build_handlers()
        
    def _build_handlers(self) -> Dict[str, Callable]:
        """Build a dictionary of request handlers."""
        return {
            "list_subscriptions": self._handle_list_subscriptions,
            "list_articles": self._handle_list_articles,
            "get_article_content": self._handle_get_article_content,
            "search_articles": self._handle_search_articles,
            "get_digest": self._handle_get_digest,
            "get_feed_stats": self._handle_get_feed_stats,
            "query": self._handle_query,
        }
    
    def _handle_list_subscriptions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to list subscriptions."""
        subscriptions = self.mcp.get_subscriptions()
        return {"subscriptions": subscriptions}
    
    def _handle_list_articles(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to list articles."""
        feed_id = params.get("feed_id")
        days = params.get("days", 7)
        limit = params.get("limit", 50)
        include_content = params.get("include_content", False)
        min_quality = params.get("min_quality")
        
        articles = self.mcp.get_articles(
            feed_id=feed_id, 
            days=days, 
            limit=limit, 
            include_content=include_content,
            min_quality=min_quality
        )
        return {"articles": articles}
    
    def _handle_get_article_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get article content."""
        article_id = params.get("article_id")
        if not article_id:
            return {"error": "Missing required parameter: article_id"}
        
        content = self.mcp.get_article_content(article_id)
        return {"content": content}
    
    def _handle_search_articles(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to search articles."""
        query = params.get("query")
        if not query:
            return {"error": "Missing required parameter: query"}
        
        relevance = params.get("relevance", 70)
        days = params.get("days", 30)
        limit = params.get("limit", 20)
        
        results = self.mcp.search_articles(
            query=query,
            relevance=relevance,
            days=days,
            limit=limit
        )
        return {"results": results}
    
    def _handle_get_digest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get a digest."""
        days = params.get("days", 7)
        min_quality = params.get("min_quality")
        
        digest = self.mcp.get_digest(days=days, min_quality=min_quality)
        return {"digest": digest}
    
    def _handle_get_feed_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to get feed statistics."""
        days = params.get("days", 30)
        
        stats = self.mcp.get_feed_stats(days=days)
        return {"stats": stats}
    
    def _handle_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a natural language query."""
        query = params.get("query")
        if not query:
            return {"error": "Missing required parameter: query"}
        
        response = self.mcp.process_query(query)
        return {"response": response}
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request from Claude Desktop.
        
        Args:
            request: The request object from Claude Desktop
            
        Returns:
            A response object to send back to Claude Desktop
        """
        try:
            action = request.get("action")
            params = request.get("params", {})
            
            if not action:
                return {"error": "Missing required field: action"}
            
            handler = self.handlers.get(action)
            if not handler:
                return {"error": f"Unknown action: {action}"}
            
            return handler(params)
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run(self):
        """Run the STDIO server, processing requests from stdin and writing responses to stdout."""
        logger.info("Starting RSSidian MCP STDIO server")
        print("RSSidian MCP Server running on stdio", file=sys.stderr)
        
        # Print capabilities for Claude Desktop
        capabilities = {
            "prompts": {},
            "resources": {},
            "tools": {
                "list_subscriptions": {
                    "description": "List all RSS feed subscriptions",
                    "parameters": {}
                },
                "list_articles": {
                    "description": "List articles from feeds with optional filtering",
                    "parameters": {
                        "feed_id": {"type": "string", "description": "Optional feed ID to filter by"},
                        "days": {"type": "integer", "description": "Number of days to look back"},
                        "limit": {"type": "integer", "description": "Maximum number of articles to return"},
                        "include_content": {"type": "boolean", "description": "Whether to include full content"},
                        "min_quality": {"type": "string", "description": "Minimum quality tier (S, A, B, C, D)"}
                    }
                },
                "get_article_content": {
                    "description": "Get the full content of an article",
                    "parameters": {
                        "article_id": {"type": "string", "description": "ID of the article"}
                    }
                },
                "search_articles": {
                    "description": "Search articles using semantic search",
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "relevance": {"type": "integer", "description": "Relevance threshold (0-100)"},
                        "days": {"type": "integer", "description": "Number of days to look back"},
                        "limit": {"type": "integer", "description": "Maximum number of results"}
                    }
                },
                "get_digest": {
                    "description": "Get a digest of high-value articles",
                    "parameters": {
                        "days": {"type": "integer", "description": "Number of days to look back"},
                        "min_quality": {"type": "string", "description": "Minimum quality tier (S, A, B, C, D)"}
                    }
                },
                "get_feed_stats": {
                    "description": "Get statistics about feeds",
                    "parameters": {
                        "days": {"type": "integer", "description": "Number of days to look back"}
                    }
                },
                "query": {
                    "description": "Process a natural language query about RSS content",
                    "parameters": {
                        "query": {"type": "string", "description": "Natural language query"}
                    }
                }
            }
        }
        
        print(json.dumps(capabilities), file=sys.stderr)
        
        try:
            while True:
                # Read a line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                # Parse the request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    response = {"error": "Invalid JSON"}
                    print(json.dumps(response))
                    continue
                
                # Handle the request
                response = self.handle_request(request)
                
                # Write the response to stdout
                print(json.dumps(response))
                sys.stdout.flush()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Shutting down RSSidian MCP STDIO server")


def run_stdio_server(config: Config):
    """Run the STDIO server."""
    server = MCPStdioServer(config)
    server.run()
