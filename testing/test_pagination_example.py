#!/usr/bin/env python
"""
Example demonstrating pagination with the list_articles tool.
"""

import json

def demonstrate_pagination():
    """Show how to use pagination with the list_articles tool."""

    print("RSSidian MCP Pagination Examples")
    print("=" * 40)

    # Example 1: Get first 10 articles
    request1 = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "list_articles",
            "arguments": {
                "limit": 10,
                "offset": 0
            }
        }
    }

    print("1. Get first 10 articles:")
    print(json.dumps(request1, indent=2))
    print()

    # Example 2: Get next 10 articles (pagination)
    request2 = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "list_articles",
            "arguments": {
                "limit": 10,
                "offset": 10
            }
        }
    }

    print("2. Get next 10 articles (offset=10):")
    print(json.dumps(request2, indent=2))
    print()

    # Example 3: Get 50 articles from a specific feed
    request3 = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "list_articles",
            "arguments": {
                "limit": 50,
                "offset": 0,
                "feed_title": "TechCrunch",
                "days_back": 7
            }
        }
    }

    print("3. Get 50 articles from TechCrunch in last 7 days:")
    print(json.dumps(request3, indent=2))
    print()

    # Example 4: Get high-quality articles with pagination
    request4 = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "list_articles",
            "arguments": {
                "limit": 25,
                "offset": 50,
                "min_quality_tier": "A",
                "days_back": 14
            }
        }
    }

    print("4. Get high-quality articles (tier A+) with pagination:")
    print(json.dumps(request4, indent=2))
    print()

    print("Key Pagination Features:")
    print("=" * 25)
    print("✓ limit: Maximum articles to return (1-200)")
    print("✓ offset: Number of articles to skip (for pagination)")
    print("✓ Response includes pagination guidance")
    print("✓ Article IDs included for easy reference")
    print("✓ Works with all filtering options")
    print()

    print("Pagination Strategy:")
    print("- Start with offset=0, limit=50")
    print("- If you get 50 results, there may be more")
    print("- Use offset=50, limit=50 for next page")
    print("- Continue until you get fewer than 'limit' results")
    print()

    print("Response Format:")
    print("- Shows 'offset: X, limit: Y' in header")
    print("- Articles numbered from offset+1")
    print("- Includes pagination guidance at bottom")
    print("- Article IDs provided for get_article_content calls")

if __name__ == "__main__":
    demonstrate_pagination()