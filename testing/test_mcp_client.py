#!/usr/bin/env python3
"""
Test client for RSSidian MCP (Model Context Protocol) service.

This client tests all the available MCP endpoints exposed by the RSSidian service
running on http://127.0.0.1:8080.

Usage:
    python test_mcp_client.py
"""

import requests
import json
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import time


class RSSidianMCPClient:
    """Client for testing RSSidian MCP service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """Initialize the client with the base URL of the MCP service."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request and return the JSON response."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error details: {error_detail}")
                except:
                    print(f"   Response text: {e.response.text}")
            return {"error": str(e)}

    def test_service_discovery(self) -> bool:
        """Test the MCP discovery endpoint."""
        print("🔍 Testing MCP Discovery...")

        result = self._make_request('GET', '/api/v1/mcp')

        if 'error' in result:
            return False

        print(f"✅ Service: {result.get('name', 'Unknown')}")
        print(f"   Version: {result.get('version', 'Unknown')}")
        print(f"   Description: {result.get('description', 'No description')}")
        print(f"   Capabilities: {', '.join(result.get('capabilities', []))}")
        print()

        return True

    def test_list_subscriptions(self) -> bool:
        """Test listing feed subscriptions."""
        print("📋 Testing List Subscriptions...")

        result = self._make_request('GET', '/api/v1/mcp/subscriptions')

        if 'error' in result:
            return False

        if isinstance(result, list):
            print(f"✅ Found {len(result)} subscriptions")
            for i, feed in enumerate(result[:3]):  # Show first 3
                print(f"   {i+1}. {feed.get('title', 'No title')} ({feed.get('url', 'No URL')})")
                print(f"      Muted: {feed.get('muted', False)}, Articles: {feed.get('article_count', 0)}")
            if len(result) > 3:
                print(f"   ... and {len(result) - 3} more")
        else:
            print(f"✅ Response: {result}")

        print()
        return True

    def test_list_articles(self) -> bool:
        """Test listing articles with various filters."""
        print("📰 Testing List Articles...")

        # Test basic article listing
        result = self._make_request('GET', '/api/v1/mcp/articles', params={'limit': 5})

        if 'error' in result:
            return False

        if isinstance(result, list):
            print(f"✅ Found {len(result)} articles (limited to 5)")
            for i, article in enumerate(result):
                print(f"   {i+1}. {article.get('title', 'No title')}")
                print(f"      Feed: {article.get('feed_title', 'Unknown')}")
                print(f"      Quality: {article.get('quality_tier', 'N/A')}")
                print(f"      Published: {article.get('published_at', 'Unknown')}")
        else:
            print(f"✅ Response: {result}")

        # Test with quality filter
        print("\n   Testing with quality filter (A-tier or better)...")
        result = self._make_request('GET', '/api/v1/mcp/articles', params={
            'limit': 3,
            'min_quality_tier': 'A'
        })

        if 'error' not in result and isinstance(result, list):
            print(f"   ✅ Found {len(result)} high-quality articles")

        # Test with date filter
        print("   Testing with date filter (last 7 days)...")
        result = self._make_request('GET', '/api/v1/mcp/articles', params={
            'limit': 3,
            'days_back': 7
        })

        if 'error' not in result and isinstance(result, list):
            print(f"   ✅ Found {len(result)} recent articles")

        print()
        return True

    def test_get_article_content(self) -> bool:
        """Test getting full article content."""
        print("📄 Testing Get Article Content...")

        # First get an article ID
        articles = self._make_request('GET', '/api/v1/mcp/articles', params={'limit': 1})

        if 'error' in articles or not isinstance(articles, list) or len(articles) == 0:
            print("❌ No articles available to test content retrieval")
            return False

        article_id = articles[0]['id']
        result = self._make_request('GET', f'/api/v1/mcp/articles/{article_id}/content')

        if 'error' in result:
            return False

        print(f"✅ Retrieved content for article: {result.get('title', 'No title')}")
        print(f"   Author: {result.get('author', 'Unknown')}")
        print(f"   Word count: {result.get('word_count', 'Unknown')}")
        print(f"   Content length: {len(result.get('content', '')) if result.get('content') else 0} characters")
        print(f"   Summary: {result.get('summary', 'No summary')[:100]}...")

        print()
        return True

    def test_search_articles(self) -> bool:
        """Test semantic search functionality."""
        print("🔍 Testing Article Search...")

        search_queries = [
            "artificial intelligence",
            "machine learning",
            "technology trends",
            "programming"
        ]

        for query in search_queries:
            print(f"   Searching for: '{query}'")
            result = self._make_request('GET', '/api/v1/mcp/search', params={
                'query': query,
                'relevance': 0.3,
                'max_results': 3
            })

            if 'error' in result:
                print(f"   ❌ Search failed for '{query}'")
                continue

            if isinstance(result, list):
                print(f"   ✅ Found {len(result)} results")
                for i, article in enumerate(result):
                    relevance = article.get('relevance', 0) * 100
                    print(f"      {i+1}. {article.get('title', 'No title')} (Relevance: {relevance:.1f}%)")
            else:
                print(f"   ✅ Search response: {result}")

            time.sleep(0.5)  # Small delay between searches

        print()
        return True

    def test_get_digest(self) -> bool:
        """Test digest generation."""
        print("📊 Testing Digest Generation...")

        result = self._make_request('GET', '/api/v1/mcp/digest', params={'days_back': 7})

        if 'error' in result:
            return False

        print("✅ Generated digest successfully")

        # Print digest structure
        if isinstance(result, dict):
            print(f"   Period: {result.get('period', 'Unknown')}")
            print(f"   Total articles: {result.get('total_articles', 'Unknown')}")

            sections = result.get('sections', [])
            if sections:
                print(f"   Sections: {len(sections)}")
                for section in sections[:3]:  # Show first 3 sections
                    print(f"      - {section.get('title', 'No title')} ({len(section.get('articles', []))} articles)")

        print()
        return True

    def test_get_feed_stats(self) -> bool:
        """Test feed statistics."""
        print("📈 Testing Feed Statistics...")

        # Test overall stats
        result = self._make_request('GET', '/api/v1/mcp/feed-stats')

        if 'error' in result:
            return False

        print("✅ Retrieved feed statistics")
        print(f"   Total feeds: {result.get('total_feeds', 'Unknown')}")
        print(f"   Total articles: {result.get('total_articles', 'Unknown')}")

        feeds = result.get('feeds', [])
        if feeds:
            print(f"   Top feeds by article count:")
            # Sort by article count and show top 3
            sorted_feeds = sorted(feeds, key=lambda x: x.get('article_count', 0), reverse=True)
            for i, feed in enumerate(sorted_feeds[:3]):
                print(f"      {i+1}. {feed.get('title', 'No title')} ({feed.get('article_count', 0)} articles)")

        print()
        return True

    def test_natural_language_query(self) -> bool:
        """Test natural language query endpoint."""
        print("🤖 Testing Natural Language Query...")

        query_data = {
            "query": "What are the most interesting articles about artificial intelligence from the last week?",
            "context": {
                "relevance_threshold": 0.4,
                "max_results": 5,
                "days_back": 7
            }
        }

        result = self._make_request('POST', '/api/v1/mcp/query', json=query_data)

        if 'error' in result:
            return False

        print("✅ Natural language query processed successfully")
        print(f"   Query: {result.get('query', 'Unknown')}")

        search_results = result.get('search_results', [])
        print(f"   Found {len(search_results)} relevant articles")

        for i, article in enumerate(search_results[:3]):  # Show first 3
            relevance = article.get('relevance', 0) * 100
            print(f"      {i+1}. {article.get('title', 'No title')} (Relevance: {relevance:.1f}%)")

        print()
        return True

    def test_error_handling(self) -> bool:
        """Test error handling for invalid requests."""
        print("⚠️  Testing Error Handling...")

        # Test invalid article ID
        result = self._make_request('GET', '/api/v1/mcp/articles/999999/content')
        if 'error' in result:
            print("✅ Correctly handled invalid article ID")

        # Test invalid feed stats
        result = self._make_request('GET', '/api/v1/mcp/feed-stats', params={'feed_title': 'NonexistentFeed'})
        if 'error' in result:
            print("✅ Correctly handled invalid feed title")

        # Test invalid query
        result = self._make_request('POST', '/api/v1/mcp/query', json={})
        if 'error' in result:
            print("✅ Correctly handled empty query")

        print()
        return True

    def run_all_tests(self) -> bool:
        """Run all test methods."""
        print("🚀 Starting RSSidian MCP Service Tests")
        print("=" * 50)

        tests = [
            ("Service Discovery", self.test_service_discovery),
            ("List Subscriptions", self.test_list_subscriptions),
            ("List Articles", self.test_list_articles),
            ("Get Article Content", self.test_get_article_content),
            ("Search Articles", self.test_search_articles),
            ("Get Digest", self.test_get_digest),
            ("Get Feed Stats", self.test_get_feed_stats),
            ("Natural Language Query", self.test_natural_language_query),
            ("Error Handling", self.test_error_handling),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"Running: {test_name}")
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"❌ {test_name} failed")
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")

            print("-" * 30)

        print(f"\n📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed")
            return False


def main():
    """Main function to run the test client."""
    import argparse

    parser = argparse.ArgumentParser(description="Test client for RSSidian MCP service")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080",
        help="Base URL of the MCP service (default: http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "--test",
        choices=[
            "discovery", "subscriptions", "articles", "content",
            "search", "digest", "stats", "query", "errors", "all"
        ],
        default="all",
        help="Specific test to run (default: all)"
    )

    args = parser.parse_args()

    client = RSSidianMCPClient(args.url)

    # Check if service is available
    try:
        response = requests.get(f"{args.url}/api/v1/mcp", timeout=5)
        if response.status_code != 200:
            print(f"❌ MCP service not available at {args.url}")
            print("   Make sure the service is running with: rssidian mcp")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to MCP service at {args.url}")
        print("   Make sure the service is running with: rssidian mcp")
        sys.exit(1)

    # Run specific test or all tests
    if args.test == "all":
        success = client.run_all_tests()
    else:
        test_methods = {
            "discovery": client.test_service_discovery,
            "subscriptions": client.test_list_subscriptions,
            "articles": client.test_list_articles,
            "content": client.test_get_article_content,
            "search": client.test_search_articles,
            "digest": client.test_get_digest,
            "stats": client.test_get_feed_stats,
            "query": client.test_natural_language_query,
            "errors": client.test_error_handling,
        }

        test_func = test_methods.get(args.test)
        if test_func:
            success = test_func()
        else:
            print(f"❌ Unknown test: {args.test}")
            sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()