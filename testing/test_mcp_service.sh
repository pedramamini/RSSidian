#!/bin/bash

echo "🔍 Testing RSSidian MCP Service..."
echo "=================================="

# Test if service is running
echo "Checking if MCP service is accessible..."
if curl -s -f "http://127.0.0.1:8080/api/v1/mcp" > /dev/null; then
    echo "✅ MCP service is running!"

    echo ""
    echo "📋 Service Discovery:"
    curl -s "http://127.0.0.1:8080/api/v1/mcp" | python3 -m json.tool

    echo ""
    echo "🎉 Ready to run full test suite!"
    echo "   Run: python test_mcp_client.py"
else
    echo "❌ MCP service is not accessible at http://127.0.0.1:8080"
    echo "   Make sure it's running with: rssidian mcp"
    exit 1
fi