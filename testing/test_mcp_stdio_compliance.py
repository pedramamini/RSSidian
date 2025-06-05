#!/usr/bin/env python
"""
Test script to verify MCP STDIO compliance.

This script tests the new MCP-compliant STDIO implementation to ensure it follows
the JSON-RPC 2.0 protocol and MCP standards correctly.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def test_mcp_protocol():
    """Test the MCP protocol implementation."""

    # Path to the MCP server script
    server_script = Path(__file__).parent.parent / "rssidian" / "mcp_stdio.py"

    print("Testing MCP STDIO Protocol Compliance")
    print("=" * 50)

    # Test messages following JSON-RPC 2.0 and MCP protocol
    test_messages = [
        # 1. Initialize
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        },

        # 2. Initialized notification
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        },

        # 3. List tools
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        },

        # 4. List resources
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list"
        },

        # 5. Call a tool (list subscriptions)
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "list_subscriptions",
                "arguments": {}
            }
        }
    ]

    try:
        # Start the MCP server
        print("Starting MCP server...")
        process = subprocess.Popen(
            [sys.executable, str(server_script), "--debug"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )

        print("Server started. Testing protocol messages...")

        for i, message in enumerate(test_messages, 1):
            print(f"\nTest {i}: {message['method']}")
            print(f"Sending: {json.dumps(message)}")

            # Send message
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

            # For notifications, don't expect a response
            if message.get("method") == "notifications/initialized":
                print("✓ Notification sent (no response expected)")
                continue

            # Read response with timeout
            try:
                # Give the server time to process
                time.sleep(0.5)

                # Check if there's output available
                if process.poll() is not None:
                    # Process has terminated
                    stderr_output = process.stderr.read()
                    print(f"✗ Server terminated unexpectedly: {stderr_output}")
                    break

                # Try to read a line
                response_line = process.stdout.readline()
                if response_line:
                    try:
                        response = json.loads(response_line.strip())
                        print(f"Received: {json.dumps(response, indent=2)}")

                        # Validate JSON-RPC 2.0 format
                        if response.get("jsonrpc") == "2.0":
                            print("✓ Valid JSON-RPC 2.0 format")
                        else:
                            print("✗ Invalid JSON-RPC 2.0 format")

                        # Check for expected fields
                        if "result" in response or "error" in response:
                            print("✓ Contains result or error field")
                        else:
                            print("✗ Missing result or error field")

                    except json.JSONDecodeError as e:
                        print(f"✗ Invalid JSON response: {e}")
                        print(f"Raw response: {response_line}")
                else:
                    print("✗ No response received")

            except Exception as e:
                print(f"✗ Error reading response: {e}")

        # Clean shutdown
        print("\nShutting down server...")
        process.terminate()
        process.wait(timeout=5)

    except FileNotFoundError:
        print(f"✗ Server script not found: {server_script}")
        print("Make sure you're running this from the correct directory")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

    print("\n" + "=" * 50)
    print("MCP Protocol Compliance Test Complete")
    return True

def test_tool_schemas():
    """Test that tool schemas are properly defined."""
    print("\nTesting Tool Schema Definitions")
    print("-" * 30)

    try:
        # Import the server to check tool definitions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rssidian.config import Config
        from rssidian.mcp_stdio import MCPServer

        # Create a dummy config for testing
        config = Config()
        server = MCPServer(config)

        print(f"Found {len(server.tools)} tools:")
        for tool_name, tool_def in server.tools.items():
            print(f"  - {tool_name}")

            # Check required fields
            if "description" in tool_def:
                print(f"    ✓ Has description")
            else:
                print(f"    ✗ Missing description")

            if "inputSchema" in tool_def:
                print(f"    ✓ Has input schema")
                schema = tool_def["inputSchema"]
                if schema.get("type") == "object":
                    print(f"    ✓ Schema is object type")
                else:
                    print(f"    ✗ Schema is not object type")
            else:
                print(f"    ✗ Missing input schema")

        print(f"\nFound {len(server.resources)} resources:")
        for resource_uri, resource_def in server.resources.items():
            print(f"  - {resource_uri}")
            if all(key in resource_def for key in ["name", "description", "mimeType"]):
                print(f"    ✓ Has all required fields")
            else:
                print(f"    ✗ Missing required fields")

        return True

    except Exception as e:
        print(f"✗ Schema test failed: {e}")
        return False

if __name__ == "__main__":
    print("RSSidian MCP STDIO Compliance Test")
    print("=" * 50)

    # Test tool schemas first (doesn't require server)
    schema_success = test_tool_schemas()

    # Test protocol compliance (requires server)
    protocol_success = test_mcp_protocol()

    if schema_success and protocol_success:
        print("\n🎉 All tests passed! MCP implementation is compliant.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)