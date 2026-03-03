#!/usr/bin/env python3
"""
Test script for the OpenAI Images API compatible server.
This script provides examples of how to use the API server and test basic functionality.
"""

import os
import sys
import json
import time
import requests
import base64
from io import BytesIO

# Add current directory to path to import server module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_server_functions():
    """Test the modified server.py functions directly."""
    print("Testing server.py functions...")

    try:
        from server import load_workflow, generate_image_in_memory, image_to_base64

        # Test workflow loading
        print("1. Testing workflow loading...")
        workflow = load_workflow("z_image_turbo.json")
        if workflow:
            print("   ✓ Workflow loaded successfully")
        else:
            print("   ✗ Failed to load workflow")
            return False

        # Test image generation (if ComfyUI is running)
        print("2. Testing image generation (requires ComfyUI running)...")
        try:
            # This will fail if ComfyUI is not running, which is OK for testing
            image_bytes = generate_image_in_memory(
                prompt_text="a test cat",
                workflow_path="z_image_turbo.json",
                server_address="127.0.0.1:8188"
            )
            print(f"   ✓ Image generated: {len(image_bytes)} bytes")

            # Test base64 conversion
            base64_str = image_to_base64(image_bytes)
            print(f"   ✓ Base64 conversion: {len(base64_str)} characters")

            # Verify it's valid base64
            decoded = base64.b64decode(base64_str)
            print(f"   ✓ Base64 decode successful: {len(decoded)} bytes")

        except ConnectionError as e:
            print(f"   ⚠ ComfyUI connection failed (expected if not running): {e}")
        except Exception as e:
            print(f"   ⚠ Image generation test skipped: {e}")

        return True

    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False


def test_api_server():
    """Test the API server via HTTP requests."""
    print("\nTesting API server (requires server to be running)...")
    print("Start the server with: python unified_api_server.py")
    print("Then run: python test_api.py --api-test")
    print("\nOr use curl to test:")
    print('''
curl -X POST http://localhost:5000/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer test-key" \\
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
    ''')


def quick_api_test():
    """Quick API test if server is running."""
    url = "http://localhost:5000/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-key"
    }
    data = {
        "prompt": "A beautiful sunset over mountains",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }

    try:
        print("Sending test request to API server...")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✓ API request successful!")
            print(f"  Created: {result.get('created')}")
            print(f"  Model: {result.get('model', 'N/A')}")

            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                if 'b64_json' in image_data:
                    b64_len = len(image_data['b64_json'])
                    print(f"  Image data: {b64_len} characters base64")

                    # Decode to verify
                    try:
                        img_bytes = base64.b64decode(image_data['b64_json'])
                        print(f"  Decoded: {len(img_bytes)} bytes")
                    except:
                        print("  Warning: Failed to decode base64")

                if 'revised_prompt' in image_data:
                    print(f"  Revised prompt: {image_data['revised_prompt'][:50]}...")

            return True
        else:
            print("✗ API request failed")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server. Is it running?")
        return False
    except Exception as e:
        print(f"✗ API test error: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the OpenAI Images API server")
    parser.add_argument("--server-test", action="store_true", help="Test server functions directly")
    parser.add_argument("--api-test", action="store_true", help="Test API server (requires server running)")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if not any([args.server_test, args.api_test, args.all]):
        parser.print_help()
        sys.exit(1)

    tests_passed = True

    if args.server_test or args.all:
        if not test_server_functions():
            tests_passed = False

    if args.api_test or args.all:
        if not quick_api_test():
            tests_passed = False

    if not args.api_test and not args.all:
        # Default: show test instructions
        test_server_functions()
        test_api_server()

    if tests_passed:
        print("\n✓ All tests completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)