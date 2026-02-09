#!/usr/bin/env python3
"""
Test script for OpenAI client compatibility with our API server.
This script demonstrates how to use the official OpenAI Python SDK
to interact with our ComfyUI-based image generation API server.

Usage:
1. Start the API server: python api_server.py
2. Run this test: python test_openai_client.py
"""

import os
import sys
import time
import base64
from io import BytesIO
from PIL import Image

# Try to import OpenAI SDK
try:
    import openai
except ImportError:
    print("Error: OpenAI Python SDK not installed.")
    print("Install it with: pip install openai")
    sys.exit(1)

def test_openai_client():
    """Test the API server using the official OpenAI client."""
    print("Testing OpenAI client compatibility...")
    print("=" * 60)

    # Configuration
    api_base = "http://localhost:5000/v1"  # Our server's base URL
    api_key = "test-key-123"  # Any non-empty string works
    model = "dall-e-3"  # Model name (will be echoed by server)

    print(f"API Base URL: {api_base}")
    print(f"API Key: {api_key}")
    print(f"Model: {model}")
    print()

    # Create OpenAI client
    print("1. Creating OpenAI client...")
    try:
        client = openai.Client(api_key=api_key, base_url=api_base)
        print("   ✓ Client created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create client: {e}")
        return False

    # Test image generation
    print("\n2. Testing image generation...")
    prompt = "a red apple on a wooden table"
    print(f"   Prompt: '{prompt}'")

    try:
        start_time = time.time()

        # Call the images.generate endpoint
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            quality="standard",
            style="vivid"
        )

        elapsed = time.time() - start_time
        print(f"   ✓ Image generated in {elapsed:.2f} seconds")

        # Check response structure
        print("\n3. Validating response structure...")

        # Check required fields
        required_fields = ["created", "data"]
        for field in required_fields:
            if hasattr(response, field):
                print(f"   ✓ Response has '{field}' field")
            else:
                print(f"   ✗ Missing '{field}' field")
                return False

        # Check data array
        if not hasattr(response, 'data') or not isinstance(response.data, list):
            print("   ✗ 'data' field is not a list")
            return False

        print(f"   ✓ Response contains {len(response.data)} image(s)")

        if len(response.data) > 0:
            image_data = response.data[0]

            # Check for b64_json or url
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                print("   ✓ Image data available in b64_json format")

                # Decode and verify the image
                try:
                    image_bytes = base64.b64decode(image_data.b64_json)
                    print(f"   ✓ Decoded image: {len(image_bytes)} bytes")

                    # Try to open as PIL image
                    image = Image.open(BytesIO(image_bytes))
                    print(f"   ✓ Valid image: {image.format}, size: {image.size}")

                    # Save for inspection
                    output_path = "test_output_openai.png"
                    image.save(output_path)
                    print(f"   ✓ Saved to: {output_path}")

                except Exception as e:
                    print(f"   ⚠ Could not decode/verify image: {e}")

            elif hasattr(image_data, 'url') and image_data.url:
                print("   ✓ Image available via URL")
                print(f"   URL: {image_data.url}")
            else:
                print("   ✗ No image data in response")
                return False

            # Check for optional revised_prompt
            if hasattr(image_data, 'revised_prompt'):
                print(f"   ✓ Revised prompt: '{image_data.revised_prompt[:50]}...'")

        # Check model field
        if hasattr(response, 'model'):
            print(f"   ✓ Model: {response.model}")
        else:
            print("   ⚠ No model field in response (optional)")

        print("\n4. Testing error handling...")

        # Test with empty prompt (should fail)
        try:
            response = client.images.generate(
                model=model,
                prompt="",  # Empty prompt should trigger validation error
                n=1
            )
            print("   ✗ Empty prompt should have failed but didn't")
            return False
        except Exception as e:
            print(f"   ✓ Empty prompt correctly rejected: {type(e).__name__}")

        # Test with invalid model (should work since we ignore model)
        try:
            response = client.images.generate(
                model="invalid-model-name",  # Should be accepted but ignored
                prompt="a test image",
                n=1
            )
            print("   ✓ Invalid model name accepted (ignored by server)")
        except Exception as e:
            print(f"   ⚠ Invalid model name rejected: {e}")

        print("\n" + "=" * 60)
        print("SUCCESS: OpenAI client compatibility test passed!")
        print("\nSummary:")
        print(f"- Server correctly handles OpenAI SDK requests")
        print(f"- Images are generated via ComfyUI")
        print(f"- Response format matches OpenAI API specification")
        print(f"- Error handling works correctly")

        return True

    except openai.APIError as e:
        print(f"   ✗ OpenAI API error: {e}")
        print(f"   Error type: {e.type if hasattr(e, 'type') else 'N/A'}")
        print(f"   Error code: {e.code if hasattr(e, 'code') else 'N/A'}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_api_call():
    """Test the API directly with requests for comparison."""
    print("\n" + "=" * 60)
    print("Additional test: Direct API call comparison")

    import requests

    url = "http://localhost:5000/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-key"
    }
    data = {
        "prompt": "a simple test",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Direct API call status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            print("✓ Direct API call successful")
        else:
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Direct API call failed: {e}")


if __name__ == "__main__":
    print("OpenAI Client Compatibility Test")
    print("=" * 60)
    print("Note: Make sure the API server is running:")
    print("  python api_server.py")
    print("Or with custom port:")
    print("  API_PORT=9997 python api_server.py")
    print("Then update api_base in this script if needed.")
    print()

    # Check if server is reachable
    try:
        import requests
        health_check = requests.get("http://localhost:5000/health", timeout=5)
        if health_check.status_code == 200:
            print("✓ API server is running and healthy")
        else:
            print(f"⚠ API server returned status {health_check.status_code}")
            print("  Starting server automatically is not supported.")
            print("  Please start the server manually and try again.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server at http://localhost:5000")
        print("  Please start the server with: python api_server.py")
        print("  Or if using a different port, update the api_base variable.")
        sys.exit(1)

    # Run the main test
    success = test_openai_client()

    # Optional: run direct API comparison
    if success:
        test_direct_api_call()

    # Exit with appropriate code
    sys.exit(0 if success else 1)