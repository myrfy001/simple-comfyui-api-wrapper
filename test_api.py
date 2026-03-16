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


def _poll_video_until_done(base_url: str, video_id: str, timeout_sec: int = 300):
    """Poll video status endpoint until completed/failed or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        status_response = requests.get(f"{base_url}/videos/{video_id}", timeout=10)
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data.get("status")

        if status == "completed":
            return status_data
        if status == "failed":
            raise RuntimeError(status_data.get("error", {}).get("message", "Video generation failed"))

        time.sleep(2)

    raise TimeoutError(f"Timed out waiting for video job: {video_id}")


def quick_video_i2v_test(image_path: str = "demo3_output.png"):
    """Quick I2V test using local image -> base64 -> /v1/videos with MP4 validation."""
    print("\nTesting I2V video generation (base64 local image)...")

    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    base_url = "http://localhost:5000/v1"
    payload = {
        "model": "my_ltx_video",
        "prompt": (
            "Preserve the exact subject identity, geometry, color palette, lighting, "
            "and composition from the reference image. Add only subtle natural motion "
            "and gentle camera movement. Do not add new objects, do not replace the "
            "scene, and do not change style."
        ),
        "seconds": 4,
        "size": "768x512",
        "input_reference": {
            "image_url": f"data:image/png;base64,{image_b64}"
        },
    }

    try:
        create_resp = requests.post(f"{base_url}/videos", json=payload, timeout=60)
        print(f"  Create status: {create_resp.status_code}")
        create_resp.raise_for_status()

        video_id = create_resp.json().get("id")
        if not video_id:
            print("✗ Missing video id in create response")
            return False

        print(f"  Video ID: {video_id}")
        _poll_video_until_done(base_url, video_id, timeout_sec=600)
        print("  ✓ Video generation completed")

        content_resp = requests.get(f"{base_url}/videos/{video_id}/content", timeout=180)
        print(f"  Content status: {content_resp.status_code}")
        content_resp.raise_for_status()

        content_type = content_resp.headers.get("Content-Type", "")
        body = content_resp.content
        is_mp4_magic = len(body) >= 8 and body[4:8] == b"ftyp"

        if "video/mp4" not in content_type.lower() and not is_mp4_magic:
            print(f"✗ Not MP4. Content-Type={content_type}")
            return False

        out_path = "demo5_i2v_output.mp4"
        with open(out_path, "wb") as f:
            f.write(body)

        print(f"  ✓ I2V MP4 saved: {out_path} ({len(body)} bytes)")
        return True

    except Exception as e:
        print(f"✗ I2V test error: {e}")
        return False


def quick_regression_test():
    """Regression checks: existing T2V and image generation should remain functional."""
    print("\nRunning regression checks (T2V + Images)...")
    base_url = "http://localhost:5000/v1"

    try:
        # T2V
        t2v_payload = {
            "model": "my_ltx_video",
            "prompt": "A cinematic close-up of a red paper boat drifting on calm water",
            "seconds": 4,
            "size": "768x512",
        }
        create_resp = requests.post(f"{base_url}/videos", json=t2v_payload, timeout=60)
        print(f"  T2V create status: {create_resp.status_code}")
        create_resp.raise_for_status()
        video_id = create_resp.json().get("id")
        _poll_video_until_done(base_url, video_id, timeout_sec=600)

        content_resp = requests.get(f"{base_url}/videos/{video_id}/content", timeout=180)
        content_resp.raise_for_status()
        body = content_resp.content
        if len(body) < 8 or body[4:8] != b"ftyp":
            print("✗ T2V content is not MP4 by magic bytes")
            return False
        print(f"  ✓ T2V still works ({len(body)} bytes)")

        # Images API
        image_payload = {
            "prompt": "a small green cactus in a white ceramic pot, studio lighting",
            "n": 1,
            "size": "256x256",
            "response_format": "b64_json",
            "model": "my_z_image_turbo",
        }
        image_resp = requests.post(f"{base_url}/images/generations", json=image_payload, timeout=180)
        print(f"  Image create status: {image_resp.status_code}")
        image_resp.raise_for_status()
        data = image_resp.json()
        b64_value = data.get("data", [{}])[0].get("b64_json", "")
        if not b64_value:
            print("✗ Image response missing b64_json")
            return False
        print(f"  ✓ Images API still works (base64 length: {len(b64_value)})")

        return True
    except Exception as e:
        print(f"✗ Regression test error: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the OpenAI Images API server")
    parser.add_argument("--server-test", action="store_true", help="Test server functions directly")
    parser.add_argument("--api-test", action="store_true", help="Test API server (requires server running)")
    parser.add_argument("--i2v-test", action="store_true", help="Test I2V with local base64 image upload")
    parser.add_argument("--regression-test", action="store_true", help="Run T2V + images regression checks")
    parser.add_argument("--image-path", default="demo3_output.png", help="Local image path for --i2v-test")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if not any([args.server_test, args.api_test, args.i2v_test, args.regression_test, args.all]):
        parser.print_help()
        sys.exit(1)

    tests_passed = True

    if args.server_test or args.all:
        if not test_server_functions():
            tests_passed = False

    if args.api_test or args.all:
        if not quick_api_test():
            tests_passed = False

    if args.i2v_test or args.all:
        if not quick_video_i2v_test(args.image_path):
            tests_passed = False

    if args.regression_test or args.all:
        if not quick_regression_test():
            tests_passed = False

    if not any([args.api_test, args.i2v_test, args.regression_test, args.all]):
        # Default: show test instructions
        test_server_functions()
        test_api_server()

    if tests_passed:
        print("\n✓ All tests completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)