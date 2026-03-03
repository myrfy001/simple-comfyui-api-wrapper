#!/usr/bin/env python3
"""
Example usage of the ComfyUI OpenAI API wrapper.
Demonstrates how to use both image and video generation APIs.
"""

import os
import sys
import json
import time
import requests
import base64
from io import BytesIO

# Configuration
API_BASE = "http://localhost:5000/v1"
API_KEY = "test-key-123"  # Any non-empty string works


def generate_image_example():
    """Example: Generate an image using the Images API."""
    print("=== Image Generation Example ===")

    url = f"{API_BASE}/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Different model examples
    models_to_test = ["dall-e-3", "dall-e"]  # 'dall-e' is an alias for 'dall-e-3'

    for model in models_to_test:
        print(f"\nGenerating image with model: {model}")

        data = {
            "prompt": "a majestic eagle flying over snow-capped mountains at sunrise",
            "n": 1,
            "size": "512x512",  # Will be adjusted to 256x256 if not supported
            "response_format": "b64_json",
            "model": model,
            "quality": "standard",
            "style": "vivid"
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Success! Model: {result.get('model')}")

                # Save the image
                if 'data' in result and len(result['data']) > 0:
                    image_data = result['data'][0]
                    if 'b64_json' in image_data:
                        # Decode and save
                        image_bytes = base64.b64decode(image_data['b64_json'])
                        filename = f"generated_image_{model.replace('-', '_')}.png"

                        with open(filename, 'wb') as f:
                            f.write(image_bytes)

                        print(f"  Image saved as: {filename}")
                        print(f"  Size: {len(image_bytes)} bytes")

            elif response.status_code == 400:
                error = response.json()
                print(f"  ✗ Validation error: {error.get('error', {}).get('message')}")
            else:
                print(f"  ✗ Error: {response.text[:200]}...")

        except requests.exceptions.Timeout:
            print("  ⚠ Request timed out (image generation may take time)")
        except Exception as e:
            print(f"  ✗ Exception: {e}")


def generate_video_example():
    """Example: Generate a video using the Video API."""
    print("\n\n=== Video Generation Example ===")

    url = f"{API_BASE}/videos"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Video generation models
    models_to_test = ["sora-2", "sora"]  # 'sora' is an alias for 'sora-2'

    for model in models_to_test:
        print(f"\nCreating video job with model: {model}")

        data = {
            "prompt": "a tranquil river flowing through a forest, sunlight filtering through leaves",
            "model": model,
            "size": "768x512",
            "seconds": 4
        }

        try:
            # Create video job
            response = requests.post(url, headers=headers, json=data, timeout=10)
            print(f"  Create job status: {response.status_code}")

            if response.status_code == 202:  # Accepted
                result = response.json()
                video_id = result.get('id')
                print(f"  ✓ Video job created: {video_id}")
                print(f"    Initial status: {result.get('status')}")

                # Poll for status (in a real app, you'd use webhooks or async polling)
                print(f"\n  Polling for completion (will timeout after 30 seconds)...")

                status_url = f"{API_BASE}/videos/{video_id}"
                start_time = time.time()
                timeout = 30  # seconds

                while time.time() - start_time < timeout:
                    status_response = requests.get(status_url, timeout=5)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        current_status = status_data.get('status')
                        progress = status_data.get('progress', 0)

                        print(f"    Status: {current_status}, Progress: {progress}%")

                        if current_status == 'completed':
                            print(f"  ✓ Video generation completed!")

                            # Download the video
                            download_url = f"{API_BASE}/videos/{video_id}/content"
                            download_response = requests.get(download_url, timeout=10)

                            if download_response.status_code == 200:
                                video_bytes = download_response.content
                                filename = f"generated_video_{model.replace('-', '_')}.webp"

                                with open(filename, 'wb') as f:
                                    f.write(video_bytes)

                                print(f"  Video saved as: {filename}")
                                print(f"  Size: {len(video_bytes)} bytes")
                            else:
                                print(f"  ✗ Failed to download video: {download_response.status_code}")

                            break
                        elif current_status == 'failed':
                            error = status_data.get('error', {})
                            print(f"  ✗ Video generation failed: {error.get('message')}")
                            break

                    time.sleep(2)  # Poll every 2 seconds

                if time.time() - start_time >= timeout:
                    print("  ⚠ Polling timeout - video may still be processing")

                # Clean up: delete the video job
                print(f"\n  Cleaning up video job...")
                delete_url = f"{API_BASE}/videos/{video_id}"
                delete_response = requests.delete(delete_url, timeout=5)

                if delete_response.status_code == 200:
                    print(f"  ✓ Video job deleted")
                else:
                    print(f"  ⚠ Failed to delete video job: {delete_response.status_code}")

            elif response.status_code == 400:
                error = response.json()
                print(f"  ✗ Validation error: {error.get('error', {}).get('message')}")
            else:
                print(f"  ✗ Error: {response.text[:200]}...")

        except Exception as e:
            print(f"  ✗ Exception: {e}")


def list_videos_example():
    """Example: List video jobs."""
    print("\n\n=== List Videos Example ===")

    url = f"{API_BASE}/videos"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        # Get first page of videos
        response = requests.get(url, headers=headers, timeout=5)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            videos = result.get('data', [])
            has_more = result.get('has_more', False)

            print(f"  Found {len(videos)} video jobs")
            print(f"  Has more pages: {has_more}")

            for i, video in enumerate(videos[:3]):  # Show first 3
                print(f"\n  Video {i + 1}:")
                print(f"    ID: {video.get('id')}")
                print(f"    Prompt: {video.get('prompt', '')[:50]}...")
                print(f"    Status: {video.get('status')}")
                print(f"    Created: {time.ctime(video.get('created_at'))}")

            if len(videos) > 3:
                print(f"\n  ... and {len(videos) - 3} more videos")

        else:
            print(f"  ✗ Error: {response.text[:200]}...")

    except Exception as e:
        print(f"  ✗ Exception: {e}")


def check_server_info():
    """Check server information and available models."""
    print("=== Server Information ===")

    try:
        # Get server info
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"Server: {info.get('service')}")
            print(f"Version: {info.get('version')}")

            # Check available models
            if 'available_models' in info:
                models = info['available_models']
                print(f"\nAvailable Image Models: {', '.join(models.get('image_models', []))}")
                print(f"Available Video Models: {', '.join(models.get('video_models', []))}")

            # Check health
            health_response = requests.get("http://localhost:5000/health", timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"\nHealth: {health.get('status')}")
                print(f"Active Models: {health.get('models')}")
                print(f"Total Backends: {health.get('total_backends')}")

        else:
            print(f"Failed to get server info: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Is it running?")
        print("Start with: python unified_api_server.py")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def main():
    """Run all examples."""
    print("ComfyUI OpenAI API Examples")
    print("=" * 60)

    # First check if server is running
    if not check_server_info():
        print("\nPlease start the server and try again.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Running examples...")

    # Run examples
    examples = [
        ("Image Generation", generate_image_example),
        ("Video Generation", generate_video_example),
        ("List Videos", list_videos_example),
    ]

    for example_name, example_func in examples:
        print(f"\n{'='*40}")
        print(f"Running: {example_name}")
        print('='*40)
        try:
            example_func()
        except KeyboardInterrupt:
            print("\nExample interrupted by user")
            break
        except Exception as e:
            print(f"Error in {example_name}: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nNote: Actual image/video generation requires ComfyUI backends to be running.")
    print("The examples above demonstrate the API interface even if backends are not available.")


if __name__ == "__main__":
    main()