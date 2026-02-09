#!/usr/bin/env python3
"""
OpenAI Images API compatible server for ComfyUI image generation.
This server implements the /v1/images/generations endpoint as defined in the
OpenAI API documentation, using ComfyUI for actual image generation.
"""

import os
import json
import time
import base64
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional

# Import functions from our modified server.py
try:
    from server import generate_image_in_memory, image_to_base64, load_workflow
except ImportError as e:
    print(f"Error importing from server.py: {e}")
    print("Make sure server.py is in the same directory and has been modified with the required functions.")
    raise

app = Flask(__name__)

# Configuration from environment variables
COMFYUI_SERVER_ADDRESS = os.getenv("COMFYUI_SERVER_ADDRESS", "127.0.0.1:8188")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
DEFAULT_WORKFLOW = os.getenv("DEFAULT_WORKFLOW", "z_image_turbo.json")

# Fixed values for OpenAI API compatibility
FIXED_MODEL = "dall-e-3"
FIXED_SIZE = "256x256"  # Matches the workflow's EmptySD3LatentImage node
FIXED_QUALITY = "standard"
FIXED_STYLE = "vivid"


def validate_request(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate request data and return error dict if invalid."""
    if not isinstance(data, dict):
        return {
            "error": {
                "message": "Invalid JSON payload",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_json"
            }
        }

    # Check required field
    if "prompt" not in data:
        return {
            "error": {
                "message": "'prompt' is a required field",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "missing_field"
            }
        }

    # Validate prompt is a string
    if not isinstance(data["prompt"], str):
        return {
            "error": {
                "message": "'prompt' must be a string",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "invalid_type"
            }
        }

    # Validate prompt is not empty
    if len(data["prompt"].strip()) == 0:
        return {
            "error": {
                "message": "'prompt' cannot be empty",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "empty_field"
            }
        }

    # Validate n parameter if present
    if "n" in data:
        if not isinstance(data["n"], int) or data["n"] != 1:
            return {
                "error": {
                    "message": "'n' must be 1 (only single image generation supported)",
                    "type": "invalid_request_error",
                    "param": "n",
                    "code": "invalid_value"
                }
            }

    # Validate response_format if present
    if "response_format" in data:
        if data["response_format"] not in ["url", "b64_json"]:
            return {
                "error": {
                    "message": "'response_format' must be 'url' or 'b64_json'",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "invalid_value"
                }
            }

    return None


def create_openai_response(image_bytes: bytes, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create OpenAI-compatible response with image data."""
    # Convert image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Create response structure
    response = {
        "created": int(time.time()),
        "data": [
            {
                "b64_json": image_base64,
                "revised_prompt": request_data.get("prompt", "")  # Always include original prompt
            }
        ]
    }

    # Use the model from request or default
    response["model"] = request_data.get("model", FIXED_MODEL)

    return response


@app.route('/v1/images/generations', methods=['POST'])
def images_generations():
    """Handle image generation requests following OpenAI API format."""
    # Check content type
    if not request.is_json:
        return jsonify({
            "error": {
                "message": "Content-Type must be application/json",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_content_type"
            }
        }), 400

    # Parse JSON data
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Invalid JSON: {str(e)}",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_json"
            }
        }), 400

    # Validate request
    validation_error = validate_request(data)
    if validation_error:
        return jsonify(validation_error), 400

    # Extract parameters with defaults
    prompt = data["prompt"]
    n = data.get("n", 1)
    size = data.get("size", FIXED_SIZE)
    response_format = data.get("response_format", "b64_json")
    model = data.get("model", FIXED_MODEL)
    quality = data.get("quality", FIXED_QUALITY)
    style = data.get("style", FIXED_STYLE)

    # Log request (for debugging)
    app.logger.info(f"Generating image with prompt: '{prompt[:50]}...'")

    try:
        # Generate image using ComfyUI
        image_bytes = generate_image_in_memory(
            prompt_text=prompt,
            workflow_path=DEFAULT_WORKFLOW,
            server_address=COMFYUI_SERVER_ADDRESS
        )

        # Create OpenAI-compatible response
        response = create_openai_response(image_bytes, data)

        # If response_format is "url", we'd need to host the image
        # For now, we only support b64_json
        if response_format == "url":
            # This would require saving the image and hosting it
            # For simplicity, we still return b64_json but could add URL support later
            app.logger.warning("URL response format requested but not yet implemented, returning b64_json")

        return jsonify(response)

    except ValueError as e:
        # Workflow loading error
        return jsonify({
            "error": {
                "message": f"Workflow error: {str(e)}",
                "type": "invalid_request_error",
                "param": "workflow",
                "code": "workflow_error"
            }
        }), 400

    except ConnectionError as e:
        # ComfyUI connection error
        return jsonify({
            "error": {
                "message": f"ComfyUI connection failed: {str(e)}",
                "type": "server_error",
                "param": None,
                "code": "comfyui_connection_error"
            }
        }), 503

    except Exception as e:
        # Generic error
        app.logger.error(f"Image generation failed: {str(e)}")
        return jsonify({
            "error": {
                "message": f"Image generation failed: {str(e)}",
                "type": "server_error",
                "param": None,
                "code": "image_generation_error"
            }
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Basic health check - could be extended to check ComfyUI connection
    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time()),
        "service": "comfyui-openai-api"
    })


@app.route('/')
def index():
    """Root endpoint with basic information."""
    return jsonify({
        "service": "ComfyUI OpenAI Images API Wrapper",
        "version": "1.0.0",
        "endpoints": {
            "POST /v1/images/generations": "Generate images from text prompts",
            "GET /health": "Health check"
        },
        "compatibility": "OpenAI Images API v1"
    })


if __name__ == '__main__':
    print(f"Starting OpenAI Images API compatible server on {API_HOST}:{API_PORT}")
    print(f"Using ComfyUI server at {COMFYUI_SERVER_ADDRESS}")
    print(f"Default workflow: {DEFAULT_WORKFLOW}")
    print(f"API endpoint: POST http://{API_HOST}:{API_PORT}/v1/images/generations")
    print("Press Ctrl+C to stop")

    app.run(host=API_HOST, port=API_PORT, debug=False)