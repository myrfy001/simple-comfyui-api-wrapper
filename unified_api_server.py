#!/usr/bin/env python3
"""
Unified OpenAI API compatible server for ComfyUI with model-based routing.
This server implements both Images API and Video API endpoints with model-based backend routing.
"""

import os
import json
import time
import base64
import threading
import re
from flask import Flask, request, jsonify, send_file
from typing import Dict, Any, Optional
import logging
from io import BytesIO

# Import model router
try:
    from model_router import ModelRouter, ImageModelBackendConfig, VideoModelBackendConfig
except ImportError as e:
    print(f"Error importing model_router: {e}")
    print("Make sure model_router.py is in the same directory.")
    raise

# Import task manager for video API compatibility
try:
    from server import task_manager, TaskStatus
except ImportError as e:
    print(f"Error importing from server.py: {e}")
    print("Make sure server.py is in the same directory and has been modified with the required functions.")
    raise

app = Flask(__name__)

# Configuration from environment variables
CONFIG_PATH = os.getenv("BACKEND_CONFIG_PATH", "backend_config.json")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))

# Initialize model router
model_router = ModelRouter(config_path=CONFIG_PATH)

# Log available models
print("Available models:")
for model_info in model_router.list_models():
    print(f"  - {model_info['name']}: {model_info['description']}")
    print(f"    Workflow: {model_info['workflow']}, Backends: {model_info['backends']}")


# ========== Common Validation Functions ==========

def validate_request_json() -> Optional[Dict[str, Any]]:
    """Validate that request has valid JSON."""
    if not request.is_json:
        return {
            "error": {
                "message": "Content-Type must be application/json",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_content_type"
            }
        }

    try:
        data = request.get_json()
        return data
    except Exception as e:
        return {
            "error": {
                "message": f"Invalid JSON: {str(e)}",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_json"
            }
        }


def validate_prompt(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate prompt field in request data."""
    if "prompt" not in data:
        return {
            "error": {
                "message": "'prompt' is a required field",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "missing_field"
            }
        }

    if not isinstance(data["prompt"], str):
        return {
            "error": {
                "message": "'prompt' must be a string",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "invalid_type"
            }
        }

    if len(data["prompt"].strip()) == 0:
        return {
            "error": {
                "message": "'prompt' cannot be empty",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "empty_field"
            }
        }

    return None


def validate_size(size: str) -> Optional[Dict[str, Any]]:
    """Validate size parameter."""
    if not isinstance(size, str):
        return {
            "error": {
                "message": "'size' must be a string",
                "type": "invalid_request_error",
                "param": "size",
                "code": "invalid_type"
            }
        }

    # Validate format: WIDTHxHEIGHT
    if not re.match(r'^\d+x\d+$', size):
        return {
            "error": {
                "message": "'size' must be in format WIDTHxHEIGHT (e.g., 256x256)",
                "type": "invalid_request_error",
                "param": "size",
                "code": "invalid_format"
            }
        }

    # Validate width and height are positive integers
    try:
        width_str, height_str = size.split('x')
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
    except ValueError:
        return {
            "error": {
                "message": "'size' must contain positive integers (e.g., 256x256)",
                "type": "invalid_request_error",
                "param": "size",
                "code": "invalid_dimensions"
            }
        }

    return None


def parse_video_input_image(data: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse optional I2V image input from request body.

    Supported inputs:
    1) OpenAI-style: input_reference.image_url (base64/data-url string)
    2) Compatibility: image (base64/data-url string)
    """
    input_reference = data.get("input_reference")
    image_value = data.get("image")

    if input_reference is not None and image_value is not None:
        return None, {
            "error": {
                "message": "Use either 'input_reference' or 'image', not both",
                "type": "invalid_request_error",
                "param": "input_reference",
                "code": "invalid_value"
            }
        }

    if input_reference is None and image_value is None:
        return None, None

    if image_value is not None:
        if not isinstance(image_value, str) or not image_value.strip():
            return None, {
                "error": {
                    "message": "'image' must be a non-empty base64/data-url string",
                    "type": "invalid_request_error",
                    "param": "image",
                    "code": "invalid_type"
                }
            }
        return image_value.strip(), None

    if not isinstance(input_reference, dict):
        return None, {
            "error": {
                "message": "'input_reference' must be an object",
                "type": "invalid_request_error",
                "param": "input_reference",
                "code": "invalid_type"
            }
        }

    image_url = input_reference.get("image_url")
    file_id = input_reference.get("file_id")

    if image_url and file_id:
        return None, {
            "error": {
                "message": "Provide exactly one of input_reference.image_url or input_reference.file_id",
                "type": "invalid_request_error",
                "param": "input_reference",
                "code": "invalid_value"
            }
        }

    if file_id:
        return None, {
            "error": {
                "message": "'input_reference.file_id' is not supported yet; use input_reference.image_url with base64/data-url",
                "type": "invalid_request_error",
                "param": "input_reference.file_id",
                "code": "unsupported_value"
            }
        }

    if not image_url:
        return None, {
            "error": {
                "message": "'input_reference.image_url' is required for Image-to-Video",
                "type": "invalid_request_error",
                "param": "input_reference.image_url",
                "code": "missing_field"
            }
        }

    if not isinstance(image_url, str) or not image_url.strip():
        return None, {
            "error": {
                "message": "'input_reference.image_url' must be a non-empty base64/data-url string",
                "type": "invalid_request_error",
                "param": "input_reference.image_url",
                "code": "invalid_type"
            }
        }

    return image_url.strip(), None


# ========== Images API Endpoints ==========

@app.route('/v1/images/generations', methods=['POST'])
def images_generations():
    """Handle image generation requests with model-based routing."""
    # Validate JSON
    json_result = validate_request_json()
    if isinstance(json_result, dict) and "error" in json_result:
        return jsonify(json_result), 400
    data = json_result

    # Validate required fields
    prompt_error = validate_prompt(data)
    if prompt_error:
        return jsonify(prompt_error), 400

    # Extract parameters with defaults
    prompt = data["prompt"]
    model = data.get("model", "dall-e-3")
    n = data.get("n", 1)
    size = data.get("size", "256x256")
    response_format = data.get("response_format", "b64_json")
    quality = data.get("quality", "standard")
    style = data.get("style", "vivid")

    # Validate n parameter
    if n != 1:
        return jsonify({
            "error": {
                "message": "'n' must be 1 (only single image generation supported)",
                "type": "invalid_request_error",
                "param": "n",
                "code": "invalid_value"
            }
        }), 400

    # Validate size parameter
    size_error = validate_size(size)
    if size_error:
        return jsonify(size_error), 400

    # Validate response_format
    if response_format not in ["url", "b64_json"]:
        return jsonify({
            "error": {
                "message": "'response_format' must be 'url' or 'b64_json'",
                "type": "invalid_request_error",
                "param": "response_format",
                "code": "invalid_value"
            }
        }), 400

    # Resolve model
    resolved_model = model_router.resolve_model(model)
    model_config = model_router.get_model_config(resolved_model)

    if not model_config:
        return jsonify({
            "error": {
                "message": f"Model '{model}' not supported",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }), 400

    # Check if this is an image model
    if not isinstance(model_config, ImageModelBackendConfig):
        return jsonify({
            "error": {
                "message": f"Model '{model}' is not an image generation model",
                "type": "invalid_request_error",
                "param": "model",
                "code": "invalid_model_type"
            }
        }), 400

    app.logger.info(f"Generating image with model '{resolved_model}': '{prompt[:50]}...'")

    try:
        # Prepare request data
        request_data = {
            'prompt': prompt,
            'size': size,
            'model': resolved_model,
            'quality': quality,
            'style': style,
            'response_format': response_format,
            'n': n,
        }

        # Add request to model backend
        backend_id, job_id = model_config.add_image_request(request_data)
        app.logger.info(f"Image job created: {job_id} on backend {backend_id}")

        # Wait for job completion with timeout
        timeout = 300  # 5 minutes timeout
        status, result = model_config.wait_for_job(job_id, timeout=timeout)

        if status == 'completed':
            image_bytes = result
            # Create OpenAI-compatible response
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            response = {
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": image_base64,
                        "revised_prompt": prompt
                    }
                ],
                "model": resolved_model
            }

            # If response_format is "url", we'd need to host the image
            if response_format == "url":
                app.logger.warning("URL response format requested but not yet implemented, returning b64_json")

            return jsonify(response)
        else:
            # Job failed
            error_message = result
            raise RuntimeError(f"Image generation failed: {error_message}")

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
    except TimeoutError as e:
        # Timeout error
        return jsonify({
            "error": {
                "message": f"Image generation timed out: {str(e)}",
                "type": "server_error",
                "param": None,
                "code": "timeout_error"
            }
        }), 504
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


# ========== Video API Endpoints ==========

@app.route('/v1/videos', methods=['POST'])
def create_video():
    """Create a new video generation job with model-based routing."""
    # Validate JSON
    json_result = validate_request_json()
    if isinstance(json_result, dict) and "error" in json_result:
        return jsonify(json_result), 400
    data = json_result

    # Validate required fields
    prompt_error = validate_prompt(data)
    if prompt_error:
        return jsonify(prompt_error), 400

    # Extract parameters
    prompt = data["prompt"]
    model = data.get("model", "sora-2")
    size = data.get("size", "768x512")
    seconds = data.get("seconds", 4)

    # Optional I2V source image (base64/data-url)
    input_image_base64, input_image_error = parse_video_input_image(data)
    if input_image_error:
        return jsonify(input_image_error), 400

    # Validate size parameter
    size_error = validate_size(size)
    if size_error:
        return jsonify(size_error), 400

    # Validate seconds
    try:
        seconds_int = int(seconds)
        if seconds_int <= 0:
            raise ValueError("Seconds must be positive")
    except (ValueError, TypeError):
        return jsonify({
            "error": {
                "message": "'seconds' must be a positive integer",
                "type": "invalid_request_error",
                "param": "seconds",
                "code": "invalid_value"
            }
        }), 400

    # Resolve model
    resolved_model = model_router.resolve_model(model)
    model_config = model_router.get_model_config(resolved_model)

    if not model_config:
        return jsonify({
            "error": {
                "message": f"Model '{model}' not supported",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }), 400

    # Check if this is a video model
    if not isinstance(model_config, VideoModelBackendConfig):
        return jsonify({
            "error": {
                "message": f"Model '{model}' is not a video generation model",
                "type": "invalid_request_error",
                "param": "model",
                "code": "invalid_model_type"
            }
        }), 400

    app.logger.info(f"Creating video job with model '{resolved_model}': '{prompt[:50]}...'")

    try:
        # Prepare request data
        request_data = {
            'prompt': prompt,
            'model': resolved_model,
            'size': size,
            'seconds': seconds,
            'input_image_base64': input_image_base64,
        }

        # Add request to model backend
        backend_id, job_id = model_config.add_video_request(request_data)
        app.logger.info(f"Video job created: {job_id} on backend {backend_id}")

        # Also create a task in the global task manager for compatibility
        # (This allows the video API to work with the existing task manager)
        from server import TaskStatus
        # Use the same job_id for consistency
        task_id = job_id  # Use the same ID
        # Update status - this will create the task if it doesn't exist
        task_manager.update_task_status(
            task_id,
            TaskStatus.QUEUED,
            progress=0,
            prompt=prompt,
            workflow_path=model_config.workflow
        )
        app.logger.info(f"Task status updated in task manager: {task_id}")

        # Return initial response
        response = {
            "id": job_id,
            "object": "video",
            "created_at": int(time.time()),
            "status": "queued",
            "model": resolved_model,
            "progress": 0,
            "seconds": str(seconds),
            "size": size
        }
        return jsonify(response), 202  # Accepted

    except Exception as e:
        app.logger.error(f"Failed to create video job: {str(e)}")
        return jsonify({
            "error": {
                "message": f"Failed to create video generation job: {str(e)}",
                "type": "server_error",
                "param": None,
                "code": "job_creation_error"
            }
        }), 500


@app.route('/v1/videos/<video_id>', methods=['GET'])
def get_video_status(video_id: str):
    """Get the status of a video generation job."""
    app.logger.info(f"Getting status for video: {video_id}")

    # Try to get from model router first
    video_status = None
    for model_name, model_config in model_router.model_configs.items():
        if isinstance(model_config, VideoModelBackendConfig):
            with model_config.backend_lock:
                if video_id in model_config.jobs:
                    job_data = model_config.jobs[video_id]
                    video_status = {
                        'id': video_id,
                        'status': job_data['status'],
                        'progress': 0,  # Progress tracking would need to be implemented
                        'error': job_data.get('error'),
                        'created_at': job_data['created_at'],
                        'model': model_name,
                        'seconds': str(job_data.get('request_data', {}).get('seconds', 4)),
                        'size': job_data.get('request_data', {}).get('size', '768x512'),
                    }
                    break

    # If not found in model router, try task manager
    if not video_status:
        task = task_manager.get_task(video_id)
        if not task:
            return jsonify({
                "error": {
                    "message": f"Video '{video_id}' not found",
                    "type": "invalid_request_error",
                    "param": "video_id",
                    "code": "not_found"
                }
            }), 404

        # Map task status
        status_map = {
            TaskStatus.QUEUED: "queued",
            TaskStatus.PROCESSING: "in_progress",
            TaskStatus.COMPLETED: "completed",
            TaskStatus.FAILED: "failed"
        }
        video_status = {
            'id': video_id,
            'status': status_map.get(task.status, "unknown"),
            'progress': task.progress,
            'error': task.error,
            'created_at': task.created_at,
            'model': 'sora-2',
            'seconds': '4',
            'size': '768x512',
        }

    # Create response
    response = {
        "id": video_id,
        "object": "video",
        "created_at": int(video_status['created_at']),
        "status": video_status['status'],
        "model": video_status.get('model', 'sora-2'),
        "progress": video_status['progress'],
        "seconds": video_status.get('seconds', '4'),
        "size": video_status.get('size', '768x512')
    }

    # Add error if failed
    if video_status['status'] == 'failed' and video_status['error']:
        response["error"] = {
            "message": video_status['error'],
            "type": "generation_error"
        }

    return jsonify(response)


@app.route('/v1/videos/<video_id>/content', methods=['GET'])
def get_video_content(video_id: str):
    """Download the generated video content."""
    app.logger.info(f"Downloading video content: {video_id}")

    # Try to find video in model router
    video_data = None
    for model_name, model_config in model_router.model_configs.items():
        if isinstance(model_config, VideoModelBackendConfig):
            with model_config.backend_lock:
                if video_id in model_config.jobs:
                    job_data = model_config.jobs[video_id]
                    if job_data['status'] == 'completed' and job_data['result']:
                        video_data = job_data['result']
                    break

    # If not found, try task manager
    if not video_data:
        task = task_manager.get_task(video_id)
        if not task:
            return jsonify({
                "error": {
                    "message": f"Video '{video_id}' not found",
                    "type": "invalid_request_error",
                    "param": "video_id",
                    "code": "not_found"
                }
            }), 404

        if task.status != TaskStatus.COMPLETED:
            return jsonify({
                "error": {
                    "message": f"Video '{video_id}' is not ready (status: {task.status.value})",
                    "type": "invalid_request_error",
                    "param": "video_id",
                    "code": "not_ready"
                }
            }), 425

        if not task.result:
            return jsonify({
                "error": {
                    "message": f"Video '{video_id}' has no content",
                    "type": "server_error",
                    "param": None,
                    "code": "no_content"
                }
            }), 500

        video_data = task.result

    # Determine file extension from file signature (magic bytes)
    file_extension = "bin"
    mimetype = "application/octet-stream"

    # MP4 usually has 'ftyp' at bytes 4..8
    if len(video_data) >= 12 and video_data[4:8] == b'ftyp':
        file_extension = "mp4"
        mimetype = "video/mp4"
    # WEBP usually starts with RIFF....WEBP
    elif len(video_data) >= 12 and video_data[0:4] == b'RIFF' and video_data[8:12] == b'WEBP':
        file_extension = "webp"
        mimetype = "image/webp"

    filename = f"video_{video_id}.{file_extension}"

    # Create BytesIO object
    video_io = BytesIO(video_data)
    video_io.seek(0)

    # Return as file download
    return send_file(
        video_io,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )


@app.route('/v1/videos', methods=['GET'])
def list_videos():
    """List video generation jobs."""
    # Parse query parameters
    limit = request.args.get('limit', default=20, type=int)
    after = request.args.get('after', default=None, type=str)
    order = request.args.get('order', default='desc', type=str)

    # Validate parameters
    if limit < 1 or limit > 100:
        return jsonify({
            "error": {
                "message": "'limit' must be between 1 and 100",
                "type": "invalid_request_error",
                "param": "limit",
                "code": "invalid_value"
            }
        }), 400

    if order not in ['asc', 'desc']:
        return jsonify({
            "error": {
                "message": "'order' must be 'asc' or 'desc'",
                "type": "invalid_request_error",
                "param": "order",
                "code": "invalid_value"
            }
        }), 400

    # Collect video jobs from all video model configs
    all_video_jobs = []
    for model_name, model_config in model_router.model_configs.items():
        if isinstance(model_config, VideoModelBackendConfig):
            with model_config.backend_lock:
                for job_id, job_data in model_config.jobs.items():
                    all_video_jobs.append({
                        'id': job_id,
                        'type': 'video',
                        'prompt': job_data['request_data'].get('prompt', ''),
                        'status': job_data['status'],
                        'progress': 0,
                        'created_at': job_data['created_at'],
                        'updated_at': job_data['created_at'],  # Same as created for now
                        'model': model_name
                    })

    # Also get from task manager
    task_jobs = task_manager.list_tasks(limit=1000)
    video_task_jobs = [task for task in task_jobs if task['type'] == 'video']
    all_video_jobs.extend(video_task_jobs)

    # Remove duplicates (by ID)
    unique_jobs = {}
    for job in all_video_jobs:
        if job['id'] not in unique_jobs:
            unique_jobs[job['id']] = job

    # Convert to list and sort
    video_jobs = list(unique_jobs.values())
    reverse = (order == 'desc')
    video_jobs.sort(key=lambda x: x['created_at'], reverse=reverse)

    # Apply pagination
    if after:
        try:
            after_index = next(i for i, t in enumerate(video_jobs) if t['id'] == after)
            video_jobs = video_jobs[after_index + 1:]
        except StopIteration:
            video_jobs = []

    # Apply limit
    video_jobs = video_jobs[:limit]

    # Format response
    response = {
        "object": "list",
        "data": video_jobs,
        "has_more": len(video_jobs) == limit
    }

    return jsonify(response)


@app.route('/v1/videos/<video_id>', methods=['DELETE'])
def delete_video(video_id: str):
    """Delete a video generation job."""
    app.logger.info(f"Deleting video: {video_id}")

    # Try to delete from model router
    deleted = False
    for model_name, model_config in model_router.model_configs.items():
        if isinstance(model_config, VideoModelBackendConfig):
            with model_config.backend_lock:
                if video_id in model_config.jobs:
                    del model_config.jobs[video_id]
                    deleted = True
                    break

    # Also try to delete from task manager
    if not deleted:
        deleted = task_manager.delete_task(video_id)

    if not deleted:
        return jsonify({
            "error": {
                "message": f"Video '{video_id}' not found",
                "type": "invalid_request_error",
                "param": "video_id",
                "code": "not_found"
            }
        }), 404

    return jsonify({
        "id": video_id,
        "object": "video",
        "deleted": True
    })


# ========== Management Endpoints ==========

@app.route('/admin/models', methods=['GET'])
def list_available_models():
    """List all available models and their configurations."""
    models = model_router.list_models()
    return jsonify({
        "object": "list",
        "data": models
    })


@app.route('/admin/backends', methods=['GET'])
def get_backend_stats():
    """Get statistics for all backends."""
    stats = model_router.get_backend_stats()
    return jsonify(stats)


@app.route('/admin/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Failed to load config: {str(e)}",
                "type": "server_error",
                "code": "config_load_error"
            }
        }), 500


# ========== Common Endpoints ==========

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    backend_stats = model_router.get_backend_stats()
    total_backends = sum(stats['total_backends'] for stats in backend_stats.values())

    return jsonify({
        "status": "healthy",
        "timestamp": int(time.time()),
        "service": "comfyui-openai-unified-api-v2",
        "models": len(model_router.model_configs),
        "total_backends": total_backends,
        "endpoints": {
            "images": "available",
            "videos": "available",
            "admin": "available"
        }
    })


@app.route('/')
def index():
    """Root endpoint with basic information."""
    models = model_router.list_models()
    image_models = [m for m in models if 'dall-e' in m['name'].lower() or 'image' in m['description'].lower()]
    video_models = [m for m in models if 'sora' in m['name'].lower() or 'video' in m['description'].lower()]

    return jsonify({
        "service": "ComfyUI OpenAI Unified API with Model Routing",
        "version": "2.0.0",
        "endpoints": {
            "Images API": {
                "POST /v1/images/generations": "Generate images from text prompts"
            },
            "Video API": {
                "POST /v1/videos": "Create video generation job",
                "GET /v1/videos/<video_id>": "Get video job status",
                "GET /v1/videos/<video_id>/content": "Download generated video",
                "GET /v1/videos": "List video jobs",
                "DELETE /v1/videos/<video_id>": "Delete video job"
            },
            "Admin API": {
                "GET /admin/models": "List available models",
                "GET /admin/backends": "Get backend statistics",
                "GET /admin/config": "Get current configuration"
            },
            "Common": {
                "GET /health": "Health check",
                "GET /": "This information"
            }
        },
        "compatibility": {
            "images": "OpenAI Images API v1",
            "videos": "OpenAI Video API v1"
        },
        "available_models": {
            "image_models": [m['name'] for m in image_models],
            "video_models": [m['name'] for m in video_models]
        },
        "configuration": {
            "config_path": CONFIG_PATH,
            "total_models": len(models)
        }
    })


if __name__ == '__main__':
    import logging

    print(f"Starting OpenAI Unified API Server v2 on {API_HOST}:{API_PORT}")
    print(f"Configuration file: {CONFIG_PATH}")
    print(f"Loaded {len(model_router.model_configs)} model configurations")

    # Print backend statistics
    stats = model_router.get_backend_stats()
    for model_name, model_stats in stats.items():
        print(f"  {model_name}: {model_stats['total_backends']} backends")

    print("\nAvailable endpoints:")
    print("  POST /v1/images/generations - Generate images")
    print("  POST /v1/videos - Create video generation job")
    print("  GET  /v1/videos/<id> - Get video status")
    print("  GET  /v1/videos/<id>/content - Download video")
    print("  GET  /v1/videos - List video jobs")
    print("  DELETE /v1/videos/<id> - Delete video job")
    print("  GET  /admin/models - List available models")
    print("  GET  /admin/backends - Get backend statistics")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop")

    app.logger.setLevel(logging.DEBUG)
    app.run(host=API_HOST, port=API_PORT, debug=False)