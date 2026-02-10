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
import threading
import queue
import re
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
import logging

# Import functions from our modified server.py
try:
    from server import generate_image_in_memory, image_to_base64, load_workflow
except ImportError as e:
    print(f"Error importing from server.py: {e}")
    print("Make sure server.py is in the same directory and has been modified with the required functions.")
    raise

class BackendManager:
    """Manages multiple ComfyUI backends with independent queues for load balancing."""

    def __init__(self, backend_addresses, default_workflow, fixed_size):
        """
        Initialize backend manager with list of backend addresses.

        Args:
            backend_addresses: List of ComfyUI backend addresses (e.g., ["127.0.0.1:8188", "127.0.0.1:8189"])
            default_workflow: Path to default workflow JSON file
            fixed_size: Default size string (e.g., "256x256")
        """
        self.backends = []
        self.backend_lock = threading.Lock()  # Lock for thread-safe access to backend list
        self.default_workflow = default_workflow
        self.fixed_size = fixed_size
        self.jobs = {}  # Dictionary to track jobs by job_id

        # Initialize each backend with its own queue
        for idx, address in enumerate(backend_addresses):
            backend = {
                'id': idx,
                'address': address,
                'queue': queue.Queue(),  # Queue for pending requests
                'queue_size': 0,  # Current queue size (cached for performance)
                'is_processing': False,  # Whether this backend is currently processing a request
                'lock': threading.Lock(),  # Lock for this backend
            }
            self.backends.append(backend)

    def add_request(self, request_data):
        """
        Add a request to the backend with the smallest queue.

        Args:
            request_data: Dictionary containing request parameters

        Returns:
            tuple: (backend_id, job_id) for tracking the request

        Raises:
            RuntimeError: If no backends are available
        """
        with self.backend_lock:
            if not self.backends:
                raise RuntimeError("No ComfyUI backends available")

            # Find backend with smallest queue
            selected_backend = min(self.backends, key=lambda b: b['queue_size'])

            # Create job ID
            job_id = str(time.time()) + "_" + str(selected_backend['id'])

            # Create job data with completion event
            job_data = {
                'job_id': job_id,
                'request_data': request_data,
                'status': 'queued',
                'result': None,
                'error': None,
                'created_at': time.time(),
                'completion_event': threading.Event(),  # Event to signal job completion
            }

            # Store job in global jobs dictionary
            self.jobs[job_id] = job_data

            # Add to backend queue
            with selected_backend['lock']:
                selected_backend['queue'].put(job_data)
                selected_backend['queue_size'] += 1
                print(f"[DEBUG] Added job {job_id} to backend {selected_backend['id']} queue. Queue size: {selected_backend['queue_size']}")

            return selected_backend['id'], job_id

    def wait_for_job(self, job_id, timeout=None):
        """
        Wait for a job to complete and return its result.

        Args:
            job_id: Job ID to wait for
            timeout: Optional timeout in seconds

        Returns:
            tuple: (status, result_or_error) - status is 'completed' or 'failed'

        Raises:
            KeyError: If job_id not found
            TimeoutError: If timeout occurs
        """
        # Get job data
        with self.backend_lock:
            if job_id not in self.jobs:
                raise KeyError(f"Job {job_id} not found")
            job_data = self.jobs[job_id]

        # Wait for completion event
        if not job_data['completion_event'].wait(timeout=timeout):
            raise TimeoutError(f"Job {job_id} timed out after {timeout} seconds")

        # Get result
        with self.backend_lock:
            status = job_data['status']
            result = job_data['result'] if status == 'completed' else job_data['error']

            # Clean up job from dictionary (optional, to prevent memory leak)
            # For long-running server, you might want to keep jobs for some time
            # or implement a cleanup mechanism
            # del self.jobs[job_id]

            return status, result

    def process_next_job(self, backend_id):
        """
        Process the next job for a specific backend.
        This method should be called in a worker thread for each backend.

        Args:
            backend_id: ID of the backend to process jobs for
        """
        backend = self.backends[backend_id]
        print(f"[WORKER] Worker thread started for backend {backend_id} ({backend['address']})")

        while True:
            # Get next job from queue
            try:
                job_data = backend['queue'].get(timeout=1)  # Wait for up to 1 second
                print(f"[WORKER] Backend {backend_id} got job: {job_data['job_id']}")
            except queue.Empty:
                # No jobs in queue, continue waiting
                continue
            except Exception as e:
                print(f"[WORKER ERROR] Backend {backend_id} failed to get job from queue: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)  # Wait before retrying
                continue
            
            try:
                # Process the job
                request_data = job_data['request_data']
                prompt = request_data.get('prompt', '')
                size = request_data.get('size', self.fixed_size)

                # Update job status
                with self.backend_lock:
                    job_data['status'] = 'processing'
                    backend['is_processing'] = True
                print(f"[WORKER] Backend {backend_id} processing job {job_data['job_id']}: '{prompt[:50]}...'")

                # Generate image using the specific backend
                try:
                    print(f"[WORKER] Backend {backend_id} calling generate_image_in_memory...")
                    image_bytes = generate_image_in_memory(
                        prompt_text=prompt,
                        workflow_path=self.default_workflow,
                        server_address=backend['address'],
                        size=size if size != self.fixed_size else None
                    )
                    print(f"[WORKER] Backend {backend_id} image generation completed")

                    # Update job result
                    with self.backend_lock:
                        job_data['status'] = 'completed'
                        job_data['result'] = image_bytes

                    # Signal job completion
                    job_data['completion_event'].set()

                except Exception as e:
                    # Update job error
                    with self.backend_lock:
                        job_data['status'] = 'failed'
                        job_data['error'] = str(e)

                    # Signal job completion (even on failure)
                    job_data['completion_event'].set()

            finally:
                # Mark backend as available for next job
                with self.backend_lock:
                    backend['is_processing'] = False

                # Update queue size AFTER task is done
                backend['queue_size'] = backend['queue'].qsize()

                # Clean up completed jobs from dictionary to prevent memory leak
                # Keep only jobs from the last hour
                current_time = time.time()
                jobs_to_remove = []
                for jid, jdata in self.jobs.items():
                    if jdata['status'] in ['completed', 'failed'] and current_time - jdata['created_at'] > 3600:
                        jobs_to_remove.append(jid)

                for jid in jobs_to_remove:
                    del self.jobs[jid]

                # Mark queue task as done
                backend['queue'].task_done()

    def get_job_status(self, job_id):
        """
        Get the status and result of a job.

        Args:
            job_id: Job ID to look up

        Returns:
            tuple: (status, result_data_or_error) or (None, None) if job not found
        """
        with self.backend_lock:
            for backend in self.backends:
                # Check if job is in this backend's queue (we'd need to track jobs better)
                # For simplicity, we'll check all backend queues and active processing
                # In a production system, you'd want a better job tracking mechanism
                pass

        # For MVP, we'll implement a simpler approach - jobs complete immediately
        # We'll modify the API to wait for job completion
        return None, None

    def get_backend_stats(self):
        """
        Get statistics for all backends.

        Returns:
            List of backend statistics dictionaries
        """
        with self.backend_lock:
            stats = []
            for backend in self.backends:
                stat = {
                    'id': backend['id'],
                    'address': backend['address'],
                    'queue_size': backend['queue_size'],
                    'is_processing': backend['is_processing'],
                    'is_available': not backend['is_processing'],
                }
                stats.append(stat)
            return stats

app = Flask(__name__)

# Configuration from environment variables
# Support multiple ComfyUI backends for load balancing
COMFYUI_SERVER_ADDRESSES = os.getenv("COMFYUI_SERVER_ADDRESSES", "127.0.0.1:8188")
# Parse comma-separated list of backend addresses
BACKEND_ADDRESSES = [addr.strip() for addr in COMFYUI_SERVER_ADDRESSES.split(",") if addr.strip()]


API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
DEFAULT_WORKFLOW = os.getenv("DEFAULT_WORKFLOW", "z_image_turbo.json")

# Fixed values for OpenAI API compatibility
FIXED_MODEL = "dall-e-3"
FIXED_SIZE = "256x256"  # Matches the workflow's EmptySD3LatentImage node
FIXED_QUALITY = "standard"
FIXED_STYLE = "vivid"

# Create backend manager for load balancing
backend_manager = BackendManager(
    backend_addresses=BACKEND_ADDRESSES,
    default_workflow=DEFAULT_WORKFLOW,
    fixed_size=FIXED_SIZE
)

# Start worker threads for each backend
worker_threads = []
for backend_id in range(len(BACKEND_ADDRESSES)):
    thread = threading.Thread(
        target=backend_manager.process_next_job,
        args=(backend_id,),
        daemon=True  # Daemon threads will exit when main program exits
    )
    thread.start()
    worker_threads.append(thread)

print(f"Started {len(worker_threads)} worker threads for backends: {BACKEND_ADDRESSES}")


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

    # Validate size parameter if present
    if "size" in data:
        size = data["size"]
        if not isinstance(size, str):
            return {
                "error": {
                    "message": "'size' must be a string",
                    "type": "invalid_request_error",
                    "param": "size",
                    "code": "invalid_type"
                }
            }
        # Validate format: WIDTHxHEIGHT (e.g., 256x256, 512x512, 1024x1024)
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
            # Optional: limit to reasonable sizes to prevent excessive memory usage
            if width > 2048 or height > 2048:
                return {
                    "error": {
                        "message": "'size' dimensions too large (maximum 2048x2048)",
                        "type": "invalid_request_error",
                        "param": "size",
                        "code": "dimension_too_large"
                    }
                }
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
        # Add request to backend manager queue
        # The backend manager will distribute the request to the backend with smallest queue
        app.logger.info(f"Queuing image generation request for prompt: '{prompt[:50]}...' (size: {size})")

        # Prepare request data for backend manager
        request_data = {
            'prompt': prompt,
            'size': size,
            'model': model,
            'quality': quality,
            'style': style,
            'response_format': response_format,
            'n': n,
        }

        # Add request to backend manager queue
        backend_id, job_id = backend_manager.add_request(request_data)
        app.logger.info(f"Request queued with job_id: {job_id} on backend: {backend_id}")

        # Wait for job completion with timeout (e.g., 5 minutes for image generation)
        timeout = 300  # 5 minutes timeout for image generation
        status, result = backend_manager.wait_for_job(job_id, timeout=timeout)

        if status == 'completed':
            image_bytes = result
            # Create OpenAI-compatible response
            response = create_openai_response(image_bytes, data)
        else:
            # Job failed
            error_message = result
            raise RuntimeError(f"Image generation failed: {error_message}")

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
    import logging

    print(f"Starting OpenAI Images API compatible server on {API_HOST}:{API_PORT}")
    print(f"Using ComfyUI backends: {BACKEND_ADDRESSES}")
    print(f"Number of backends: {len(BACKEND_ADDRESSES)}")
    print(f"Number of worker threads started: {len(worker_threads)}")
    print(f"Default workflow: {DEFAULT_WORKFLOW}")
    print(f"API endpoint: POST http://{API_HOST}:{API_PORT}/v1/images/generations")
    print("Press Ctrl+C to stop")

    # Check if worker threads are alive
    alive_threads = sum(1 for thread in worker_threads if thread.is_alive())
    print(f"Worker threads alive: {alive_threads}/{len(worker_threads)}")

    if alive_threads == 0:
        print("WARNING: No worker threads are alive! Requests will hang.")
        print("This could be due to import errors or initialization issues.")

    app.logger.setLevel(logging.DEBUG)
    app.run(host=API_HOST, port=API_PORT, debug=False)