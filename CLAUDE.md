# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple Python wrapper for the ComfyUI API, allowing programmatic image generation via ComfyUI workflows. The main script (`server.py`) provides functions to connect to a running ComfyUI instance, queue prompts, monitor progress, and retrieve generated images.

## Architecture

### Core Components

1. **WebSocket Communication**: `open_websocket_connection()` establishes a WebSocket connection to ComfyUI (default: `127.0.0.1:8188`) for real-time progress tracking.

2. **HTTP API Wrappers**: Functions like `queue_prompt()`, `get_history()`, `get_image()`, and `upload_image()` interact with ComfyUI's REST API endpoints.

3. **Workflow Management**: `load_workflow()` loads JSON workflow files (e.g., `z_image_turbo.json`), and `prompt_to_image()` modifies prompts within the workflow before execution.

4. **Image Generation Pipeline**: `generate_image_by_prompt()` orchestrates the full process: connect, queue prompt, track progress, retrieve images, and save to disk.

5. **Progress Tracking**: `track_progress()` monitors WebSocket messages to report node completion and sampler steps.

### Data Flow
1. Load a JSON workflow (ComfyUI node graph)
2. Modify prompt text in appropriate nodes
3. Queue the prompt via HTTP API
4. Track execution via WebSocket
5. Retrieve generated images from history
6. Save images to `output/` directory

### Key Dependencies
- `websocket-client`: WebSocket communication with ComfyUI
- `requests-toolbelt`: Multipart encoding for image uploads
- `Pillow` (PIL): Image processing and saving

## Development Commands

### Running the Script
```bash
python server.py
```
This executes the main block which loads `z_image_turbo.json` and generates an image with the prompt "a bottle of water".

### Dependency Installation
```bash
pip install websocket-client requests-toolbelt Pillow
```

### Testing
Since this is a client script, testing requires a running ComfyUI instance. The script assumes ComfyUI is available at `127.0.0.1:8188`. Modify `server_address` in `open_websocket_connection()` if needed.

### Custom Usage
To use the wrapper programmatically:
```python
from server import load_workflow, prompt_to_image

workflow = load_workflow("z_image_turbo.json")
prompt_to_image(workflow, "your prompt here", negative_prompt="", save_previews=False)
```

## Workflow Files

Workflow files are ComfyUI JSON exports defining the node graph for image generation. The repository includes `z_image_turbo.json` which contains:
- CLIP loader with `qwen_3_4b.safetensors`
- VAE loader with `ae.safetensors`
- UNet loader with `z_image_turbo_bf16.safetensors`
- KSampler with `res_multistep` sampler
- SaveImage node for output

The `prompt_to_image()` function locates the KSampler and CLIP text encode nodes to replace prompt text.

## Configuration

- **Server Address**: Hardcoded as `127.0.0.1:8188` in `open_websocket_connection()`
- **Output Directory**: `./output/` (created if missing)
- **Client ID**: Random UUID generated per connection

## Notes for Contributors

- The script is designed as a simple wrapper, not a full-featured SDK.
- Error handling is minimal; exceptions may occur if ComfyUI is not running.
- Image retrieval assumes specific node output structure (images in `outputs` with `type` field).
- To support new workflows, ensure the `prompt_to_image()` function can locate the appropriate nodes for prompt injection.

## OpenAI Images API Server

An HTTP server implementing OpenAI Images API compatibility has been added to the repository.

### Files

1. **`api_server.py`** - Flask-based HTTP server implementing OpenAI Images API
   - Endpoint: `POST /v1/images/generations`
   - Returns: OpenAI-compatible JSON responses with base64-encoded images
   - Validation: Request validation with OpenAI-style error responses
   - Health check: `GET /health` endpoint

2. **`requirements.txt`** - Additional dependencies for the API server
   - `Flask>=2.3.0` - Web framework
   - Existing dependencies: `websocket-client`, `requests-toolbelt`, `Pillow`

3. **`test_api.py`** - Test script for API functionality
   - Direct function testing
   - API endpoint testing
   - Example usage

### Key Modifications to `server.py`

1. **Parameterized server address**: `open_websocket_connection()` now accepts `server_address` parameter
2. **In-memory image generation**: `generate_image_by_prompt()` accepts `return_images=True` parameter
3. **New helper functions**:
   - `generate_image_in_memory()`: Generate image and return bytes
   - `image_to_base64()`: Convert image bytes to base64 string

### Running the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (default: http://0.0.0.0:5000)
python api_server.py

# Or with custom configuration
COMFYUI_SERVER_ADDRESS="127.0.0.1:8188" \
API_HOST="0.0.0.0" \
API_PORT=5000 \
python api_server.py
```

### Testing the API

```bash
# Test server functions directly
python test_api.py --server-test

# Test API endpoint (requires server running)
python test_api.py --api-test

# Example curl request
curl -X POST http://localhost:5000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "prompt": "A beautiful sunset",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

### API Compatibility

The server implements a subset of the OpenAI Images API:
- **Supported**: `prompt` (required), `n` (fixed to 1), `size` (fixed to 256x256), `response_format` (b64_json only)
- **Fixed values**: `model="dall-e-3"`, `quality="standard"`, `style="vivid"`
- **Authentication**: Header accepted but not validated for local use
- **Response format**: Matches OpenAI API with `created` timestamp and `data` array

### Architecture

1. **Request flow**: Client → Flask API → `generate_image_in_memory()` → ComfyUI → Image bytes → Base64 → JSON response
2. **Error handling**: OpenAI-style error responses for validation and generation failures
3. **Configuration**: Environment variables for server addresses and ports

### OpenAI Client Compatibility

The API server is fully compatible with the official OpenAI Python SDK. Example usage:

```python
import openai

# Any non-empty API key works for local server
client = openai.Client(api_key="any-non-empty-string", base_url="http://localhost:5000/v1")

# Generate an image
response = client.images.generate(
    model="dall-e-3",  # Model name echoed in response
    prompt="a red apple on a wooden table",
    n=1,               # Must be 1 (single image generation)
    size="256x256",    # Fixed size matching workflow
    response_format="b64_json"
)

# Access the generated image
image_data = response.data[0].b64_json
```

**Test scripts included**:
- `test_openai_client.py`: Full compatibility test using OpenAI SDK
- `test_api.py`: Basic API functionality tests

**Key compatibility features**:
- Accepts standard OpenAI API request format
- Returns OpenAI-compatible JSON responses
- Handles authentication headers (not validated)
- Provides appropriate error messages in OpenAI format
- Supports long-running image generation (may take 30-60 seconds)

**Note**: Image generation via ComfyUI may take significant time (30+ seconds). The OpenAI SDK handles this with appropriate timeouts.