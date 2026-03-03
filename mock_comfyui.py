#!/usr/bin/env python3
"""
Mock ComfyUI server for testing.
This simulates a real ComfyUI server with WebSocket and HTTP API endpoints.
"""

import json
import time
import random
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import websocket
from websocket_server import WebsocketServer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[MOCK] %(message)s')
logger = logging.getLogger(__name__)


class MockComfyUIHandler(BaseHTTPRequestHandler):
    """HTTP handler for mock ComfyUI API endpoints."""

    def log_message(self, format, *args):
        logger.info(f"HTTP {self.command} {self.path} - {args[0] if args else ''}")

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith('/history/'):
            self._handle_get_history()
        elif self.path.startswith('/view'):
            self._handle_get_image()
        elif self.path == '/':
            self._handle_root()
        else:
            self.send_error(404, f"Endpoint not found: {self.path}")

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if self.path == '/prompt':
            self._handle_post_prompt(post_data)
        elif self.path == '/upload/image':
            self._handle_upload_image(post_data)
        else:
            self.send_error(404, f"Endpoint not found: {self.path}")

    def _handle_root(self):
        """Handle root endpoint."""
        response = {
            "message": "Mock ComfyUI Server",
            "endpoints": {
                "POST /prompt": "Queue a prompt",
                "GET /history/{prompt_id}": "Get prompt history",
                "GET /view": "Get image/video",
                "POST /upload/image": "Upload image"
            }
        }
        self._send_json_response(response)

    def _handle_post_prompt(self, post_data):
        """Handle prompt queuing."""
        try:
            data = json.loads(post_data.decode('utf-8'))
            prompt = data.get('prompt', {})
            client_id = data.get('client_id', str(uuid.uuid4()))

            # Generate a prompt ID
            prompt_id = f"mock_prompt_{int(time.time())}_{random.randint(1000, 9999)}"

            # Store the prompt for later retrieval
            if not hasattr(self.server, 'prompts'):
                self.server.prompts = {}
            self.server.prompts[prompt_id] = {
                'prompt': prompt,
                'client_id': client_id,
                'created_at': time.time(),
                'status': 'queued'
            }

            # Schedule execution simulation
            threading.Thread(
                target=self._simulate_prompt_execution,
                args=(prompt_id, client_id),
                daemon=True
            ).start()

            response = {
                "prompt_id": prompt_id,
                "number": 1,
                "node_errors": {}
            }
            self._send_json_response(response)

        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            self.send_error(400, f"Invalid request: {str(e)}")

    def _handle_get_history(self):
        """Handle history retrieval."""
        try:
            # Extract prompt_id from path
            prompt_id = self.path.split('/')[-1]

            if not hasattr(self.server, 'prompts') or prompt_id not in self.server.prompts:
                self.send_error(404, f"Prompt not found: {prompt_id}")
                return

            prompt_data = self.server.prompts[prompt_id]

            # Create mock output based on prompt content
            outputs = self._create_mock_outputs(prompt_data['prompt'])

            response = {
                prompt_id: {
                    "prompt": prompt_data['prompt'],
                    "outputs": outputs
                }
            }
            self._send_json_response(response)

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            self.send_error(500, f"Internal error: {str(e)}")

    def _handle_get_image(self):
        """Handle image/video retrieval."""
        try:
            # Parse query parameters
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)

            filename = query.get('filename', ['mock_output'])[0]
            folder_type = query.get('type', ['output'])[0]

            # Create mock image or video data
            if 'video' in filename.lower() or 'webp' in filename.lower():
                # Create a simple animated WEBP (just a minimal valid file)
                # This is a minimal valid 1x1 pixel black WEBP animation
                mock_data = bytes([
                    0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,  # RIFF header
                    0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58,  # WEBPVP8X
                    0x0A, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,  # Chunk header
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,  # Animation params
                    0x00, 0x2A  # Footer
                ])
                content_type = 'image/webp'
            else:
                # Create a simple 1x1 pixel black PNG
                mock_data = bytes([
                    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 image
                    0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
                    0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
                    0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
                    0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0x57, 0x88,
                    0xB0, 0x7D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
                    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82  # IEND chunk
                ])
                content_type = 'image/png'

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(mock_data)))
            self.end_headers()
            self.wfile.write(mock_data)

        except Exception as e:
            logger.error(f"Error getting image: {e}")
            self.send_error(500, f"Internal error: {str(e)}")

    def _handle_upload_image(self, post_data):
        """Handle image upload."""
        # Just acknowledge the upload
        response = {
            "name": "mock_uploaded_image.png",
            "subfolder": "",
            "type": "input"
        }
        self._send_json_response(response)

    def _create_mock_outputs(self, prompt):
        """Create mock outputs based on prompt content."""
        outputs = {}

        # Check if this looks like a video prompt
        is_video = any(keyword in str(prompt).lower()
                      for keyword in ['video', 'animation', 'motion', 'moving'])

        if is_video:
            # Video outputs
            outputs["8"] = {  # VAE Decode node
                "animated_output": [
                    {
                        "filename": f"mock_video_{int(time.time())}.webp",
                        "subfolder": "",
                        "type": "output"
                    }
                ]
            }
            outputs["41"] = {  # SaveAnimatedWEBP node
                "outputs": {
                    "animated": [
                        {
                            "filename": f"ComfyUI_{int(time.time())}.webp",
                            "subfolder": "",
                            "type": "output"
                        }
                    ]
                }
            }
        else:
            # Image outputs
            outputs["3"] = {  # SaveImage node
                "images": [
                    {
                        "filename": f"mock_image_{int(time.time())}.png",
                        "subfolder": "",
                        "type": "output"
                    }
                ]
            }

        return outputs

    def _simulate_prompt_execution(self, prompt_id, client_id):
        """Simulate prompt execution with WebSocket updates."""
        time.sleep(0.5)  # Simulate processing time

        # Update prompt status
        if hasattr(self.server, 'prompts') and prompt_id in self.server.prompts:
            self.server.prompts[prompt_id]['status'] = 'executing'

        # Send WebSocket messages if server has WebSocket server
        if hasattr(self.server, 'ws_server'):
            # Send progress updates
            for i in range(1, 11):
                time.sleep(0.1)
                progress_msg = {
                    "type": "progress",
                    "data": {
                        "value": i * 10,
                        "max": 100,
                        "prompt_id": prompt_id
                    }
                }
                self.server.ws_server.send_message_to_all(json.dumps(progress_msg))

            # Send execution message
            exec_msg = {
                "type": "executing",
                "data": {
                    "node": "3",  # SaveImage node
                    "prompt_id": prompt_id
                }
            }
            self.server.ws_server.send_message_to_all(json.dumps(exec_msg))

            # Send completion message
            time.sleep(0.5)
            done_msg = {
                "type": "executing",
                "data": {
                    "node": None,
                    "prompt_id": prompt_id
                }
            }
            self.server.ws_server.send_message_to_all(json.dumps(done_msg))

        # Final status update
        if hasattr(self.server, 'prompts') and prompt_id in self.server.prompts:
            self.server.prompts[prompt_id]['status'] = 'completed'

    def _send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


class MockWebSocketServer:
    """WebSocket server for mock ComfyUI."""

    def __init__(self, host='127.0.0.1', port=8188):
        self.host = host
        self.port = port
        self.server = None
        self.clients = []

    def start(self):
        """Start WebSocket server."""
        self.server = WebsocketServer(host=self.host, port=self.port)
        self.server.set_fn_new_client(self._on_connect)
        self.server.set_fn_client_left(self._on_disconnect)
        self.server.set_fn_message_received(self._on_message)

        logger.info(f"Starting mock WebSocket server on ws://{self.host}:{self.port}")
        threading.Thread(target=self.server.run_forever, daemon=True).start()

    def send_message_to_all(self, message):
        """Send message to all connected clients."""
        if self.server:
            self.server.send_message_to_all(message)

    def _on_connect(self, client, server):
        """Handle new client connection."""
        client_id = str(uuid.uuid4())
        client['id'] = client_id
        self.clients.append(client)

        logger.info(f"WebSocket client connected: {client_id}")

        # Send welcome message
        welcome_msg = {
            "type": "status",
            "data": {
                "status": {"exec_info": {"queue_remaining": 0}},
                "sid": client_id
            }
        }
        server.send_message(client, json.dumps(welcome_msg))

    def _on_disconnect(self, client, server):
        """Handle client disconnect."""
        if client in self.clients:
            self.clients.remove(client)
        logger.info(f"WebSocket client disconnected: {client.get('id', 'unknown')}")

    def _on_message(self, client, server, message):
        """Handle incoming message."""
        try:
            data = json.loads(message)
            logger.info(f"WebSocket message from {client['id']}: {data.get('type', 'unknown')}")

            # Echo back for ping/pong
            if data.get('type') == 'ping':
                pong_msg = {"type": "pong"}
                server.send_message(client, json.dumps(pong_msg))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message}")


class MockComfyUIServer:
    """Main mock ComfyUI server combining HTTP and WebSocket."""

    def __init__(self, http_port=8188, ws_port=8188):
        self.http_port = http_port
        self.ws_port = ws_port
        self.http_server = None
        self.ws_server = None

    def start(self):
        """Start both HTTP and WebSocket servers."""
        logger.info(f"Starting mock ComfyUI server...")
        logger.info(f"HTTP API: http://127.0.0.1:{self.http_port}")
        logger.info(f"WebSocket: ws://127.0.0.1:{self.ws_port}")

        # Start WebSocket server
        self.ws_server = MockWebSocketServer(port=self.ws_port)
        self.ws_server.start()

        # Start HTTP server with reference to WebSocket server
        handler = type('Handler', (MockComfyUIHandler,), {})
        self.http_server = HTTPServer(('127.0.0.1', self.http_port), handler)
        self.http_server.ws_server = self.ws_server.server
        self.http_server.prompts = {}

        # Start HTTP server in background thread
        http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        http_thread.start()

        logger.info("Mock ComfyUI server started successfully")

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down mock server...")
            self.stop()

    def stop(self):
        """Stop the servers."""
        if self.http_server:
            self.http_server.shutdown()
        logger.info("Mock server stopped")


def start_mock_servers():
    """Start multiple mock ComfyUI servers on different ports."""
    servers = []

    # Start servers on different ports as defined in backend_config.json
    ports = [8188, 8189, 8190, 8191, 8192]

    for port in ports:
        try:
            server = MockComfyUIServer(http_port=port, ws_port=port)
            thread = threading.Thread(target=server.start, daemon=True)
            thread.start()
            servers.append((server, thread))
            logger.info(f"Started mock server on port {port}")
            time.sleep(0.5)  # Stagger startup
        except Exception as e:
            logger.error(f"Failed to start server on port {port}: {e}")

    logger.info(f"Started {len(servers)} mock servers")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down all mock servers...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mock ComfyUI server')
    parser.add_argument('--single', action='store_true', help='Start single server on port 8188')
    parser.add_argument('--multi', action='store_true', help='Start multiple servers on ports 8188-8192')
    parser.add_argument('--port', type=int, default=8188, help='Port for single server')

    args = parser.parse_args()

    if args.multi:
        start_mock_servers()
    else:
        server = MockComfyUIServer(http_port=args.port, ws_port=args.port)
        server.start()