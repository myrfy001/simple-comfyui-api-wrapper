
# comes from https://9elements.com/blog/hosting-a-comfyui-workflow-via-api/
import os
import websocket
import uuid
import json
import re
import urllib.request
import urllib.parse
import random
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
import io


def open_websocket_connection(server_address='127.0.0.1:8188'):
  client_id=str(uuid.uuid4())
  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
  return ws, server_address, client_id

def queue_prompt(prompt, client_id, server_address):
  p = {"prompt": prompt, "client_id": client_id}
  headers = {'Content-Type': 'application/json'}
  data = json.dumps(p).encode('utf-8')
  req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
  return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id, server_address):
  with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
      return json.loads(response.read())
  
def get_image(filename, subfolder, folder_type, server_address):
  data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
  url_values = urllib.parse.urlencode(data)
  with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
      return response.read()
  
def upload_image(input_path, name, server_address, image_type="input", overwrite=False):
  with open(input_path, 'rb') as file:
    multipart_data = MultipartEncoder(
      fields= {
        'image': (name, file, 'image/png'),
        'type': image_type,
        'overwrite': str(overwrite).lower()
      }
    )

    data = multipart_data
    headers = { 'Content-Type': multipart_data.content_type }
    request = urllib.request.Request("http://{}/upload/image".format(server_address), data=data, headers=headers)
    with urllib.request.urlopen(request) as response:
      return response.read()
    
def load_workflow(workflow_path):
  try:
      with open(workflow_path, 'r') as file:
          workflow = json.load(file)
          return json.dumps(workflow)
  except FileNotFoundError:
      print(f"The file {workflow_path} was not found.")
      return None
  except json.JSONDecodeError:
      print(f"The file {workflow_path} contains invalid JSON.")
      return None
  


def modify_workflow_prompt(workflow_json, positive_prompt, negative_prompt=''):
    """Modify prompts in a workflow JSON.

    Args:
        workflow_json: JSON string of the workflow
        positive_prompt: Positive prompt text
        negative_prompt: Negative prompt text (optional)

    Returns:
        Modified workflow as dict
    """
    prompt = json.loads(workflow_json)
    id_to_class_type = {id: details['class_type'] for id, details in prompt.items()}

    # Find KSampler or similar sampler nodes
    sampler_nodes = [key for key, value in id_to_class_type.items()
                    if value in ['KSampler', 'SamplerCustom']]

    if not sampler_nodes:
        raise ValueError("No sampler node found in workflow")

    # Use first sampler node
    sampler = sampler_nodes[0]

    # Set random seed
    if 'seed' in prompt.get(sampler, {}).get('inputs', {}):
        prompt[sampler]['inputs']['seed'] = random.randint(10**14, 10**15 - 1)

    # Update positive prompt
    if 'positive' in prompt[sampler]['inputs']:
        positive_input = prompt[sampler]['inputs']['positive']
        if isinstance(positive_input, list) and len(positive_input) > 0:
            positive_node_id = positive_input[0]
            if positive_node_id in prompt and 'text' in prompt[positive_node_id]['inputs']:
                prompt[positive_node_id]['inputs']['text'] = positive_prompt

    # Update negative prompt
    if negative_prompt and 'negative' in prompt[sampler]['inputs']:
        negative_input = prompt[sampler]['inputs']['negative']
        if isinstance(negative_input, list) and len(negative_input) > 0:
            negative_node_id = negative_input[0]
            if negative_node_id in prompt and 'text' in prompt[negative_node_id]['inputs']:
                prompt[negative_node_id]['inputs']['text'] = negative_prompt

    return prompt


def prompt_to_image(workflow, positive_prompt, negative_prompt='', save_previews=False, return_images=False, server_address='127.0.0.1:8188'):
    """Generate image from prompt using image workflow.

    Args:
        workflow: JSON string of the workflow
        positive_prompt: Positive prompt text
        negative_prompt: Negative prompt text (optional)
        save_previews: Whether to save preview images
        return_images: Whether to return image bytes instead of saving
        server_address: ComfyUI server address

    Returns:
        List of image bytes if return_images=True, otherwise None
    """
    prompt = modify_workflow_prompt(workflow, positive_prompt, negative_prompt)
    return generate_image_by_prompt(prompt, './output/', save_previews, return_images, server_address)


def generate_by_prompt(prompt, server_address='127.0.0.1:8188', output_type='image', save_previews=False, return_data=False, output_path=None):
    """Generic function to generate content (image/video) from prompt.

    Args:
        prompt: Workflow prompt dict
        server_address: ComfyUI server address
        output_type: Type of output ('image' or 'video')
        save_previews: Whether to save previews
        return_data: Whether to return data bytes instead of saving
        output_path: Output directory path

    Returns:
        List of data bytes if return_data=True, otherwise None
    """
    ws = None
    try:
        ws, server_address, client_id = open_websocket_connection(server_address)
        prompt_id = queue_prompt(prompt, client_id, server_address)['prompt_id']
        track_progress(prompt, ws, prompt_id)

        if output_type == 'image':
            outputs = get_images(prompt_id, server_address, save_previews)
            if return_data:
                # Return image bytes
                data_bytes_list = []
                for item in outputs:
                    if 'image_data' in item:
                        data_bytes_list.append(item['image_data'])
                return data_bytes_list
            else:
                # Save to disk
                if output_path is None:
                    output_path = './output/'
                save_image(outputs, output_path, save_previews)
                return None
        elif output_type == 'video':
            outputs = get_videos(prompt_id, server_address)
            if return_data:
                # Return video bytes
                data_bytes_list = []
                for item in outputs:
                    if 'video_data' in item:
                        data_bytes_list.append(item['video_data'])
                return data_bytes_list
            else:
                # Save to disk
                if output_path is None:
                    output_path = './output/'
                save_video(outputs, output_path)
                return None
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

    finally:
        if ws:
            ws.close()


def generate_image_by_prompt(prompt, output_path=None, save_previews=False, return_images=False, server_address='127.0.0.1:8188'):
    """Generate image from prompt (legacy function for backward compatibility).

    Args:
        prompt: Workflow prompt dict
        output_path: Output directory path
        save_previews: Whether to save preview images
        return_images: Whether to return image bytes instead of saving
        server_address: ComfyUI server address

    Returns:
        List of image bytes if return_images=True, otherwise None
    """
    return generate_by_prompt(
        prompt=prompt,
        server_address=server_address,
        output_type='image',
        save_previews=save_previews,
        return_data=return_images,
        output_path=output_path
    )

def track_progress(prompt, ws, prompt_id):
  node_ids = list(prompt.keys())
  finished_nodes = []

  while True:
      out = ws.recv()
      if isinstance(out, str):
          message = json.loads(out)
          if message['type'] == 'progress':
              data = message['data']
              current_step = data['value']
              print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
          if message['type'] == 'execution_cached':
              data = message['data']
              for itm in data['nodes']:
                  if itm not in finished_nodes:
                      finished_nodes.append(itm)
                      print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
          if message['type'] == 'executing':
              data = message['data']
              if data['node'] not in finished_nodes:
                  finished_nodes.append(data['node'])
                  print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')

              if data['node'] is None and data['prompt_id'] == prompt_id:
                  break #Execution is done
      else:
          continue
  return


def get_images(prompt_id, server_address, allow_preview = False):
  output_images = []

  history = get_history(prompt_id, server_address)[prompt_id]
  for node_id in history['outputs']:
      node_output = history['outputs'][node_id]
      output_data = {}
      if 'images' in node_output:
          for image in node_output['images']:
              if allow_preview and image['type'] == 'temp':
                  preview_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                  output_data['image_data'] = preview_data
              if image['type'] == 'output':
                  image_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                  output_data['image_data'] = image_data
      output_data['file_name'] = image['filename']
      output_data['type'] = image['type']
      output_images.append(output_data)

  return output_images

def get_videos(prompt_id, server_address):
    """Retrieve generated videos from ComfyUI history.

    Args:
        prompt_id: Prompt ID from queue_prompt
        server_address: ComfyUI server address

    Returns:
        List of video data dictionaries
    """
    output_videos = []

    history = get_history(prompt_id, server_address)[prompt_id]

    # Debug: log the history structure
    import logging
    logging.info(f"Getting videos for prompt {prompt_id}")
    logging.info(f"History keys: {list(history.keys())}")

    if 'outputs' in history:
        logging.info(f"Output nodes: {list(history['outputs'].keys())}")
        for node_id, node_output in history['outputs'].items():
            logging.info(f"Node {node_id} class type: {node_output.get('_meta', {}).get('title', 'Unknown')}")
            logging.info(f"Node {node_id} keys: {list(node_output.keys())}")

            # Log all data for debugging
            for key, value in node_output.items():
                if key not in ['_meta']:  # Skip metadata
                    if isinstance(value, dict):
                        logging.info(f"  {key} keys: {list(value.keys())}")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, list):
                                logging.info(f"    {subkey}: list with {len(subvalue)} items")
                                if subvalue and isinstance(subvalue[0], dict):
                                    logging.info(f"      First item keys: {list(subvalue[0].keys())}")
                    elif isinstance(value, list):
                        logging.info(f"  {key}: list with {len(value)} items")
                        if value and isinstance(value[0], dict):
                            logging.info(f"    First item keys: {list(value[0].keys())}")

    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        output_data = {}

        # Check for animated outputs (videos) - various possible formats
        found_video = False

        # Format 1: animated_output array
        if 'animated_output' in node_output:
            for video in node_output['animated_output']:
                if video.get('type') == 'output':
                    try:
                        video_data = get_image(video['filename'], video.get('subfolder', ''), video['type'], server_address)
                        output_data['video_data'] = video_data
                        output_data['file_name'] = video['filename']
                        output_data['type'] = video['type']
                        output_videos.append(output_data.copy())
                        found_video = True
                        logging.info(f"Found video in animated_output: {video['filename']}")
                    except Exception as e:
                        logging.error(f"Failed to get video from animated_output: {e}")

        # Format 2: outputs.animated array (SaveAnimatedWEBP)
        if 'outputs' in node_output and 'animated' in node_output['outputs']:
            for video in node_output['outputs']['animated']:
                if video.get('type') == 'output':
                    try:
                        video_data = get_image(video['filename'], video.get('subfolder', ''), video['type'], server_address)
                        output_data['video_data'] = video_data
                        output_data['file_name'] = video['filename']
                        output_data['type'] = video['type']
                        output_videos.append(output_data.copy())
                        found_video = True
                        logging.info(f"Found video in outputs.animated: {video['filename']}")
                    except Exception as e:
                        logging.error(f"Failed to get video from outputs.animated: {e}")

        # Format 3: direct 'images' array
        if 'images' in node_output:
            for video in node_output['images']:
                if video.get('type') == 'output':
                    # Check if this looks like a video file
                    filename = video['filename'].lower()
                    if any(ext in filename for ext in ['.webp', '.mp4', '.avi', '.mov', '.mkv']):
                        try:
                            video_data = get_image(video['filename'], video.get('subfolder', ''), video['type'], server_address)
                            output_data['video_data'] = video_data
                            output_data['file_name'] = video['filename']
                            output_data['type'] = video['type']
                            output_videos.append(output_data.copy())
                            found_video = True
                            logging.info(f"Found video in images array: {video['filename']}")
                        except Exception as e:
                            logging.error(f"Failed to get video from images array: {e}")

        # Format 4: Check for any output that looks like a video file
        if not found_video:
            # Check all keys for potential video data
            for key, value in node_output.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'filename' in item:
                            filename = item['filename'].lower()
                            if any(ext in filename for ext in ['.webp', '.mp4', '.avi', '.mov', '.mkv']):
                                if item.get('type') == 'output':
                                    try:
                                        video_data = get_image(item['filename'], item.get('subfolder', ''), item['type'], server_address)
                                        output_data['video_data'] = video_data
                                        output_data['file_name'] = item['filename']
                                        output_data['type'] = item['type']
                                        output_videos.append(output_data.copy())
                                        logging.info(f"Found video in {key} array: {item['filename']}")
                                    except Exception as e:
                                        logging.error(f"Failed to get video from {key} array: {e}")

    logging.info(f"Found {len(output_videos)} videos for prompt {prompt_id}")
    return output_videos


def save_image(images, output_path, save_previews):
    """Save images to disk.

    Args:
        images: List of image data dictionaries
        output_path: Output directory path
        save_previews: Whether to save preview images
    """
    for itm in images:
        directory = os.path.join(output_path, 'temp/') if itm['type'] == 'temp' and save_previews else output_path
        os.makedirs(directory, exist_ok=True)
        try:
            image = Image.open(io.BytesIO(itm['image_data']))
            image.save(os.path.join(directory, itm['file_name']))
        except Exception as e:
            print(f"Failed to save image {itm['file_name']}: {e}")


def save_video(videos, output_path):
    """Save videos to disk.

    Args:
        videos: List of video data dictionaries
        output_path: Output directory path
    """
    for itm in videos:
        os.makedirs(output_path, exist_ok=True)
        try:
            file_path = os.path.join(output_path, itm['file_name'])
            with open(file_path, 'wb') as f:
                f.write(itm['video_data'])
            print(f"Saved video: {file_path}")
        except Exception as e:
            print(f"Failed to save video {itm['file_name']}: {e}")


def image_to_base64(image_bytes):
  """Convert image bytes to base64 encoded string."""
  import base64
  return base64.b64encode(image_bytes).decode('utf-8')


def update_workflow_size(workflow_json, size_str):
  """Update the size parameters in a workflow JSON string.

  Args:
    workflow_json: JSON string of the workflow
    size_str: Size string in format "WIDTHxHEIGHT" (e.g., "256x256", "512x512")

  Returns:
    Modified workflow JSON string

  Raises:
    ValueError: If size_str is invalid or EmptySD3LatentImage node not found
  """
  # Parse size string
  match = re.match(r'^(\d+)x(\d+)$', size_str)
  if not match:
    raise ValueError(f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT'")

  width = int(match.group(1))
  height = int(match.group(2))

  # Parse workflow JSON
  workflow = json.loads(workflow_json)

  # Find EmptySD3LatentImage node
  empty_latent_nodes = [
    node_id for node_id, node_data in workflow.items()
    if node_data.get('class_type') == 'EmptySD3LatentImage'
  ]

  if not empty_latent_nodes:
    raise ValueError("No EmptySD3LatentImage node found in workflow")

  # Update all EmptySD3LatentImage nodes (typically there's just one)
  for node_id in empty_latent_nodes:
    workflow[node_id]['inputs']['width'] = width
    workflow[node_id]['inputs']['height'] = height

  # Return modified workflow as JSON string
  return json.dumps(workflow)


def generate_image_in_memory(prompt_text, workflow_path="z_image_turbo.json", server_address="127.0.0.1:8188", size=None):
    """Generate image from prompt and return image bytes.

    Args:
        prompt_text: Text prompt for image generation
        workflow_path: Path to workflow JSON file
        server_address: ComfyUI server address
        size: Optional size string in format "WIDTHxHEIGHT" (e.g., "256x256")

    Returns:
        Image bytes
    """
    workflow = load_workflow(workflow_path)
    if workflow is None:
        raise ValueError(f"Failed to load workflow from {workflow_path}")

    # Update workflow size if specified
    if size is not None:
        workflow = update_workflow_size(workflow, size)

    image_bytes_list = prompt_to_image(
        workflow,
        prompt_text,
        negative_prompt='',
        save_previews=False,
        return_images=True,
        server_address=server_address
    )

    if image_bytes_list and len(image_bytes_list) > 0:
        return image_bytes_list[0]  # Return first image
    else:
        raise RuntimeError("No images generated")


def prompt_to_video(workflow, positive_prompt, negative_prompt='', return_videos=False, server_address='127.0.0.1:8188'):
    """Generate video from prompt using video workflow.

    Args:
        workflow: JSON string of the workflow
        positive_prompt: Positive prompt text
        negative_prompt: Negative prompt text (optional)
        return_videos: Whether to return video bytes instead of saving
        server_address: ComfyUI server address

    Returns:
        List of video bytes if return_videos=True, otherwise None
    """
    prompt = modify_workflow_prompt(workflow, positive_prompt, negative_prompt)
    return generate_by_prompt(
        prompt=prompt,
        server_address=server_address,
        output_type='video',
        save_previews=False,
        return_data=return_videos,
        output_path='./output/'
    )


def generate_video_in_memory(prompt_text, workflow_path="ltx-video-t2v.json", server_address="127.0.0.1:8188"):
    """Generate video from prompt and return video bytes.

    Args:
        prompt_text: Text prompt for video generation
        workflow_path: Path to workflow JSON file
        server_address: ComfyUI server address

    Returns:
        Video bytes
    """
    import logging
    logging.info(f"[generate_video_in_memory] Starting video generation")
    logging.info(f"  Prompt: {prompt_text[:50]}...")
    logging.info(f"  Workflow: {workflow_path}")
    logging.info(f"  Server: {server_address}")

    workflow = load_workflow(workflow_path)
    if workflow is None:
        logging.error(f"[generate_video_in_memory] Failed to load workflow from {workflow_path}")
        raise ValueError(f"Failed to load workflow from {workflow_path}")

    logging.info(f"[generate_video_in_memory] Workflow loaded successfully")

    try:
        video_bytes_list = prompt_to_video(
            workflow,
            prompt_text,
            negative_prompt='',
            return_videos=True,
            server_address=server_address
        )

        logging.info(f"[generate_video_in_memory] prompt_to_video returned {len(video_bytes_list) if video_bytes_list else 0} videos")

        if video_bytes_list and len(video_bytes_list) > 0:
            video_size = len(video_bytes_list[0])
            logging.info(f"[generate_video_in_memory] Video generated successfully, size: {video_size} bytes")
            return video_bytes_list[0]  # Return first video
        else:
            logging.error(f"[generate_video_in_memory] No videos generated. video_bytes_list: {video_bytes_list}")
            raise RuntimeError("No videos generated")

    except Exception as e:
        logging.error(f"[generate_video_in_memory] Error: {e}")
        raise


def video_to_base64(video_bytes):
    """Convert video bytes to base64 encoded string.

    Args:
        video_bytes: Video bytes

    Returns:
        Base64 encoded string
    """
    import base64
    return base64.b64encode(video_bytes).decode('utf-8')


# Task management for async operations
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """Task status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task data class for tracking generation tasks."""
    task_id: str
    task_type: str  # 'image' or 'video'
    prompt: str
    workflow_path: str
    status: TaskStatus = TaskStatus.QUEUED
    progress: int = 0
    result: Optional[bytes] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completion_event: threading.Event = field(default_factory=threading.Event)


class TaskManager:
    """Manages generation tasks with in-memory storage."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_tasks, daemon=True)
        self.cleanup_thread.start()

    def create_task(self, task_type: str, prompt: str, workflow_path: str) -> str:
        """Create a new task.

        Args:
            task_type: Type of task ('image' or 'video')
            prompt: Text prompt
            workflow_path: Path to workflow JSON file

        Returns:
            Task ID
        """
        task_id = f"{task_type}_{int(time.time())}_{hash(prompt) % 10000:04d}"
        task = Task(
            task_id=task_id,
            task_type=task_type,
            prompt=prompt,
            workflow_path=workflow_path
        )

        with self.lock:
            self.tasks[task_id] = task

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task object or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus, progress: int = 0,
                          result: Optional[bytes] = None, error: Optional[str] = None,
                          prompt: str = "", workflow_path: str = ""):
        """Update task status. Creates task if it doesn't exist.

        Args:
            task_id: Task ID
            status: New status
            progress: Progress percentage (0-100)
            result: Result data (bytes)
            error: Error message
            prompt: Prompt text (required if creating new task)
            workflow_path: Workflow path (required if creating new task)
        """
        with self.lock:
            if task_id not in self.tasks:
                # Create new task if it doesn't exist
                task = Task(
                    task_id=task_id,
                    task_type='video' if 'vid_' in task_id else 'image',
                    prompt=prompt,
                    workflow_path=workflow_path,
                    status=status,
                    progress=progress,
                    result=result,
                    error=error,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                self.tasks[task_id] = task
                # Use print instead of app.logger since we don't have app context here
                print(f"[TaskManager] Created new task: {task_id}")
            else:
                task = self.tasks[task_id]
                task.status = status
                task.progress = progress
                task.updated_at = time.time()

                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error

            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completion_event.set()

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for task completion.

        Args:
            task_id: Task ID
            timeout: Timeout in seconds

        Returns:
            True if task completed, False if timeout
        """
        task = self.get_task(task_id)
        if task is None:
            return False

        return task.completion_event.wait(timeout=timeout)

    def list_tasks(self, limit: int = 100, offset: int = 0) -> list:
        """List tasks with pagination.

        Args:
            limit: Maximum number of tasks to return
            offset: Offset for pagination

        Returns:
            List of task dictionaries
        """
        with self.lock:
            tasks = list(self.tasks.values())
            tasks.sort(key=lambda x: x.created_at, reverse=True)
            return [
                {
                    'id': task.task_id,
                    'type': task.task_type,
                    'prompt': task.prompt,
                    'status': task.status.value,
                    'progress': task.progress,
                    'created_at': task.created_at,
                    'updated_at': task.updated_at
                }
                for task in tasks[offset:offset + limit]
            ]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID

        Returns:
            True if task was deleted, False if not found
        """
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False

    def _cleanup_old_tasks(self):
        """Clean up old tasks to prevent memory leak."""
        while True:
            time.sleep(3600)  # Run every hour
            with self.lock:
                current_time = time.time()
                tasks_to_delete = []
                for task_id, task in self.tasks.items():
                    # Delete tasks older than 24 hours
                    if current_time - task.updated_at > 86400:
                        tasks_to_delete.append(task_id)

                for task_id in tasks_to_delete:
                    del self.tasks[task_id]

                if tasks_to_delete:
                    print(f"Cleaned up {len(tasks_to_delete)} old tasks")


# Global task manager instance
task_manager = TaskManager()


def generate_image_async(prompt_text: str, workflow_path: str = "z_image_turbo.json",
                        server_address: str = "127.0.0.1:8188", size: Optional[str] = None) -> str:
    """Generate image asynchronously.

    Args:
        prompt_text: Text prompt for image generation
        workflow_path: Path to workflow JSON file
        server_address: ComfyUI server address
        size: Optional size string

    Returns:
        Task ID
    """
    task_id = task_manager.create_task('image', prompt_text, workflow_path)

    def worker():
        try:
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 10)
            image_bytes = generate_image_in_memory(prompt_text, workflow_path, server_address, size)
            task_manager.update_task_status(task_id, TaskStatus.COMPLETED, 100, result=image_bytes)
        except Exception as e:
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error=str(e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    return task_id


def generate_video_async(prompt_text: str, workflow_path: str = "ltx-video-t2v.json",
                        server_address: str = "127.0.0.1:8188") -> str:
    """Generate video asynchronously.

    Args:
        prompt_text: Text prompt for video generation
        workflow_path: Path to workflow JSON file
        server_address: ComfyUI server address

    Returns:
        Task ID
    """
    task_id = task_manager.create_task('video', prompt_text, workflow_path)

    def worker():
        try:
            task_manager.update_task_status(task_id, TaskStatus.PROCESSING, 10)
            video_bytes = generate_video_in_memory(prompt_text, workflow_path, server_address)
            task_manager.update_task_status(task_id, TaskStatus.COMPLETED, 100, result=video_bytes)
        except Exception as e:
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error=str(e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    return task_id


if __name__ == "__main__":
    # Test both image and video generation
    print("Testing image generation...")
    workflow = load_workflow("z_image_turbo.json")
    if workflow:
        prompt_to_image(workflow, "a bottle of water")
        print("Image generation test completed")

    print("\nTesting video generation...")
    video_workflow = load_workflow("ltx-video-t2v.json")
    if video_workflow:
        try:
            prompt_to_video(video_workflow, "a beautiful sunset")
            print("Video generation test completed")
        except Exception as e:
            print(f"Video generation test failed (may need ComfyUI setup): {e}")
