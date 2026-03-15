#!/usr/bin/env python3
"""
Model-based backend router for ComfyUI API.
Routes requests to different backend clusters based on model type.
"""

import os
import json
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple
import logging


class ModelBackendConfig:
    """Configuration for a model's backend cluster."""

    def __init__(self, model_name: str, config_data: Dict[str, Any]):
        self.model_name = model_name
        self.description = config_data.get('description', '')
        self.workflow = config_data.get('workflow', '')
        self.i2v_workflow = config_data.get('i2v_workflow', '')
        self.backend_addresses = config_data.get('backends', [])
        self.fixed_size = config_data.get('fixed_size', '256x256')
        self.default_model = config_data.get('default_model', model_name)

        # Video-specific parameters
        self.fixed_seconds = config_data.get('fixed_seconds', 4)

        # Initialize backends
        self.backends = []
        self.backend_lock = threading.Lock()
        self.jobs = {}

        for idx, address in enumerate(self.backend_addresses):
            backend = {
                'id': f"{model_name}_{idx}",
                'address': address,
                'queue': queue.Queue(),
                'queue_size': 0,
                'is_processing': False,
                'lock': threading.Lock(),
            }
            self.backends.append(backend)

        # Start worker threads
        self.worker_threads = []
        for backend in self.backends:
            thread = threading.Thread(
                target=self._process_backend_jobs,
                args=(backend,),
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)

    def _process_backend_jobs(self, backend: Dict[str, Any]):
        """Process jobs for a specific backend."""
        while True:
            try:
                job_data = backend['queue'].get(timeout=1)
                self._process_job(backend, job_data)
                backend['queue'].task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing job on backend {backend['id']}: {e}")
                time.sleep(5)

    def _process_job(self, backend: Dict[str, Any], job_data: Dict[str, Any]):
        """Process a single job."""
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _process_job")


class ModelRouter:
    """Routes requests to appropriate model backend clusters."""

    def __init__(self, config_path: str = "backend_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.model_configs: Dict[str, ModelBackendConfig] = {}
        self.model_aliases = self.config.get('model_aliases', {})
        self.default_model = self.config.get('default_model', 'dall-e-3')
        self.lock = threading.Lock()

        # Initialize model configurations
        self._init_model_configs()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def _init_model_configs(self):
        """Initialize model backend configurations."""
        model_backends = self.config.get('model_backends', {})
        for model_name, config_data in model_backends.items():
            try:
                # Determine config class based on model type
                if 'sora' in model_name.lower() or 'video' in config_data.get('description', '').lower():
                    config = VideoModelBackendConfig(model_name, config_data)
                else:
                    config = ImageModelBackendConfig(model_name, config_data)

                self.model_configs[model_name] = config
                logging.info(f"Initialized backend config for model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to initialize config for model {model_name}: {e}")

    def resolve_model(self, requested_model: str) -> str:
        """Resolve model name, handling aliases."""
        # Check aliases first
        resolved = self.model_aliases.get(requested_model, requested_model)

        # Check if resolved model exists
        if resolved in self.model_configs:
            return resolved

        # If not found, try to find a matching model
        for model_name in self.model_configs:
            if requested_model in model_name or model_name in requested_model:
                return model_name

        # Return default model
        logging.warning(f"Model '{requested_model}' not found, using default: {self.default_model}")
        return self.default_model

    def get_model_config(self, model_name: str) -> Optional[ModelBackendConfig]:
        """Get configuration for a specific model."""
        resolved_name = self.resolve_model(model_name)
        return self.model_configs.get(resolved_name)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        for model_name, config in self.model_configs.items():
            models.append({
                'name': model_name,
                'description': config.description,
                'workflow': config.workflow,
                'backends': len(config.backends),
                'default_model': config.default_model
            })
        return models

    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics for all backends."""
        stats = {}
        for model_name, config in self.model_configs.items():
            model_stats = []
            for backend in config.backends:
                model_stats.append({
                    'id': backend['id'],
                    'address': backend['address'],
                    'queue_size': backend['queue_size'],
                    'is_processing': backend['is_processing']
                })
            stats[model_name] = {
                'total_backends': len(config.backends),
                'backends': model_stats
            }
        return stats


class ImageModelBackendConfig(ModelBackendConfig):
    """Backend configuration for image generation models."""

    def __init__(self, model_name: str, config_data: Dict[str, Any]):
        super().__init__(model_name, config_data)
        # Import here to avoid circular imports
        from server import generate_image_in_memory
        self.generate_image_in_memory = generate_image_in_memory

    def _process_job(self, backend: Dict[str, Any], job_data: Dict[str, Any]):
        """Process an image generation job."""
        try:
            request_data = job_data['request_data']
            prompt = request_data.get('prompt', '')
            size = request_data.get('size', self.fixed_size)

            # Update job status
            with self.backend_lock:
                job_data['status'] = 'processing'
                backend['is_processing'] = True

            logging.info(f"Processing image job {job_data['job_id']} on backend {backend['id']}")

            # Generate image
            image_bytes = self.generate_image_in_memory(
                prompt_text=prompt,
                workflow_path=self.workflow,
                server_address=backend['address'],
                size=size if size != self.fixed_size else None
            )

            # Update job result
            with self.backend_lock:
                job_data['status'] = 'completed'
                job_data['result'] = image_bytes
                job_data['completion_event'].set()

        except Exception as e:
            # Update job error
            with self.backend_lock:
                job_data['status'] = 'failed'
                job_data['error'] = str(e)
                job_data['completion_event'].set()
            logging.error(f"Image generation failed for job {job_data['job_id']}: {e}")

        finally:
            # Update backend status
            with self.backend_lock:
                backend['is_processing'] = False
                backend['queue_size'] = backend['queue'].qsize()

    def add_image_request(self, request_data: Dict[str, Any]) -> Tuple[str, str]:
        """Add an image generation request."""
        with self.backend_lock:
            if not self.backends:
                raise RuntimeError(f"No backends available for model {self.model_name}")

            # Find backend with smallest queue
            selected_backend = min(self.backends, key=lambda b: b['queue_size'])

            # Create job ID
            job_id = f"img_{int(time.time())}_{hash(str(request_data)) % 10000:04d}"

            # Create job data
            job_data = {
                'job_id': job_id,
                'request_data': request_data,
                'status': 'queued',
                'result': None,
                'error': None,
                'created_at': time.time(),
                'completion_event': threading.Event(),
            }

            # Store job
            self.jobs[job_id] = job_data

            # Add to backend queue
            with selected_backend['lock']:
                selected_backend['queue'].put(job_data)
                selected_backend['queue_size'] += 1

            return selected_backend['id'], job_id

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """Wait for a job to complete."""
        with self.backend_lock:
            if job_id not in self.jobs:
                raise KeyError(f"Job {job_id} not found")
            job_data = self.jobs[job_id]

        # Wait for completion
        if not job_data['completion_event'].wait(timeout=timeout):
            raise TimeoutError(f"Job {job_id} timed out")

        # Get result
        with self.backend_lock:
            status = job_data['status']
            result = job_data['result'] if status == 'completed' else job_data['error']

            # Clean up old jobs
            self._cleanup_old_jobs()

            return status, result

    def _cleanup_old_jobs(self):
        """Clean up old jobs to prevent memory leak."""
        current_time = time.time()
        jobs_to_delete = []
        for job_id, job_data in self.jobs.items():
            if job_data['status'] in ['completed', 'failed'] and current_time - job_data['created_at'] > 3600:
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del self.jobs[job_id]


class VideoModelBackendConfig(ModelBackendConfig):
    """Backend configuration for video generation models."""

    def __init__(self, model_name: str, config_data: Dict[str, Any]):
        super().__init__(model_name, config_data)
        # Import here to avoid circular imports
        from server import generate_video_in_memory
        self.generate_video_in_memory = generate_video_in_memory

    def _process_job(self, backend: Dict[str, Any], job_data: Dict[str, Any]):
        """Process a video generation job."""
        try:
            request_data = job_data['request_data']
            prompt = request_data.get('prompt', '')
            input_image_base64 = request_data.get('input_image_base64')

            # Update job status
            with self.backend_lock:
                job_data['status'] = 'in_progress'  # OpenAI API uses 'in_progress'
                backend['is_processing'] = True

            logging.info(f"Processing video job {job_data['job_id']} on backend {backend['id']}")

            # Generate video
            workflow_path = self.i2v_workflow if (input_image_base64 and self.i2v_workflow) else self.workflow
            video_bytes = self.generate_video_in_memory(
                prompt_text=prompt,
                workflow_path=workflow_path,
                server_address=backend['address'],
                task_id=job_data['job_id'],  # Pass job ID for cleanup tracking
                input_image_base64=input_image_base64,
            )

            # Update job result
            with self.backend_lock:
                job_data['status'] = 'completed'
                job_data['result'] = video_bytes
                job_data['completion_event'].set()

        except Exception as e:
            # Update job error
            with self.backend_lock:
                job_data['status'] = 'failed'
                job_data['error'] = str(e)
                job_data['completion_event'].set()
            logging.error(f"Video generation failed for job {job_data['job_id']}: {e}")

        finally:
            # Update backend status
            with self.backend_lock:
                backend['is_processing'] = False
                backend['queue_size'] = backend['queue'].qsize()

    def add_video_request(self, request_data: Dict[str, Any]) -> Tuple[str, str]:
        """Add a video generation request."""
        with self.backend_lock:
            if not self.backends:
                raise RuntimeError(f"No backends available for model {self.model_name}")

            # Find backend with smallest queue
            selected_backend = min(self.backends, key=lambda b: b['queue_size'])

            # Create job ID
            job_id = f"vid_{int(time.time())}_{hash(str(request_data)) % 10000:04d}"

            # Create job data
            job_data = {
                'job_id': job_id,
                'request_data': request_data,
                'status': 'queued',
                'result': None,
                'error': None,
                'created_at': time.time(),
                'completion_event': threading.Event(),
            }

            # Store job
            self.jobs[job_id] = job_data

            # Add to backend queue
            with selected_backend['lock']:
                selected_backend['queue'].put(job_data)
                selected_backend['queue_size'] += 1

            return selected_backend['id'], job_id

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Tuple[str, Any]:
        """Wait for a job to complete."""
        with self.backend_lock:
            if job_id not in self.jobs:
                raise KeyError(f"Job {job_id} not found")
            job_data = self.jobs[job_id]

        # Wait for completion
        if not job_data['completion_event'].wait(timeout=timeout):
            raise TimeoutError(f"Job {job_id} timed out")

        # Get result
        with self.backend_lock:
            status = job_data['status']
            result = job_data['result'] if status == 'completed' else job_data['error']

            # Clean up old jobs
            self._cleanup_old_jobs()

            return status, result

    def _cleanup_old_jobs(self):
        """Clean up old jobs to prevent memory leak."""
        current_time = time.time()
        jobs_to_delete = []
        for job_id, job_data in self.jobs.items():
            if job_data['status'] in ['completed', 'failed'] and current_time - job_data['created_at'] > 3600:
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            del self.jobs[job_id]