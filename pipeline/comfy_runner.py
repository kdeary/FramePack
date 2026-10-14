"""Runs ComfyUI headless in its own process and pushes single-image edit jobs through its HTTP API.

The edit model (Qwen-Image-Edit) needs newer diffusers/transformers than FramePack pins, so it
lives in a separate venv with ComfyUI. Running it as a subprocess also means closing it hands every
byte of VRAM and RAM back to FramePack before the video stage.

Workflows are ComfyUI "API format" JSON files. The nodes this module fills in are found by their
title (`_meta.title`), not their numeric id, so any workflow tagged like this drops in unchanged:

    IN_IMAGE    LoadImage node              -> inputs.image
    IN_PROMPT   the positive text encoder   -> inputs.prompt
    IN_SEED     KSampler or RandomNoise     -> inputs.seed / inputs.noise_seed
    OUT_IMAGE   SaveImage node whose output is returned

Stdlib only, so the workflow helpers can be unit-tested without ComfyUI.
"""

import copy
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

TAG_IMAGE = 'IN_IMAGE'
TAG_PROMPT = 'IN_PROMPT'
TAG_SEED = 'IN_SEED'
TAG_OUTPUT = 'OUT_IMAGE'
SEED_INPUTS = ('seed', 'noise_seed')


class ComfyError(RuntimeError):
    pass


def load_workflow(path):
    with open(path, encoding='utf-8') as f:
        workflow = json.load(f)
    for tag in (TAG_IMAGE, TAG_PROMPT, TAG_SEED, TAG_OUTPUT):
        find_node(workflow, tag)
    return workflow


def find_node(workflow, title):
    """Returns the id of the one node titled `title`."""
    matches = [node_id for node_id, node in workflow.items()
               if isinstance(node, dict) and node.get('_meta', {}).get('title') == title]
    if len(matches) != 1:
        found = 'no node' if not matches else f'{len(matches)} nodes'
        raise ComfyError(f'Workflow needs exactly one node titled "{title}", found {found}.')
    return matches[0]


def apply_inputs(workflow, image_name, prompt, seed):
    """Returns a copy of the workflow with the image, prompt and seed filled in."""
    wf = copy.deepcopy(workflow)
    wf[find_node(wf, TAG_IMAGE)]['inputs']['image'] = image_name
    wf[find_node(wf, TAG_PROMPT)]['inputs']['prompt'] = prompt

    seed_inputs = wf[find_node(wf, TAG_SEED)]['inputs']
    seed_keys = [key for key in SEED_INPUTS if key in seed_inputs]
    if not seed_keys:
        raise ComfyError(f'The "{TAG_SEED}" node has no {" or ".join(SEED_INPUTS)} input.')
    seed_inputs[seed_keys[0]] = int(seed)
    return wf


def _encode_multipart(fields, files):
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key, (filename, data) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n'.encode() + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


class ComfyServer:
    """A ComfyUI process on localhost. Use as a context manager, or call start() and close()."""

    def __init__(self, comfy_dir, python=None, port=8188, extra_args=(), log_path=None, startup_timeout=600):
        self.comfy_dir = comfy_dir
        self.python = python or sys.executable
        self.port = port
        self.extra_args = list(extra_args)
        self.log_path = log_path or os.path.join(comfy_dir, 'comfy_server.log')
        self.startup_timeout = startup_timeout
        self.base_url = f'http://127.0.0.1:{port}'
        self.client_id = uuid.uuid4().hex
        self.process = None
        self._log_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        if self.is_running():
            return
        main_py = os.path.join(self.comfy_dir, 'main.py')
        if not os.path.exists(main_py):
            raise ComfyError(f'ComfyUI not found at {self.comfy_dir}. Run the Colab setup cell, or pass --comfy_dir.')

        cmd = [self.python, main_py, '--listen', '127.0.0.1', '--port', str(self.port),
               '--disable-auto-launch', *self.extra_args]
        print('Starting ComfyUI:', ' '.join(cmd))
        self._log_file = open(self.log_path, 'ab')
        self.process = subprocess.Popen(cmd, cwd=self.comfy_dir, stdout=self._log_file, stderr=subprocess.STDOUT)

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise ComfyError(f'ComfyUI exited during startup (code {self.process.returncode}).\n{self.log_tail()}')
            try:
                self._get('/system_stats', timeout=5)
                print('ComfyUI is up.')
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(1)
        self.close()
        raise ComfyError(f'ComfyUI did not start within {self.startup_timeout}s.\n{self.log_tail()}')

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            print('ComfyUI stopped.')
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def log_tail(self, lines=30):
        try:
            with open(self.log_path, 'rb') as f:
                return b''.join(f.readlines()[-lines:]).decode('utf-8', 'replace')
        except OSError:
            return ''

    def _get(self, path, timeout=30):
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as response:
            return response.read()

    def _post(self, path, body, content_type, timeout=60):
        request = urllib.request.Request(self.base_url + path, data=body, headers={'Content-Type': content_type})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            # ComfyUI explains validation errors in the response body.
            raise ComfyError(f'ComfyUI rejected {path}: {e.read().decode("utf-8", "replace")}') from e

    def upload_image(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        # A unique upload name keeps concurrent or repeated jobs from reading each other's input.
        filename = f'{uuid.uuid4().hex[:8]}_{os.path.basename(path)}'
        body, content_type = _encode_multipart({'type': 'input', 'overwrite': 'true'}, {'image': (filename, data)})
        result = json.loads(self._post('/upload/image', body, content_type))
        return f'{result["subfolder"]}/{result["name"]}' if result.get('subfolder') else result['name']

    def queue(self, workflow):
        body = json.dumps({'prompt': workflow, 'client_id': self.client_id}).encode()
        result = json.loads(self._post('/prompt', body, 'application/json'))
        if result.get('node_errors'):
            raise ComfyError(f'Workflow errors: {json.dumps(result["node_errors"])}')
        return result['prompt_id']

    def wait(self, prompt_id, timeout=1800, should_stop=None):
        """Polls until the job finishes and returns its history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_running():
                raise ComfyError(f'ComfyUI exited while working.\n{self.log_tail()}')
            if should_stop is not None and should_stop():
                self._post('/interrupt', b'{}', 'application/json')
                raise InterruptedError('Stopped by user.')
            history = json.loads(self._get(f'/history/{prompt_id}'))
            entry = history.get(prompt_id)
            if entry:
                status = entry.get('status', {})
                if status.get('status_str') == 'error':
                    raise ComfyError(f'ComfyUI job failed: {json.dumps(status.get("messages", []))[-2000:]}')
                if status.get('completed', True):
                    return entry
            time.sleep(0.5)
        raise ComfyError(f'ComfyUI job {prompt_id} timed out after {timeout}s.')

    def fetch_output(self, entry, workflow):
        """Returns the bytes of the first image saved by the OUT_IMAGE node."""
        output_node = find_node(workflow, TAG_OUTPUT)
        images = entry.get('outputs', {}).get(output_node, {}).get('images', [])
        if not images:
            raise ComfyError('ComfyUI finished but the OUT_IMAGE node saved no image.')
        query = urllib.parse.urlencode({key: images[0].get(key, '') for key in ('filename', 'subfolder', 'type')})
        return self._get(f'/view?{query}')

    def edit_image(self, image_path, prompt, seed, workflow, should_stop=None):
        """Runs one image through the edit workflow and returns the result as encoded image bytes."""
        image_name = self.upload_image(image_path)
        prepared = apply_inputs(workflow, image_name, prompt, seed)
        entry = self.wait(self.queue(prepared), should_stop=should_stop)
        return self.fetch_output(entry, prepared)
