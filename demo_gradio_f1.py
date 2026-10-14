from diffusers_helper.hf_login import login
import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import gradio as gr
import torch
import traceback
import einops
import numpy as np
import argparse
import atexit
import math
import gc
import io
import sys
import time
import shutil
import tempfile
from huggingface_hub import hf_hub_download
import requests
import threading
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from utils.lora_utils import merge_lora_to_state_dict
from pipeline import batch_store
from pipeline.batch_store import Batch
from pipeline.comfy_runner import ComfyServer, load_workflow

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--high_vram", choices=['off', 'auto'], default='off',
                    help="'auto' keeps every model on the GPU when it has more than 60 GB free (A100/H100 80GB).")
parser.add_argument("--comfy_dir", type=str, default=os.environ.get('COMFY_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ComfyUI'))),
                    help='ComfyUI checkout used to generate end frames.')
parser.add_argument("--comfy_python", type=str, default=os.environ.get('COMFY_PYTHON', sys.executable),
                    help="Python interpreter of ComfyUI's venv.")
parser.add_argument("--comfy_port", type=int, default=8188)
parser.add_argument("--edit_workflow", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline', 'workflows', 'qwen_edit_2511.api.json'),
                    help='ComfyUI API-format workflow that turns a start frame into an end frame.')
args = parser.parse_args()

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = args.high_vram == 'auto' and free_mem_gb > 60
print(f'Free VRAM {free_mem_gb:.1f} GB, high-VRAM mode: {high_vram}')

# The sampling loop below is the original FramePack's inverted (end-first) loop, which the original
# weights were trained for. F1 was trained to predict forward, so it is offered for comparison.
VIDEO_MODELS = {
    'FramePack (end-frame trained)': 'lllyasviel/FramePackI2V_HY',
    'FramePack-F1': 'lllyasviel/FramePack_F1_I2V_HY_20250503',
}
DEFAULT_VIDEO_MODEL = 'FramePack (end-frame trained)'

# Model loading
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = None
transformer_dtype = torch.bfloat16
previous_lora_file = None
previous_lora_multiplier = None
previous_video_model = None

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)

stream = AsyncStream()
outputs_folder = args.output_dir
os.makedirs(outputs_folder, exist_ok=True)

lora_download_folder = './loras'
os.makedirs(lora_download_folder, exist_ok=True)


def download_lora(url_or_path):
    if not url_or_path or not url_or_path.strip():
        return None
    url_or_path = url_or_path.strip()
    if os.path.exists(url_or_path):
        return url_or_path

    if "huggingface.co" in url_or_path:
        try:
            url = url_or_path.replace("https://huggingface.co/", "").strip("/")
            path_parts = url.split("/")
            if "blob" in path_parts or "resolve" in path_parts:
                idx = path_parts.index("blob") if "blob" in path_parts else path_parts.index("resolve")
                # Detect filename and repo_id correctly
                # Parts before blob/resolve is repo_id, after it is file path
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                filename = "/".join(path_parts[idx+2:])
            elif len(path_parts) >= 3:
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                filename = "/".join(path_parts[2:])
            else:
                print(f"Malformed HF URL: {url_or_path}")
                return None

            # Simple check if it's likely a lora
            if not filename.endswith(".safetensors"):
                print(f"Skipping non-safetensors file: {filename}")
                return None

            print(f"Downloading LoRA from HF: {filename} from {repo_id}...")
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            print(f"HF Download failed: {e}")
            return None

    if url_or_path.startswith("http"):
        try:
            filename = url_or_path.split("/")[-1].split("?")[0]
            if not filename.endswith(".safetensors"):
                filename += ".safetensors"
            local_path = os.path.join(lora_download_folder, filename)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1024: # basic size check
                print(f"Using cached LoRA: {local_path}")
                return local_path
            print(f"Downloading LoRA from URL: {url_or_path} ...")
            response = requests.get(url_or_path, stream=True, timeout=60)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return local_path
        except Exception as e:
            print(f"URL Download failed for {url_or_path}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return None
    return None


def load_image_as_numpy(path):
    with Image.open(path) as img:
        return np.array(img.convert('RGB'))


def collect_loras(lora_files, lora_links, lora_multiplier):
    active_loras = []
    if lora_files:
        for f in lora_files:
            active_loras.append((f, lora_multiplier))
    if lora_links:
        for line in lora_links.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                if ',' in line:
                    parts = line.split(',')
                    active_loras.append((parts[0].strip(), float(parts[1].strip())))
                else:
                    active_loras.append((line, lora_multiplier))
            except Exception as e:
                print(f"Skipping malformed LoRA config line '{line}': {e}")
    return active_loras


def log(text):
    stream.output_queue.push(('log', text))


def stop_requested():
    return stream.input_queue.top() == 'end'


def ensure_transformer(active_loras, lora_multiplier, video_model):
    """Loads (or reloads) the transformer only when the model or LoRA setup changed, so a batch loads it once."""
    global transformer, previous_lora_file, previous_lora_multiplier, previous_video_model

    model_changed = transformer is None or (
        str(active_loras) != str(previous_lora_file)
        or lora_multiplier != previous_lora_multiplier
        or video_model != previous_video_model
    )
    if not model_changed:
        return

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading {video_model} ...'))))
    transformer = None
    time.sleep(1.0)
    torch.cuda.empty_cache()
    gc.collect()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(VIDEO_MODELS[video_model], torch_dtype=torch.bfloat16).cpu()
    transformer.eval()
    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)

    if len(active_loras) > 0:
        state_dict = transformer.state_dict()
        for lora_path_or_url, multiplier in active_loras:
            local_path = download_lora(lora_path_or_url)
            if local_path:
                print(f"Merging LoRA file {os.path.basename(local_path)} with multiplier {multiplier}...")
                state_dict = merge_lora_to_state_dict(state_dict, local_path, multiplier, device=gpu)
                gc.collect()
        info = transformer.load_state_dict(state_dict, strict=True, assign=True)
        print(f"All LoRAs applied: {info}")

    previous_lora_file = str(active_loras)
    previous_lora_multiplier = lora_multiplier
    previous_video_model = video_model
    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
    else:
        transformer.to(gpu)


def release_video_models():
    """Hands the GPU and most of the RAM to ComfyUI. The transformer is reloaded lazily for the next videos."""
    global transformer, previous_lora_file, previous_lora_multiplier, previous_video_model
    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae)
    transformer = None
    previous_lora_file = previous_lora_multiplier = previous_video_model = None
    gc.collect()
    torch.cuda.empty_cache()


def restore_high_vram_models():
    if high_vram:
        for model in (text_encoder, text_encoder_2, image_encoder, vae):
            model.to(gpu)


# ComfyUI runs only while end frames are being made or reviewed.
comfy = None
comfy_lock = threading.Lock()


def get_comfy():
    global comfy
    with comfy_lock:
        if comfy is None or not comfy.is_running():
            comfy = ComfyServer(args.comfy_dir, python=args.comfy_python, port=args.comfy_port)
            comfy.start()
        return comfy


def stop_comfy():
    global comfy
    with comfy_lock:
        if comfy is not None:
            comfy.close()
            comfy = None


atexit.register(stop_comfy)


def fit_to_size(image, size):
    """Scales an image to cover `size` and center-crops it, so an end frame matches its start frame exactly."""
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))), Image.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def make_review_thumbnail(start_path, end_path, out_path, height=360):
    images = []
    for path in (start_path, end_path):
        with Image.open(path) as img:
            img = img.convert('RGB')
            images.append(img.resize((max(1, round(img.width * height / img.height)), height), Image.LANCZOS))
    gap = 8
    canvas = Image.new('RGB', (images[0].width + images[1].width + gap, height), (255, 255, 255))
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (images[0].width + gap, 0))
    canvas.save(out_path, quality=90)


def review_payload(batch):
    """Gallery items (start|end thumbnails) and the names that can be re-rolled."""
    gallery = []
    for name in batch.names:
        end_path = batch.end_path(name)
        if end_path is None:
            continue
        thumb = batch.review_path(name)
        if not os.path.exists(thumb):
            make_review_thumbnail(batch.start_path(name), end_path, thumb)
        item = batch.item(name)
        detail = f'seed {item["end_seed"]}' if item.get('end_source') == 'generated' else 'uploaded'
        gallery.append((thumb, f'{name} · {detail}'))
    return batch.root, gallery, batch.names


def push_review(batch):
    stream.output_queue.push(('review', review_payload(batch)))


def stage_end_frames(batch, names, prompt, seeds):
    """Generates an end frame for each name with ComfyUI. Returns False if the user stopped it."""
    release_video_models()
    total = len(names)
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ComfyUI (the first end frame also loads the edit model) ...'))))
    server = get_comfy()
    workflow = load_workflow(args.edit_workflow)
    stage_start = time.time()

    for index, name in enumerate(names):
        if stop_requested():
            log(f'Stopped making end frames at {index + 1}/{total}.')
            return False
        desc = f'**End frame {index + 1}/{total}** — {name}'
        stream.output_queue.push(('progress', (None, desc, make_progress_bar_html(int(100 * index / total), 'Editing ...'))))
        started = time.time()
        try:
            data = server.edit_image(batch.start_path(name), prompt, seeds[name], workflow, should_stop=stop_requested)
            with Image.open(batch.start_path(name)) as start:
                start_size = start.size
            with Image.open(io.BytesIO(data)) as edited:
                end = fit_to_size(edited.convert('RGB'), start_size)
            end_path = batch.generated_end_path(name)
            end.save(end_path)
            batch.record_end(name, end_path, 'generated', seed=seeds[name], prompt=prompt, seconds=time.time() - started)
            make_review_thumbnail(batch.start_path(name), end_path, batch.review_path(name))
            log(f'End frame {index + 1}/{total} — {name} ({time.time() - started:.1f}s)')
        except InterruptedError:
            log(f'Stopped making end frames at {index + 1}/{total}.')
            return False
        except Exception as e:
            traceback.print_exc()
            log(f'End frame failed for {name}: {e}')

    log(f'End frames done in {time.time() - stage_start:.0f}s.')
    return True


@torch.no_grad()
def encode_text(prompt, n_prompt, cfg):
    """Encodes the prompt once; a batch reuses the result for every image."""
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
    if not high_vram:
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)

    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    if cfg == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    return dict(
        llama_vec=llama_vec.to(transformer_dtype),
        llama_attention_mask=llama_attention_mask,
        clip_l_pooler=clip_l_pooler.to(transformer_dtype),
        llama_vec_n=llama_vec_n.to(transformer_dtype),
        llama_attention_mask_n=llama_attention_mask_n,
        clip_l_pooler_n=clip_l_pooler_n.to(transformer_dtype),
    )


def end_frame_gap(pixels, end_image_np, frames_from_end=5):
    """How far the model's own last frame is from the end frame: mean abs difference, 0-255.

    The final latent frame is the end frame itself (4 pixel frames), so the 5th frame from the end
    is the last one the model generated. Lower means a smoother arrival at the end frame.
    """
    t = pixels.shape[2]
    frame = pixels[0, :, max(0, t - frames_from_end)].float().clamp(-1, 1)
    frame = ((frame + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    return float(np.abs(frame - end_image_np.astype(np.float32)).mean())


@torch.no_grad()
def generate_video(input_image, end_image, text_cond, output_path, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_fps, mp4_crf, resolution, blend_end_embeds=True, preview_dir=None, item_desc=''):
    """Generates a single video into output_path. Returns (status, end_gap) with status 'ok', 'aborted' or 'error'.

    With preview_dir set, the growing video is also written there after every section for the UI.
    """
    llama_vec = text_cond['llama_vec']
    llama_attention_mask = text_cond['llama_attention_mask']
    clip_l_pooler = text_cond['clip_l_pooler']
    llama_vec_n = text_cond['llama_vec_n']
    llama_attention_mask_n = text_cond['llama_attention_mask_n']
    clip_l_pooler_n = text_cond['clip_l_pooler_n']

    def desc_with(body=''):
        if item_desc and body:
            return f'{item_desc}  \n{body}'  # two trailing spaces = markdown hard break
        return item_desc or body

    target_pixel_frames = int(total_second_length * mp4_fps)
    frames_per_extension = latent_window_size * 4
    total_latent_sections = math.ceil((target_pixel_frames - 1) / frames_per_extension)
    total_latent_sections = int(max(total_latent_sections, 1))

    status = 'ok'
    end_gap = None
    save_thread = None
    # The final file is written to fast local disk first, then moved to output_path (usually Drive).
    local_output = os.path.join(tempfile.gettempdir(), f'framepack_{os.getpid()}_{os.path.basename(output_path)}')

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # Image processing
        stream.output_queue.push(('progress', (None, desc_with(), make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, desc_with(), make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # VAE encoding for end image if provided
        end_latent = None
        end_image_np = None
        if end_image is not None:
            stream.output_queue.push(('progress', (None, desc_with(), make_progress_bar_html(0, 'VAE encoding end image ...'))))
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
            end_latent = vae_encode(end_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, desc_with(), make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        start_image_embeds = image_encoder_output.last_hidden_state.to(transformer_dtype)
        end_image_embeds = None
        if end_image_np is not None and blend_end_embeds:
            end_image_embeds = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder).last_hidden_state.to(transformer_dtype)

        # Sampling Loop (Inverted/Backward Mode for End Frame Support)
        stream.output_queue.push(('progress', (None, desc_with(), make_progress_bar_html(0, 'Start sampling ...'))))

        if not high_vram:
            torch.cuda.empty_cache()
            gc.collect()

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        if end_latent is not None:
            history_latents[:, :, 0, :, :] = end_latent[:, :, 0, :, :].to(history_latents)
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))
        # Optional: heuristic from demo_gradio.py for smoother sequences
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        is_first_iteration = True
        for section_index, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                status = 'aborted'
                break

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            # Sections are generated end-first, so the image conditioning slides from the end frame
            # (first section) to the start frame (last section).
            if end_image_embeds is not None:
                weight = section_index / (len(latent_paddings) - 1) if len(latent_paddings) > 1 else 1.0
                image_encoder_last_hidden_state = torch.lerp(end_image_embeds, start_image_embeds, weight)
            else:
                image_encoder_last_hidden_state = start_image_embeds

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    raise KeyboardInterrupt('Stop signal.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                px_count = int(max(0, total_generated_latent_frames * 4 - 3))
                desc = desc_with(f'Generated: {px_count} frames (~{px_count/mp4_fps:.1f}s). Expanding...')
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))

            generated_latents = sample_hunyuan(
                transformer=transformer, sampler='unipc', width=width, height=height,
                frames=num_frames, real_guidance_scale=cfg, distilled_guidance_scale=gs,
                guidance_rescale=rs, num_inference_steps=steps, generator=rnd, prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n, device=gpu, dtype=transformer.dtype,
                image_embeddings=image_encoder_last_hidden_state, latent_indices=latent_indices,
                clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_first_iteration and end_latent is not None:
                generated_latents = torch.cat([generated_latents, end_latent.to(generated_latents)], dim=2)
            is_first_iteration = False

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            # Update history counts and latents
            new_latent_frames = int(generated_latents.shape[2])
            total_generated_latent_frames += new_latent_frames
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # Optimized VAE decoding: only decode the newly generated section
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()
                torch.cuda.empty_cache()

            current_px_count = int(max(0, total_generated_latent_frames * 4 - 3))
            is_final_section = is_last_section or current_px_count >= target_pixel_frames

            # Only the finished video is written to output_path; previews (if any) go to preview_dir.
            if is_final_section or preview_dir is not None:
                if is_final_section:
                    save_path = local_output
                else:
                    save_path = os.path.join(preview_dir, f'{os.path.splitext(os.path.basename(output_path))[0]}_{total_generated_latent_frames}.mp4')

                # Speedup: Save video asynchronously in a background thread
                def async_save_video(px, path, fps, crf, stream_ptr, announce):
                    save_bcthw_as_mp4(px, path, fps=fps, crf=crf)
                    if announce:
                        stream_ptr.output_queue.push(('file', path))

                if save_thread is not None:
                    save_thread.join()
                save_thread = threading.Thread(target=async_save_video, args=(history_pixels.clone(), save_path, mp4_fps, mp4_crf, stream, not is_final_section))
                save_thread.start()

            if not high_vram:
                gc.collect()
                torch.cuda.empty_cache()

            if is_final_section:
                if end_image_np is not None:
                    end_gap = end_frame_gap(history_pixels, end_image_np)
                break

    except KeyboardInterrupt:
        # Raised from the sampler callback when the user hits "End Generation".
        print('Generation stopped by user.')
        status = 'aborted'
    except Exception:
        traceback.print_exc()
        status = 'error'
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    finally:
        # Let the file land before the next job starts, so its 'file' signal is never lost.
        if save_thread is not None:
            save_thread.join()

    if status == 'ok':
        if os.path.exists(local_output):
            shutil.move(local_output, output_path)
            stream.output_queue.push(('file', output_path))
        else:
            status = 'error'
    elif os.path.exists(local_output):
        os.remove(local_output)

    return status, end_gap


VIDEO_SETTING_NAMES = ['prompt', 'n_prompt', 'seed', 'total_second_length', 'latent_window_size', 'steps', 'cfg', 'gs', 'rs',
                       'gpu_memory_preservation', 'use_teacache', 'mp4_fps', 'mp4_crf', 'resolution', 'lora_files', 'lora_links',
                       'lora_multiplier', 'video_model', 'blend_end_embeds']


@torch.no_grad()
def stage_videos(batch, edit_prompt, vs):
    """Makes every video the batch is still missing, one after another with the same settings."""
    stop_comfy()
    names = batch.pending_videos()
    if not names:
        log('Every video in this batch is already done.')
        return

    # An empty motion prompt reuses the end-frame prompt, so a run can need only one prompt.
    motion_prompt = (vs['prompt'] or '').strip() or (edit_prompt or '').strip()
    batch.update(motion_prompt=motion_prompt, settings={k: v for k, v in vs.items() if k not in ('prompt', 'lora_files')})

    restore_high_vram_models()
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    active_loras = collect_loras(vs['lora_files'], vs['lora_links'], vs['lora_multiplier'])
    text_cond = encode_text(motion_prompt, vs['n_prompt'], vs['cfg'])
    ensure_transformer(active_loras, vs['lora_multiplier'], vs['video_model'])

    # A one-item batch shows the video growing section by section; bigger batches skip that for speed.
    preview_dir = tempfile.mkdtemp(prefix='framepack_preview_') if len(batch.names) == 1 else None
    total = len(names)
    log(f'Making {total} video(s) with {vs["video_model"]}.')

    for index, name in enumerate(names):
        if stop_requested():
            log(f'Stopped before video {index + 1}/{total}.')
            break

        try:
            image = load_image_as_numpy(batch.start_path(name))
        except Exception as e:
            log(f'Skipped {name}: start frame could not be read ({e})')
            batch.record_video(name, 'error')
            continue

        end_image = None
        end_path = batch.end_path(name)
        if end_path is not None:
            try:
                end_image = load_image_as_numpy(end_path)
            except Exception as e:
                log(f'End frame for {name} could not be read — generating without it ({e})')

        pair_label = f'{name} → end frame' if end_image is not None else f'{name} (no end frame)'
        started = time.time()
        status, end_gap = generate_video(
            image, end_image, text_cond, batch.video_path(name),
            seed=vs['seed'], total_second_length=vs['total_second_length'], latent_window_size=vs['latent_window_size'],
            steps=vs['steps'], cfg=vs['cfg'], gs=vs['gs'], rs=vs['rs'], gpu_memory_preservation=vs['gpu_memory_preservation'],
            use_teacache=vs['use_teacache'], mp4_fps=vs['mp4_fps'], mp4_crf=vs['mp4_crf'], resolution=vs['resolution'],
            blend_end_embeds=vs['blend_end_embeds'], preview_dir=preview_dir,
            item_desc=f'**Video {index + 1}/{total}** — {pair_label}',
        )
        seconds = time.time() - started

        if status == 'aborted':
            batch.record_video(name, 'pending')
            log(f'Stopped at {index + 1}/{total} ({name}).')
            break
        batch.record_video(name, 'done' if status == 'ok' else 'error', seconds=seconds, end_gap=end_gap)
        if status == 'ok':
            gap = f', end-frame gap {end_gap:.1f}' if end_gap is not None else ''
            log(f'Finished {index + 1}/{total} — {name}.mp4 in {seconds:.0f}s{gap}')
        else:
            log(f'Failed {index + 1}/{total} — {name} (see console)')

    if preview_dir is not None:
        shutil.rmtree(preview_dir, ignore_errors=True)


def save_upload(image_np, folder, filename):
    path = os.path.join(folder, filename)
    Image.fromarray(image_np).save(path)
    return path


def build_batch(start_files, input_image, end_files, end_image, generate_end, scratch):
    start_paths = batch_store.expand_image_inputs(start_files, scratch)
    if not start_paths:
        if input_image is None:
            raise ValueError('Upload start images (or a ZIP of them), or a single Start Image.')
        start_paths = [save_upload(input_image, scratch, 'start.png')]
    end_paths = batch_store.expand_image_inputs(end_files, scratch)

    # The single End Image belongs to a single start. With many starts it fills in for unmatched
    # ones only when end frames are not being generated.
    fallback = None
    if end_image is not None:
        if len(start_paths) == 1 or not generate_end:
            fallback = save_upload(end_image, scratch, 'end_image.png')
        else:
            log('The single End Image is ignored: start images without an uploaded end frame get a generated one.')

    return batch_store.create_batch(outputs_folder, start_paths, end_paths, fallback_end_path=fallback)


def job_run(start_files, input_image, end_files, end_image, edit_prompt, generate_end, pause_review, end_seed, vs):
    scratch = tempfile.mkdtemp(prefix='pipeline_inputs_')
    try:
        batch = build_batch(start_files, input_image, end_files, end_image, generate_end, scratch)
        batch.update(edit_prompt=edit_prompt)
        stream.output_queue.push(('batch', batch.root))
        uploaded = len(batch.names) - len(batch.pending_end_frames())
        log(f'Batch {batch.batch_id}: {len(batch.names)} start image(s), {uploaded} with an uploaded end frame.')

        pending = batch.pending_end_frames()
        if generate_end and pending:
            if not (edit_prompt or '').strip():
                log('"Generate end frames" is on but the end-frame prompt is empty. Add a prompt, or turn it off to make videos without end frames.')
                return
            finished = stage_end_frames(batch, pending, edit_prompt, {name: int(end_seed or 0) for name in pending})
            if not finished:
                push_review(batch)  # show whatever was made before the stop
                return
            if pause_review:
                push_review(batch)
                log('Check the end frames. Re-roll any you don\'t like, then press "Approve → make videos".')
                return

        stage_videos(batch, edit_prompt, vs)
    except Exception as e:
        traceback.print_exc()
        log(f'Error: {e}')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        stream.output_queue.push(('end', None))


def job_reroll(batch_root, names, edit_prompt, end_seed):
    try:
        batch = Batch(batch_root)
        if not names:
            log('Tick the items to re-roll first.')
            return
        # Re-rolls use the current prompt box, so the prompt can be tweaked between tries.
        prompt = (edit_prompt or '').strip() or batch.manifest.get('edit_prompt') or ''
        seeds = {}
        for name in names:
            previous = batch.item(name).get('end_seed')
            seeds[name] = int(end_seed or 0) if previous is None else previous + 1
        stage_end_frames(batch, names, prompt, seeds)
        push_review(batch)
    except Exception as e:
        traceback.print_exc()
        log(f'Error: {e}')
    finally:
        stream.output_queue.push(('end', None))


def job_videos(batch_root, edit_prompt, vs):
    try:
        batch = Batch(batch_root)
        stage_videos(batch, batch.manifest.get('edit_prompt') or edit_prompt, vs)
    except Exception as e:
        traceback.print_exc()
        log(f'Error: {e}')
    finally:
        stream.output_queue.push(('end', None))


def stream_job(job, *job_args):
    """Runs a job on the worker thread and turns its signals into UI updates."""
    global stream
    stream = AsyncStream()
    yield {start_button: gr.update(interactive=False), end_button: gr.update(interactive=True),
           reroll_button: gr.update(interactive=False), approve_button: gr.update(interactive=False), batch_log: ''}
    async_run(job, *job_args)

    try:
        log_lines = []
        while True:
            res = stream.output_queue.pop()
            if res is None:
                time.sleep(0.01)
                continue

            signal, data = res
            if signal == 'file':
                yield {result_video: data}
            elif signal == 'progress':
                preview, desc, bar = data
                yield {preview_image: preview if preview is not None else gr.update(), progress_desc: desc, progress_bar: bar}
            elif signal == 'log':
                log_lines.append(data)
                yield {batch_log: '\n'.join(f'- {line}' for line in log_lines)}
            elif signal == 'batch':
                yield {batch_state: data}
            elif signal == 'review':
                root, gallery, names = data
                yield {batch_state: root, review_group: gr.update(visible=True), review_gallery: gallery,
                       reroll_choice: gr.update(choices=names, value=[])}
            elif signal == 'end':
                break
    except Exception as e:
        print(f"UI Loop Error: {e}")
        traceback.print_exc()
    finally:
        yield {progress_desc: 'Done.', progress_bar: '', start_button: gr.update(interactive=True), end_button: gr.update(interactive=False),
               reroll_button: gr.update(interactive=True), approve_button: gr.update(interactive=True),
               resume_choice: gr.update(choices=batch_store.list_resumable(outputs_folder))}


def on_run(start_files, input_image, end_files, end_image, edit_prompt, generate_end, pause_review, end_seed, *video_values):
    vs = dict(zip(VIDEO_SETTING_NAMES, video_values))
    yield {review_group: gr.update(visible=False)}
    yield from stream_job(job_run, start_files, input_image, end_files, end_image, edit_prompt, generate_end, pause_review, end_seed, vs)


def on_reroll(batch_root, names, edit_prompt, end_seed):
    if not batch_root:
        yield {batch_log: '- Nothing to re-roll yet: run a batch or load one first.'}
        return
    yield from stream_job(job_reroll, batch_root, names, edit_prompt, end_seed)


def on_approve(batch_root, edit_prompt, *video_values):
    if not batch_root:
        yield {batch_log: '- Nothing to make videos for yet: run a batch or load one first.'}
        return
    vs = dict(zip(VIDEO_SETTING_NAMES, video_values))
    yield from stream_job(job_videos, batch_root, edit_prompt, vs)


def on_load_batch(batch_root):
    if not batch_root:
        return {batch_log: '- Pick a batch to load.'}
    batch = Batch(batch_root)
    root, gallery, names = review_payload(batch)
    return {batch_state: root, review_group: gr.update(visible=True), review_gallery: gallery,
            reroll_choice: gr.update(choices=names, value=[]), edit_prompt_box: batch.manifest.get('edit_prompt') or '',
            batch_log: f'- Loaded {batch.batch_id}: {batch.summary()}. Press "Approve → make videos" to finish it.'}


def on_refresh_batches():
    return gr.update(choices=batch_store.list_resumable(outputs_folder), value=None)


def end_process():
    stream.input_queue.push('end')


quick_prompts = [['The girl dances gracefully, with clear movements, full of charm.'], ['A character doing some simple body movements.']]
css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack pipeline: start frame → end frame → video')
    batch_state = gr.State(None)
    with gr.Row():
        with gr.Column():
            start_files = gr.File(label="Start Images (several images or a ZIP)", file_count="multiple", file_types=["image", ".zip"], type="filepath")
            edit_prompt_box = gr.Textbox(label="End-frame prompt", lines=2, placeholder="What the last frame should look like, e.g. 'Same scene and person, now standing up at the window, smiling'")
            prompt = gr.Textbox(label="Motion prompt", value='', placeholder="Leave empty to reuse the end-frame prompt")
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt)
            with gr.Row():
                generate_end = gr.Checkbox(label='Generate end frames', value=True)
                pause_review = gr.Checkbox(label='Pause to review end frames', value=True)

            with gr.Row():
                start_button = gr.Button(value="Run", variant="primary")
                end_button = gr.Button(value="Stop", variant="stop", interactive=False)

            with gr.Accordion("Your own end frames / single image", open=False):
                end_files = gr.File(label="End Images (optional; several images or a ZIP)", file_count="multiple", file_types=["image", ".zip"], type="filepath")
                gr.Markdown('End images are paired with start images by closest file name (`shot3_start.png` → `shot3_end.png`) and take priority over generated ones. The single Start/End Image below are for one-off runs when no Start Images are uploaded.')
                with gr.Row():
                    input_image = gr.Image(sources='upload', type="numpy", label="Start Image", height=240)
                    end_image = gr.Image(sources='upload', type="numpy", label="End Image (Optional)", height=240)

            with gr.Accordion("Resume a batch", open=False):
                resume_choice = gr.Dropdown(label="Unfinished batches", choices=[], value=None)
                with gr.Row():
                    refresh_button = gr.Button(value="Refresh")
                    load_button = gr.Button(value="Load batch")

            with gr.Accordion("Advanced Settings", open=False):
                resolution = gr.Slider(label="Resolution", minimum=240, maximum=720, value=416, step=16)
                video_model = gr.Dropdown(label="Video model", choices=list(VIDEO_MODELS), value=DEFAULT_VIDEO_MODEL)
                blend_end_embeds = gr.Checkbox(label='Blend end-frame image embedding', value=True, info='Image conditioning slides from the end frame to the start frame across sections.')
                end_seed = gr.Number(label="End-frame Seed", value=42, precision=0, info='Same seed for every image keeps the look consistent. Re-rolls add 1.')
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                seed = gr.Number(label="Seed", value=31344344537, precision=0)
                total_second_length = gr.Slider(label="Total Video Length", minimum=1, maximum=120, value=30, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window", minimum=1, maximum=33, value=9, step=1, visible=False)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG", value=1.0, visible=False)
                gs = gr.Slider(label="Distilled CFG", value=10.0)
                rs = gr.Slider(label="CFG Re-Scale", value=0.0, visible=False)
                gpu_memory_preservation = gr.Slider(label="GPU Preserved Memory (GB)", minimum=6, maximum=128, value=6, visible=not high_vram)
                mp4_fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=15, step=1)
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16)

            with gr.Group():
                gr.Markdown("### LoRA Settings")
                lora_links = gr.Textbox(label="LoRA Links / Paths", placeholder="One link or path per line. Optional multiplier: url,0.8", lines=5)
                lora_files = gr.File(label="Local LoRA Files", file_count="multiple", type="filepath")
                lora_multiplier = gr.Slider(label="Global LoRA Multiplier", minimum=0.0, maximum=1.0, value=0.8, step=0.1)

        with gr.Column():
            with gr.Group(visible=False) as review_group:
                gr.Markdown('### Review end frames (start | end)')
                review_gallery = gr.Gallery(label="End frames", columns=2, height=420, object_fit="contain")
                reroll_choice = gr.CheckboxGroup(label="Re-roll these (uses the current end-frame prompt)", choices=[])
                with gr.Row():
                    reroll_button = gr.Button(value="Re-roll selected")
                    approve_button = gr.Button(value="Approve → make videos", variant="primary")
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('Ready.')
            progress_bar = gr.HTML('')
            batch_log = gr.Markdown('')

    video_inputs = [prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                    use_teacache, mp4_fps, mp4_crf, resolution, lora_files, lora_links, lora_multiplier, video_model, blend_end_embeds]
    assert len(video_inputs) == len(VIDEO_SETTING_NAMES)
    all_outputs = [result_video, preview_image, progress_desc, progress_bar, batch_log, start_button, end_button,
                   review_group, review_gallery, reroll_choice, reroll_button, approve_button, batch_state, resume_choice, edit_prompt_box]

    start_button.click(fn=on_run, inputs=[start_files, input_image, end_files, end_image, edit_prompt_box, generate_end, pause_review, end_seed] + video_inputs, outputs=all_outputs)
    reroll_button.click(fn=on_reroll, inputs=[batch_state, reroll_choice, edit_prompt_box, end_seed], outputs=all_outputs)
    approve_button.click(fn=on_approve, inputs=[batch_state, edit_prompt_box] + video_inputs, outputs=all_outputs)
    load_button.click(fn=on_load_batch, inputs=[resume_choice], outputs=all_outputs)
    refresh_button.click(fn=on_refresh_batches, outputs=[resume_choice])
    end_button.click(fn=end_process)
    block.load(fn=on_refresh_batches, outputs=[resume_choice])

block.launch(server_name=args.server, server_port=args.port, share=args.share, inbrowser=args.inbrowser,
             allowed_paths=[outputs_folder, os.path.realpath(outputs_folder), tempfile.gettempdir()])
