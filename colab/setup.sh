#!/usr/bin/env bash
# Per-session setup for the FramePack pipeline on Colab.
#
# Two isolated uv venvs (FramePack pins old diffusers/transformers; ComfyUI needs new ones) are
# installed while every model downloads in the background, so setup takes about as long as the
# slowest download instead of the sum of all steps. Logs: $LOG_DIR/*.log
set -uo pipefail

FP_DIR=${FP_DIR:-/content/FramePack}
COMFY_DIR=${COMFY_DIR:-/content/ComfyUI}
FP_VENV=${FP_VENV:-/content/venv-fp}
COMFY_VENV=${COMFY_VENV:-/content/venv-comfy}
DL_DIR=${DL_DIR:-/content/model_downloads}
LOG_DIR=${LOG_DIR:-/content/setup_logs}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
PREFETCH_F1=${PREFETCH_F1:-0}
DONE_MARKER=/content/.pipeline_setup_done

if [[ -f $DONE_MARKER ]]; then
  echo "Setup already done this session (delete $DONE_MARKER to redo it)."
  exit 0
fi

mkdir -p "$LOG_DIR" "$DL_DIR"
start=$SECONDS
step() { echo "[$(( SECONDS - start ))s] $*"; }

free_gb=$(df -BG --output=avail /content | tail -1 | tr -dc '0-9')
if (( free_gb < 90 )); then
  echo "WARNING: only ${free_gb} GB free on /content; the models need about 75 GB."
fi

step "Installing uv"
pip install -q uv || exit 1

if [[ ! -d $COMFY_DIR ]]; then
  step "Cloning ComfyUI"
  git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR" || exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
hf() { uvx -q --from 'huggingface_hub[hf_xet]>=1.0,<2' hf "$@"; }

download_framepack_models() {
  export HF_HOME="$FP_DIR/hf_download"  # where demo_gradio_f1.py looks for them
  # Trailing slashes download just those subfolders (FramePack doesn't use HunyuanVideo's own transformer).
  hf download hunyuanvideo-community/HunyuanVideo text_encoder/ text_encoder_2/ tokenizer/ tokenizer_2/ vae/ &&
  hf download lllyasviel/flux_redux_bfl feature_extractor/ image_encoder/ &&
  hf download lllyasviel/FramePackI2V_HY &&
  if [[ $PREFETCH_F1 == 1 ]]; then hf download lllyasviel/FramePack_F1_I2V_HY_20250503; fi
}

# Downloads one file of a repo and links it into ComfyUI's model folder of the given kind.
comfy_model() {
  local repo=$1 file=$2 kind=$3
  hf download "$repo" "$file" --local-dir "$DL_DIR/$repo" &&
  mkdir -p "$COMFY_DIR/models/$kind" &&
  ln -sf "$DL_DIR/$repo/$file" "$COMFY_DIR/models/$kind/$(basename "$file")"
}

download_edit_models() {
  comfy_model Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors diffusion_models &&
  comfy_model Comfy-Org/HunyuanVideo_1.5_repackaged split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors text_encoders &&
  comfy_model Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors vae &&
  comfy_model lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors loras
}

install_framepack_env() {
  { [[ -x $FP_VENV/bin/python ]] || uv venv -q --python 3.10 "$FP_VENV"; } &&
  uv pip install -q --python "$FP_VENV/bin/python" torch torchvision torchaudio --index-url "$TORCH_INDEX" &&
  uv pip install -q --python "$FP_VENV/bin/python" -r "$FP_DIR/requirements.txt" sageattention==1.0.6
}

install_comfy_env() {
  { [[ -x $COMFY_VENV/bin/python ]] || uv venv -q --python 3.12 "$COMFY_VENV"; } &&
  uv pip install -q --python "$COMFY_VENV/bin/python" torch torchvision torchaudio --index-url "$TORCH_INDEX" &&
  uv pip install -q --python "$COMFY_VENV/bin/python" -r "$COMFY_DIR/requirements.txt"
}

declare -A jobs
for job in download_framepack_models download_edit_models install_framepack_env install_comfy_env; do
  "$job" > "$LOG_DIR/$job.log" 2>&1 &
  jobs[$job]=$!
done
step "Started downloads and installs in parallel"

failed=0
for job in install_framepack_env install_comfy_env download_edit_models download_framepack_models; do
  if wait "${jobs[$job]}"; then
    step "$job done"
  else
    failed=1
    echo "!! $job FAILED. Last lines of $LOG_DIR/$job.log:"
    tail -n 25 "$LOG_DIR/$job.log"
  fi
done

if (( failed )); then
  echo "Setup failed after $(( SECONDS - start ))s. Re-run the cell to retry; finished downloads are reused."
  exit 1
fi

touch "$DONE_MARKER"
step "Setup complete"
