#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
RUN_CRAWL4AI_SETUP="${RUN_CRAWL4AI_SETUP:-1}"
LOCAL_EDITABLE_PATHS="${LOCAL_EDITABLE_PATHS:-verl_rl}"

log() {
  printf '[install_with_uv] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

install_system_deps() {
  if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "INSTALL_SYSTEM_DEPS=1 but apt-get is unavailable; skipping system packages"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    SUDO=""
  fi

  log "installing system packages with apt-get"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y \
    git curl wget build-essential gcc g++ make cmake pkg-config \
    tmux screen ffmpeg libsndfile1
}

ensure_uv() {
  require_cmd uv
}

check_gpu_driver() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi detected"
    nvidia-smi || true
  else
    log "warning: nvidia-smi not found; continuing, but CUDA runtime may be unavailable"
  fi
}

create_venv() {
  log "installing Python ${PYTHON_VERSION} with uv"
  uv python install "${PYTHON_VERSION}"

  log "creating virtualenv at ${VENV_DIR}"
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
}

install_torch_stack() {
  log "installing torch/cu124 wheels"
  uv pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

  log "installing vllm and core runtime packages"
  uv pip install \
    "vllm==${VLLM_VERSION}" \
    "tensordict==0.6.2" \
    torchdata
}

install_python_packages() {
  log "installing training and utility dependencies"
  uv pip install \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=15.0.0" \
    pandas \
    "ray[default]>=2.41.0" \
    codetiming \
    hydra-core \
    pylatexenc \
    wandb \
    dill \
    pybind11 \
    liger-kernel \
    "nvidia-ml-py>=12.560.30" \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1"

  log "installing code_environment extras"
  uv pip install \
    boto3 \
    awscli \
    omegaconf \
    ujson \
    trafilatura \
    pathvalidate \
    smolagents \
    mammoth \
    markdownify \
    python-pptx \
    pdfminer.six \
    puremagic \
    pydub \
    requests \
    SpeechRecognition \
    beautifulsoup4 \
    youtube-transcript-api \
    html2text \
    jieba \
    crawl4ai \
    nltk
}

install_flash_attn() {
  local py_tag tmp_dir wheel_url wheel_path
  py_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"

  if [[ "${py_tag}" == "cp310" ]]; then
    tmp_dir="$(mktemp -d)"
    wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    wheel_path="${tmp_dir}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

    log "installing flash-attn from prebuilt wheel"
    require_cmd curl
    curl -L --fail --output "${wheel_path}" "${wheel_url}"
    uv pip install "${wheel_path}"
  else
    log "python tag is ${py_tag}; no matching prebuilt wheel configured"
    log "falling back to source install for flash-attn"
    uv pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"
  fi
}

install_local_editables() {
  local path
  for path in ${LOCAL_EDITABLE_PATHS}; do
    if [[ -d "${REPO_ROOT}/${path}" ]]; then
      log "installing local package in editable mode: ${path}"
      uv pip install -e "${REPO_ROOT}/${path}"
    else
      log "skipping missing local package path: ${path}"
    fi
  done
}

post_install_setup() {
  if [[ "${RUN_CRAWL4AI_SETUP}" == "1" ]]; then
    log "running crawl4ai setup"
    crawl4ai-setup || true
    crawl4ai-doctor || true
  fi

  log "downloading nltk punkt models"
  python -m nltk.downloader punkt punkt_tab || true
}

verify_install() {
  log "verifying torch / CUDA / vllm"
  python - <<'PY'
import torch
import vllm

print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("vllm", vllm.__version__)
PY

  log "verifying flash-attn and key extras"
  python - <<'PY'
import flash_attn  # noqa: F401
import crawl4ai  # noqa: F401
import pptx  # noqa: F401
import speech_recognition  # noqa: F401

print("import check: ok")
PY
}

main() {
  cd "${REPO_ROOT}"
  install_system_deps
  ensure_uv
  check_gpu_driver
  create_venv
  install_torch_stack
  install_python_packages
  install_flash_attn
  install_local_editables
  post_install_setup
  verify_install

  log "done"
  log "activate with: source ${VENV_DIR}/bin/activate"
}

main "$@"
