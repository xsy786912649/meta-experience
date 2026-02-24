#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_docker_verl_public.sh
#   BASE_IMAGE=vllm/vllm-openai:v0.8.5 bash run_docker_verl_public.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_TAG="${IMAGE_TAG:-verl-local:dev}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.8.5}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
CONTAINER_NAME="${CONTAINER_NAME:-verl-local-dev}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-$SCRIPT_DIR}"
HOST_PORT="${HOST_PORT:-8011}"
CONTAINER_PORT="${CONTAINER_PORT:-8011}"

echo "[build] image=${IMAGE_TAG}"
echo "[build] base_image=${BASE_IMAGE}"
echo "[build] flash_attn_version=${FLASH_ATTN_VERSION} (required)"

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
  -t "${IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile_verl.public" \
  "${SCRIPT_DIR}"

echo "[run] container=${CONTAINER_NAME}"
echo "[run] mount=${WORKSPACE_MOUNT} -> /workspace/verl_toolmock"
echo "[run] ports=${HOST_PORT}:${CONTAINER_PORT}"

docker run --gpus all -p "${HOST_PORT}:${CONTAINER_PORT}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -it --rm \
  --name "${CONTAINER_NAME}" \
  -v "${WORKSPACE_MOUNT}:/workspace/verl_toolmock" \
  --mount type=tmpfs,destination=/tmpfs \
  "${IMAGE_TAG}"
