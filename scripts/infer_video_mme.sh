#!/usr/bin/env bash
# 首次推理实验：Video-MME 全量 900 题
# 用法:
#   bash scripts/infer_video_mme.sh                    # 默认 GPU 0，全量
#   MODE=mock N_SAMPLES=10 bash scripts/infer_video_mme.sh  # smoke test
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# 加载环境变量（API key 等）
set -a
source .env
set +a

PYTHON="$(conda run -n Video-Tree-TRM which python)"

"${PYTHON}" main.py \
    --workspace-dir workspaces/video-mme-v1 \
    --store-dir store \
    --mode infer \
    --concurrency 12 \
    --max-steps 15 \
    --skill-mode auto \
    --n-samples "${N_SAMPLES:-0}" \
    --questions benchmarks/Video-MME \
    --skills-version v1 \
    --prompts-version v1
