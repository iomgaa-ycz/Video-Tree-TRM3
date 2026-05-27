#!/usr/bin/env bash
# 诊断分析：对 Video-MME 推理结果执行两阶段诊断
# 用法:
#   bash scripts/diagnose_video_mme.sh                              # 默认：最近一次 run，全量
#   RUN_ID=20260526_135810 bash scripts/diagnose_video_mme.sh       # 指定 run
#   TASK_TYPES="Temporal Reasoning" bash scripts/diagnose_video_mme.sh  # 只诊断某题型
#   ONLY_INCORRECT=1 bash scripts/diagnose_video_mme.sh             # 只诊断错题
#   ONLY_INCORRECT=1 TASK_TYPES="OCR Problems" bash scripts/diagnose_video_mme.sh  # 组合筛选
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

set -a
source .env
set +a

PYTHON="$(conda run -n Video-Tree-TRM which python)"

# 构建可选参数
EXTRA_ARGS=()

if [[ -n "${RUN_ID:-}" ]]; then
    EXTRA_ARGS+=(--run-id "${RUN_ID}")
fi

if [[ -n "${TASK_TYPES:-}" ]]; then
    # 支持空格分隔多个题型：TASK_TYPES="Temporal Reasoning,OCR Problems"
    IFS=',' read -ra TYPES <<< "${TASK_TYPES}"
    EXTRA_ARGS+=(--task-types "${TYPES[@]}")
fi

if [[ "${ONLY_INCORRECT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--only-incorrect)
fi

"${PYTHON}" main.py \
    --workspace-dir workspaces/video-mme-v1 \
    --store-dir store \
    --mode diagnose \
    --concurrency "${CONCURRENCY:-4}" \
    --max-steps 15 \
    --skill-mode auto \
    --n-samples 0 \
    --questions benchmarks/Video-MME \
    --skills-version v1 \
    --prompts-version v1 \
    "${EXTRA_ARGS[@]}"
