#!/usr/bin/env bash
set -euo pipefail

# 收集/拷贝 SeRL 仓库内的「全量」训练集到本地目录（用于后续转 parquet）
#
# 说明：
# - SeRL 的 math 训练集在 openrlhf/dataset/math/ 下
# - “非全量”通常以 sample_500 / 0_2_0_8_train_with_idx_sample_500 命名
# - 本脚本只做文件收集/拷贝，不依赖 Python 环境
#
# 用法：
#   bash scripts/serl_download_full_datasets.sh \
#     --serl-dir /scratch/u6g/zhouyihe.u6g/SeRL \
#     --out-dir  /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/data/serl_raw

SERL_DIR=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serl-dir)
      SERL_DIR="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --serl-dir <SeRL path> --out-dir <output dir>"
      exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1;;
  esac
done

if [[ -z "${SERL_DIR}" ]]; then
  SERL_DIR="/scratch/u6g/zhouyihe.u6g/SeRL"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="/scratch/u6g/zhouyihe.u6g/verl_v0.4.x/data/serl_raw"
fi

if [[ ! -d "${SERL_DIR}" ]]; then
  echo "ERROR: SeRL dir not found: ${SERL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/train" "${OUT_DIR}/eval"

echo "[SeRL] SERL_DIR=${SERL_DIR}"
echo "[SeRL] OUT_DIR=${OUT_DIR}"

# 1) 收集 openrlhf/dataset 下所有 jsonl，排除 sample_500 等“非全量”
DATASET_ROOT="${SERL_DIR}/openrlhf/dataset"
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset dir not found: ${DATASET_ROOT}" >&2
  exit 1
fi

echo "[SeRL] Collecting full training JSONL under: ${DATASET_ROOT}"
found_any=0
while IFS= read -r -d '' f; do
  found_any=1
  rel="${f#${DATASET_ROOT}/}"
  # 保留结构：math/train_with_idx.jsonl -> train/math/train_with_idx.jsonl
  dst="${OUT_DIR}/train/${rel}"
  mkdir -p "$(dirname "${dst}")"
  echo "  cp ${rel} -> train/${rel}"
  cp -f "${f}" "${dst}"
done < <(find "${DATASET_ROOT}" -type f -name "*.jsonl" \
  ! -name "*sample_500*.jsonl" \
  ! -name "*_sample_*.jsonl" \
  -print0)

if [[ "${found_any}" -eq 0 ]]; then
  echo "WARN: no jsonl found under ${DATASET_ROOT}" >&2
fi

# 2) 额外拷贝一个常用 eval：math_500/test_with_idx.jsonl（用于生成 test.parquet）
MATH500="${SERL_DIR}/evaluation/Math-Benchmarks/data/math_500/test_with_idx.jsonl"
if [[ -f "${MATH500}" ]]; then
  echo "[SeRL] Copy eval math_500: test_with_idx.jsonl"
  mkdir -p "${OUT_DIR}/eval/math_500"
  cp -f "${MATH500}" "${OUT_DIR}/eval/math_500/test_with_idx.jsonl"
else
  echo "WARN: eval file not found (skip): ${MATH500}" >&2
fi

echo "[SeRL] Done."

