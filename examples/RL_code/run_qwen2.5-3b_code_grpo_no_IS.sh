#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
# verl GRPO Example (no-IS): Qwen2.5 3B (Code)
# - 基于 examples/grpo_trainer/run_qwen2.5-3b_math_grpo_no_IS.sh 的脚本风格
# - 与 run_qwen2.5-3b_code_grpo.sh 唯一区别：use_importance_sampling=False
# ============================================================

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/iusers01/fatpou01/compsci01/h99859yz/miniconda3/envs/verl/bin/python}"
nnodes="${NNODES:-1}"

data_root="${DATA_ROOT:-${ROOT_DIR}/data}"
code_train_path="${CODE_TRAIN_PATH:-$data_root/code/train.parquet}"
code_test_path="${CODE_TEST_PATH:-$data_root/code/test.parquet}"

train_files="${TRAIN_FILES:-['$code_train_path']}"
test_files="${TEST_FILES:-['$code_test_path']}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-3072}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-4}"
ppo_epochs="${PPO_EPOCHS:-3}"

project_name="${PROJECT_NAME:-verl_grpo_example_code}"
exp_name="${EXP_NAME:-qwen2.5_coder_3b_code_no_IS_grpo_epochs_${ppo_epochs}}"

adv_estimator="${ADV_ESTIMATOR:-grpo}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-4}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

val_n="${VAL_N:-8}"
val_do_sample="${VAL_DO_SAMPLE:-True}"
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-1.0}"
val_top_k="${VAL_TOP_K:--1}"
val_subset_ratio="${VAL_SUBSET_RATIO:-1.0}"
val_subset_seed="${VAL_SUBSET_SEED:-42}"
val_subset_resample_each_eval="${VAL_SUBSET_RESAMPLE_EACH_EVAL:-False}"

use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
kl_coef="${KL_COEF:-0.0}"
use_kl_loss="${USE_KL_LOSS:-True}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
clip_ratio_c="${CLIP_RATIO_C:-3.0}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"

# no-IS
use_importance_sampling="${USE_IMPORTANCE_SAMPLING:-False}"

sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
offload="${OFFLOAD:-False}"
rollout_name="${ROLLOUT_NAME:-vllm}"

sandbox_fusion_url="${SANDBOX_FUSION_URL:-}"
sandbox_fusion_max_concurrent="${SANDBOX_FUSION_MAX_CONCURRENT:-64}"

out_dir="${OUT_DIR:-./outputs/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

extra_args=()
if [[ -n "${sandbox_fusion_url}" ]]; then
  extra_args+=("reward_model.sandbox_fusion.url=${sandbox_fusion_url}")
  extra_args+=("reward_model.sandbox_fusion.max_concurrent=${sandbox_fusion_max_concurrent}")
else
  extra_args+=("reward_model.sandbox_fusion.url=null")
fi

echo "============================================================"
echo "[verl][GRPO][no-IS] Qwen2.5 3B (Code)"
echo "project_name=${project_name}"
echo "exp_name=${exp_name}"
echo "model=${MODEL_PATH}"
echo "train_files=${train_files}"
echo "max_prompt_length=${max_prompt_length}, max_response_length=${max_response_length}"
echo "train_bsz=${train_prompt_bsz}, mini_bsz=${train_prompt_mini_bsz}, micro_bsz/gpu=${micro_batch_size_per_gpu}"
echo "ppo_epochs=${ppo_epochs}"
echo "n=${n_resp_per_prompt}, temp=${temperature}, top_p=${top_p}, top_k=${top_k}"
echo "val_n=${val_n}, val_do_sample=${val_do_sample}, val_temp=${val_temperature}, val_top_p=${val_top_p}, val_top_k=${val_top_k}"
echo "val_subset_ratio=${val_subset_ratio}"
echo "val_subset_seed=${val_subset_seed}, val_subset_resample_each_eval=${val_subset_resample_each_eval}"
echo "use_importance_sampling=${use_importance_sampling}"
echo "sandbox_fusion_url=${sandbox_fusion_url:-<local prime_code>}"
echo "============================================================"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size="${train_prompt_bsz}" \
  data.dataloader_num_workers=2 \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
  actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
  actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
  algorithm.use_kl_in_reward="${use_kl_in_reward}" \
  algorithm.kl_ctrl.kl_coef="${kl_coef}" \
  actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
  actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
  actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
  actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
  actor_rollout_ref.actor.use_importance_sampling="${use_importance_sampling}" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
  actor_rollout_ref.rollout.name="${rollout_name}" \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.rollout.temperature="${temperature}" \
  actor_rollout_ref.rollout.top_p="${top_p}" \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.val_kwargs.n="${val_n}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${val_do_sample}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
  trainer.val_subset_ratio="${val_subset_ratio}" \
  trainer.val_subset_seed="${val_subset_seed}" \
  trainer.val_subset_resample_each_eval="${val_subset_resample_each_eval}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-2}" \
  trainer.nnodes="${nnodes}" \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-4}" \
  trainer.default_local_dir="${out_dir}" \
  "${extra_args[@]}" \
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_grpo_no_IS.log"


