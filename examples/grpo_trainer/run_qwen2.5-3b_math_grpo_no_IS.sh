    #!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
# verl GRPO Example: Llama-3.2-3B-Instruct (Math)
# - 风格尽量仿照 ASPO/Archer2.0 的训练脚本（变量集中在顶部 + 可用环境变量覆盖）
# - 入口保持 verl：python -m verl.trainer.main_ppo
# ============================================================

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 固定使用 conda 的 verl 环境 python（避免 notebook/jupyter 里 python 指到别的环境）
PYTHON_BIN="${PYTHON_BIN:-/mnt/iusers01/fatpou01/compsci01/h99859yz/miniconda3/envs/verl/bin/python}"

nnodes="${NNODES:-1}"



# 数据（默认用仓库内的 ./data，不放在 $HOME 下）
data_root="${DATA_ROOT:-${ROOT_DIR}/data}"
gsm8k_train_path="${GSM8K_TRAIN_PATH:-$data_root/gsm8k/train.parquet}"
gsm8k_test_path="${GSM8K_TEST_PATH:-$data_root/gsm8k/test.parquet}"
math_train_path="${MATH_TRAIN_PATH:-$data_root/math/train.parquet}"
math_test_path="${MATH_TEST_PATH:-$data_root/math/test.parquet}"

# 默认：只跑 GSM8K（训练/测试）；可用环境变量 TRAIN_FILES/TEST_FILES 覆盖
train_files="${TRAIN_FILES:-['$gsm8k_train_path']}"
test_files="${TEST_FILES:-['$gsm8k_test_path']}"

# 模型（对齐 ASPO：Llama 3B）
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"

# 长度配置
max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"

# batch 配置（对齐 ASPO Llama 脚本的口径：train=64, mini=16, micro=16）
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-16}"
ppo_epochs="${PPO_EPOCHS:-3}"

project_name="${PROJECT_NAME:-verl_grpo_example}"
exp_name="${EXP_NAME:-llama3.2_3b_no_IS_gsm8k_grpo_epochs_${ppo_epochs}}"


# Algorithm
adv_estimator="${ADV_ESTIMATOR:-grpo}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

# Validation（用于 pass@k / best@k 评估）
# 说明：
# - 要测 pass@k(k>1)，必须 do_sample=True 且 temperature>0，并设置 val_n>=最大k（例如 16）
val_n="${VAL_N:-16}"
val_do_sample="${VAL_DO_SAMPLE:-True}"
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-1.0}"
val_top_k="${VAL_TOP_K:--1}"
# 默认只评估 10% 的验证集来加速（可用环境变量覆盖）
val_subset_ratio="${VAL_SUBSET_RATIO:-1.0}"
val_subset_seed="${VAL_SUBSET_SEED:-42}"
val_subset_resample_each_eval="${VAL_SUBSET_RESAMPLE_EACH_EVAL:-False}"

# KL config
use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
kl_coef="${KL_COEF:-0.0}"
use_kl_loss="${USE_KL_LOSS:-True}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-low_var_kl}"

# clip（verl PPO/GRPO 仍会用到 actor 的 clip）
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
clip_ratio_c="${CLIP_RATIO_C:-3.0}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"

# 是否使用重要性采样（PPO 默认 True；设 False 会走我们刚加的 no-IS 分支）
use_importance_sampling="${USE_IMPORTANCE_SAMPLING:-False}"

# 性能相关参数（可按机器情况覆盖）
sp_size="${SP_SIZE:-1}"
gen_tp="${GEN_TP:-1}"
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
offload="${OFFLOAD:-False}"

# 日志/输出
out_dir="${OUT_DIR:-./outputs/${project_name}/${exp_name}}"
mkdir -p "${out_dir}"

echo "============================================================"
echo "[verl][GRPO] Llama-3.2-3B"
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
echo "============================================================"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${adv_estimator}" \
  data.train_files="${train_files}" \
  data.val_files="${test_files}" \
  data.train_batch_size="${train_prompt_bsz}" \
  +data.dataloader_num_workers=2 \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
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
  "$@" 2>&1 | tee "${out_dir}/${project_name}_${exp_name}_grpo.log"


