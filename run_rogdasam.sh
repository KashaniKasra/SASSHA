#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/finetuning" || exit 1
source ../.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

PROJECT_NAME="rogdasam_glue"

MODEL="squeezebert/squeezebert-uncased"

EPOCHS=50
SEED=0
MOMENTUM=0.9
WEIGHT_DECAY=5e-4
RHO=0.05
ETA_EPS=("0.5" "1.0")
LAMBDAS=("0.01" "0.1" "0.2" "0.5")

BATCH_SIZE=8
EVAL_BATCH_SIZE=16
GPU_FRAC=0.25
NUM_WORKERS=2

MAX_PARALLEL=2

TASKS=("rte" "mrpc" "stsb") # "sst2" "qnli" "mnli" "qqp"

get_lr_for_task() {
  local task="$1"

  case "$task" in
    mrpc)
      echo "3e-4"
      ;;
    rte)
      echo "1e-5"
      ;;
    stsb)
      echo "1e-4"
      ;;
    *)
      echo "1e-5"
      ;;
  esac
}

mkdir -p ../results/rogdasam
mkdir -p ../logs/rogdasam

running_jobs=0

run_one() {
  local task="$1"
  local lambda="$2"
  local eta_eps="$3"
  local lr
  lr=$(get_lr_for_task "$task")

  local outdir="../results/rogdasam/${task}/lambda_${lambda}/eta_${eta_eps}/seed_${SEED}"
  local logfile="../logs/rogdasam/${task}_lambda_${lambda}_eta_${eta_eps}_seed_${SEED}.log"

  rm -rf "$outdir"
  mkdir -p "$outdir"

  echo "============================================================"
  echo "Starting ROGDASAM task=${task}, lambda=${lambda}, eta_eps=${eta_eps}"
  echo "Output: ${outdir}"
  echo "Log: ${logfile}"
  echo "============================================================"

  python finetune.py \
    --model_name_or_path "$MODEL" \
    --task_name "$task" \
    --max_length 512 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --optimizer rogdasam \
    --lr_scheduler_type polynomial \
    --lr "$lr" \
    --momentum "$MOMENTUM" \
    --weight_decay "$WEIGHT_DECAY" \
    --rho "$RHO" \
    --lambda_val "$lambda" \
    --eta_eps "$eta_eps" \
    --seed "$SEED" \
    --gpu_memory_fraction "$GPU_FRAC" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --project_name "$PROJECT_NAME" \
    --output_dir "$outdir" \
    > "$logfile" 2>&1

  status=$?

  if [ "$status" -eq 0 ]; then
    echo "DONE ROGDASAM task=${task}, lambda=${lambda}, eta_eps=${eta_eps}"
  else
    echo "FAILED ROGDASAM task=${task}, lambda=${lambda}, eta_eps=${eta_eps}, status=${status}"
  fi

  return "$status"
}

for task in "${TASKS[@]}"; do
  for lambda in "${LAMBDAS[@]}"; do
    for eta_eps in "${ETA_EPS[@]}"; do
      run_one "$task" "$lambda" "$eta_eps" &

      running_jobs=$((running_jobs + 1))

      if [ "$running_jobs" -ge "$MAX_PARALLEL" ]; then
        wait -n
        running_jobs=$((running_jobs - 1))
      fi
    done
  done
done

wait

echo "All ROGDASAM runs finished."