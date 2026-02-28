#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

GPU=0
WORKERS=2
DATASET=cifar10
ARCH=resnet32
EPOCHS=160
BATCH=256

LRSCHED=multi_step
LR_DECAY_EPOCH_1=80
LR_DECAY_EPOCH_2=120
LR_DECAY=0.1

OPTIM=samsgd
WD=5e-4

OUTDIR=runs
BASE_TAG="samRidge_grid_resnet32_cifar10"

LRS=(0.05 0.1 0.2)
RHOS=(0.05 0.1)
HVP_EVERYS=(1 5 10 20)
SEEDS=(0 1 2 3 4)

for lr in "${LRS[@]}"; do
    for rho in "${RHOS[@]}"; do
        for hvp_every in "${HVP_EVERYS[@]}"; do
            for seed in "${SEEDS[@]}"; do

                RUN_TAG="${BASE_TAG}_lr${lr}_rho${rho}_hvpE${hvp_every}_seed${seed}"

                echo "=== RUN: ${RUN_TAG} ==="

                python "$ROOT/image_classification/train.py" \
                --gpu "${GPU}" --workers "${WORKERS}" \
                --dataset "${DATASET}" -a "${ARCH}" \
                --epochs "${EPOCHS}" -b "${BATCH}" \
                --LRScheduler "${LRSCHED}" \
                --lr-decay-epoch "${LR_DECAY_EPOCH_1}" "${LR_DECAY_EPOCH_2}" \
                --lr-decay "${LR_DECAY}" \
                --optimizer "${OPTIM}" \
                --lr "${lr}" --wd "${WD}" \
                --rho "${rho}" --seed "${seed}" \
                --hvp_every "${hvp_every}" \
                --out-dir "${OUTDIR}" --run-tag "${RUN_TAG}" \
                --log-steps

            done
        done
    done
done