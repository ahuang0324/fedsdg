# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
DATASET="${DATASET:-cifar100}"
EPOCHS="${EPOCHS:-100}"
LOCAL_EP="${LOCAL_EP:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
ALPHA="${ALPHA:-0.1}"
FRAC="${FRAC:-0.1}"
LOGGER="${LOGGER:-tensorboard}"

python main.py \
  algorithm=fedsdg \
  dataset="${DATASET}" \
  model=vit \
  training.epochs="${EPOCHS}" \
  training.local_ep="${LOCAL_EP}" \
  training.local_bs="${BATCH_SIZE}" \
  training.lr="${LR}" \
  federated.frac="${FRAC}" \
  federated.dirichlet_alpha="${ALPHA}" \
  system.gpu="${GPU}" \
  logger="${LOGGER}"
