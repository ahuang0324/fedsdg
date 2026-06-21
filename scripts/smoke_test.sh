# Copyright (c) JLU, NODI, Huimin Huang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python main.py \
  algorithm=fedsdg \
  dataset=cifar100 \
  model=vit \
  model.variant=scratch \
  training=fast \
  training.epochs=1 \
  training.local_ep=1 \
  training.local_bs=8 \
  training.num_workers=0 \
  federated.num_users=5 \
  federated.frac=0.4 \
  federated.dirichlet_alpha=0.5 \
  system.gpu=-1 \
  logger=none \
  training.experiment_note="_smoke_test"
