#!/bin/bash


LOG_SUBDIR="${LOG_SUBDIR:-fedavg_run_cnn_cifar_dirichlet_50}"

python federated_main.py \
  --dataset cifar \
  --model cnn \
  --epochs 50 \
  --num_users 100 \
  --frac 0.1 \
  --local_ep 5 \
  --local_bs 64 \
  --lr 0.01 \
  --optimizer sgd \
  --dirichlet_alpha 50.0 \
  --log_subdir "${LOG_SUBDIR}" \
  --gpu 2