# FedSDG

**English** | [简体中文](README.zh-CN.md)

Official implementation of **FedSDG: Federated Structure-Decoupled Gating**, a personalized federated learning framework built on PyTorch and Hydra.

FedSDG studies parameter-efficient personalization in federated learning. It decouples LoRA adapters into shared and private branches, uses learnable gates to balance global and client-specific representations, and supports communication-efficient aggregation.

## Highlights

- **Structure-decoupled personalization** with shared/private LoRA branches.
- **Learnable gating** for balancing global and private representations.
- **Parameter-efficient federated training** for ViT-based image classification.
- **Hydra configuration system** for algorithms, datasets, models, logging, and training.
- **Baselines included**: FedAvg, FedProx, FedLoRA, FedRep, Ditto, FedDPA, and Local-Only.

## Repository Structure

```text
.
├── conf/              # Hydra configuration files
├── fl/                # Core federated learning library
├── scripts/           # Minimal public run scripts
├── datasets/          # Local datasets, ignored by git
├── logs/              # Runtime logs, ignored by git
├── outputs/           # Experiment outputs, ignored by git
├── main.py            # Main Hydra entry point
├── options.py         # Legacy argparse compatibility layer
├── requirements.txt   # Python dependencies
└── environment.yml    # Optional Conda environment
```

## Installation

Create a Python 3.10 environment:

```bash
conda create -n fedsdg python=3.10
conda activate fedsdg
pip install -r requirements.txt
```

Alternatively, use the provided Conda environment file:

```bash
conda env create -f environment.yml
conda activate fedsdg
```

Install a PyTorch build that matches your CUDA runtime if the default package does not match your machine.

## Quick Start

Run a small CPU smoke test:

```bash
bash scripts/smoke_test.sh
```

The smoke test is intended only to verify the environment, imports, configuration parsing, and a minimal training loop. It is not a reproduction of paper results.

Run FedSDG on CIFAR-100:

```bash
python main.py algorithm=fedsdg dataset=cifar100 model=vit system.gpu=0
```

Run a baseline:

```bash
python main.py algorithm=fedavg dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=fedlora dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=fedrep dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=ditto dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=feddpa dataset=cifar100 model=vit system.gpu=0
```

The public wrapper script can also be used:

```bash
GPU=0 DATASET=cifar100 EPOCHS=100 bash scripts/train_fedsdg.sh
```

## Supported Configurations

Hydra configuration groups:

| Group | Options |
| --- | --- |
| `algorithm` | `fedavg`, `fedprox_avg`, `fedlora`, `fedprox_lora`, `fedsdg`, `feddpa`, `fedrep`, `ditto`, `local_only` |
| `dataset` | `mnist`, `fmnist`, `cifar`, `cifar100`, `pathmnist`, `femnist`, `tinyimagenet`, `domainnet`, `officehome`, `imagenet_r` |
| `model` | `mlp`, `cnn`, `vit` |
| `training` | `default`, `fast`, `full` |
| `logger` | `tensorboard`, `wandb`, `none` |

Example with explicit overrides:

```bash
python main.py \
  algorithm=fedsdg \
  dataset=cifar100 \
  model=vit \
  training.epochs=100 \
  training.local_ep=5 \
  training.local_bs=128 \
  federated.num_users=100 \
  federated.frac=0.1 \
  federated.dirichlet_alpha=0.1 \
  system.seed=42 \
  system.gpu=0
```

## Data Preparation

Small torchvision datasets such as MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 can be downloaded automatically by the data loader.

For larger datasets, place or preprocess data under `datasets/`:

```text
datasets/
├── domainnet/
├── officehome/
├── tinyimagenet/
├── imagenet-r/
└── preprocessed/
```

Dataset-specific settings are defined in `conf/dataset/*.yaml`. Paths can be changed by editing the corresponding YAML file or by overriding values from the command line.

## Outputs

Hydra writes each experiment under `outputs/`:

```text
outputs/YYYY-MM-DD/HHMMSS_<algorithm>_E<epochs>_<dataset>_a<alpha>_seed<seed>/
├── .hydra/
├── main.log
├── console.log
├── tensorboard/
├── metrics.csv
├── final_results.json
└── checkpoint_best.pt
```

Generated outputs, logs, datasets, checkpoints, and model weights are ignored by git.

## Logging

TensorBoard is the default logger:

```bash
tensorboard --logdir outputs
```

Weights & Biases is optional:

```bash
python main.py logger=wandb ...
```

The default WandB configuration uses offline mode. Configure your own WandB account and mode before uploading runs.

## Reproducibility Notes

- Use `system.seed=<seed>` to control the random seed.
- Use `federated.dirichlet_alpha=<alpha>` to control label-distribution heterogeneity.
- Use `federated.frac=<ratio>` to control the client participation rate.
- Use `training=fast` only for environment checks or short dry runs.

## Citation

If you find this repository useful, please cite the corresponding FedSDG paper. Formal citation metadata can be updated in `CITATION.cff` once the paper information is finalized.

## License

This project is released under the [Apache License 2.0](LICENSE).
