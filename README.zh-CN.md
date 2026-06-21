# FedSDG

[English](README.md) | **简体中文**

这是 **FedSDG: Federated Structure-Decoupled Gating** 的官方 PyTorch 实现，基于 Hydra 管理实验配置。

FedSDG 面向个性化联邦学习中的参数高效微调问题。该方法将 LoRA 适配器解耦为全局共享分支和客户端私有分支，并通过可学习门控自动平衡全局表示与个性化表示，从而在通信效率和个性化性能之间取得更好的折中。

## 主要特点

- **结构解耦的个性化建模**：全局 LoRA 分支参与聚合，私有 LoRA 分支保留在客户端。
- **可学习门控机制**：自动调节全局表示与私有表示的贡献。
- **参数高效联邦训练**：支持基于 ViT 的图像分类实验。
- **Hydra 配置系统**：统一管理算法、数据集、模型、训练和日志配置。
- **内置对比方法**：FedAvg、FedProx、FedLoRA、FedRep、Ditto、FedDPA 和 Local-Only。

## 仓库结构

```text
.
├── conf/              # Hydra 配置文件
├── fl/                # 联邦学习核心代码
├── scripts/           # 最小公开运行脚本
├── datasets/          # 本地数据目录，git 默认忽略
├── logs/              # 运行日志目录，git 默认忽略
├── outputs/           # 实验输出目录，git 默认忽略
├── main.py            # Hydra 主入口
├── options.py         # 旧 argparse 参数兼容层
├── requirements.txt   # Python 依赖
└── environment.yml    # 可选 Conda 环境
```

## 环境安装

创建 Python 3.10 环境：

```bash
conda create -n fedsdg python=3.10
conda activate fedsdg
pip install -r requirements.txt
```

也可以使用 Conda 环境文件：

```bash
conda env create -f environment.yml
conda activate fedsdg
```

如果默认安装的 PyTorch 与本机 CUDA 不匹配，请根据机器环境安装对应版本的 PyTorch。

## 快速开始

运行一个 CPU 冒烟测试：

```bash
bash scripts/smoke_test.sh
```

冒烟测试只用于检查依赖、导入、配置解析和最小训练流程是否正常，不用于复现论文结果。

运行 FedSDG：

```bash
python main.py algorithm=fedsdg dataset=cifar100 model=vit system.gpu=0
```

运行对比方法：

```bash
python main.py algorithm=fedavg dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=fedlora dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=fedrep dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=ditto dataset=cifar100 model=vit system.gpu=0
python main.py algorithm=feddpa dataset=cifar100 model=vit system.gpu=0
```

也可以使用公开脚本：

```bash
GPU=0 DATASET=cifar100 EPOCHS=100 bash scripts/train_fedsdg.sh
```

## 配置说明

Hydra 配置组：

| 配置组 | 可选项 |
| --- | --- |
| `algorithm` | `fedavg`, `fedprox_avg`, `fedlora`, `fedprox_lora`, `fedsdg`, `feddpa`, `fedrep`, `ditto`, `local_only` |
| `dataset` | `mnist`, `fmnist`, `cifar`, `cifar100`, `pathmnist`, `femnist`, `tinyimagenet`, `domainnet`, `officehome`, `imagenet_r` |
| `model` | `mlp`, `cnn`, `vit` |
| `training` | `default`, `fast`, `full` |
| `logger` | `tensorboard`, `wandb`, `none` |

常用覆盖参数示例：

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

## 数据准备

MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100 等 torchvision 小型数据集可以由数据加载器自动下载。

较大的数据集建议放在 `datasets/` 下：

```text
datasets/
├── domainnet/
├── officehome/
├── tinyimagenet/
├── imagenet-r/
└── preprocessed/
```

每个数据集的配置位于 `conf/dataset/*.yaml`。如需修改路径或数据集参数，可以直接编辑 YAML 文件，也可以通过命令行覆盖。

## 输出结果

Hydra 会将每次实验写入 `outputs/`：

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

数据、日志、输出、检查点、模型权重和生成图表默认不会被 git 跟踪。

## 日志

默认使用 TensorBoard：

```bash
tensorboard --logdir outputs
```

Weights & Biases 是可选项：

```bash
python main.py logger=wandb ...
```

默认 WandB 配置为 offline 模式。如需上传实验，请先配置自己的 WandB 账号和运行模式。

## 复现说明

- 使用 `system.seed=<seed>` 控制随机种子。
- 使用 `federated.dirichlet_alpha=<alpha>` 控制标签分布异构性。
- 使用 `federated.frac=<ratio>` 控制每轮客户端参与率。
- `training=fast` 只建议用于快速检查环境或代码入口，不代表正式实验设置。

## 引用

如果本仓库对你的研究有帮助，请引用对应的 FedSDG 论文。论文正式信息确定后，可以在 `CITATION.cff` 中补全引用元数据。

## 开源协议

本项目基于 [Apache License 2.0](LICENSE) 开源。
