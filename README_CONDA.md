# LabVLA-final Conda 环境安装说明

本文档记录 `/all-flash-data/LabVLA-final` 的训练/部署环境。目标是把当前可跑的
`labvla` 环境用固定版本复现出来。所有 Python 包版本写在根目录
`requirements.txt`，不要临时 `pip install -U` 覆盖这些版本。

## 已验证环境

当前机器上的有效环境：

```text
conda env name : labvla
conda prefix   : /data/rbc/miniconda3/envs/labvla
python         : 3.10.20
torch          : 2.7.1+cu126
torchvision    : 0.22.1+cu126
transformers   : 4.57.6
accelerate     : 1.13.0
deepspeed      : 0.18.8
flash-attn     : 2.8.3
liger-kernel   : 0.7.0
lerobot        : 0.4.4
```

模型默认路径：

```text
/all-flash-data/vlm/Qwen3-VL-4B-Instruct
```

项目默认路径：

```text
/all-flash-data/LabVLA-final
```

## 安装文件

本目录包含两个环境相关文件：

```text
README_CONDA.md      # 本说明
requirements.txt     # 当前 labvla 环境的 pinned pip 包列表
```

`requirements.txt` 是从当前可用的 `labvla` 环境抽取后整理的固定版本列表。两个
原始 conda 本地构建引用已经改成可复现的固定版本：

```text
packaging==25.0
pip==26.0.1
```

`lerobot` 也被固定到当前记录的 commit：

```text
6d34a986de44c5f22a9a99ed514f1b16832c3f32
```

## 历史安装路径

这个环境最早不是直接从空环境装 LabVLA，而是先装 LeRobot，再 clone 成
`labvla` 环境，然后补装 `deepspeed`、`flash-attn`、`transformers`、
`liger-kernel` 等训练需要的库。复现时建议保留这个顺序。

历史路径大致如下：

```bash
source /data/rbc/miniconda3/etc/profile.d/conda.sh

# 1. 先创建 LeRobot 基础环境。
conda create -n lerobot python=3.10.20 -y
conda activate lerobot

# 2. 如果本机有本地 LeRobot 源码，先按 editable 安装。
cd /all-flash-data/lerobot
python -m pip install -e .

# 3. clone 出 LabVLA 环境。
conda create -n labvla --clone lerobot -y
conda activate labvla

# 4. 在 labvla 环境里补齐并锁定 LabVLA 训练/部署依赖。
cd /all-flash-data/LabVLA-final
python -m pip install -r requirements.txt
```

如果目标机器没有 `/all-flash-data/lerobot`，也可以直接用 `requirements.txt` 里的
pinned git commit 安装 LeRobot：

```bash
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda create -n labvla python=3.10.20 -y
conda activate labvla
cd /all-flash-data/LabVLA-final
python -m pip install -r requirements.txt
```

如果目标机器有本地 `/all-flash-data/lerobot`，并且你想严格保持“本地 editable
LeRobot”这一历史形式，可以在装完 requirements 后覆盖一次：

```bash
conda activate labvla
python -m pip uninstall -y lerobot
cd /all-flash-data/lerobot
python -m pip install -e .
```

注意：当前 `/all-flash-data/lerobot` 工作树有本地改动；`requirements.txt` 只能固定
到 git commit，不能表达未提交的 dirty diff。如果这些本地改动也需要复现，应单独把
LeRobot 源码同步到目标机器，然后用上面的 editable 安装。

## 从零安装推荐步骤

### 1. 创建 conda 环境

```bash
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda create -n labvla python=3.10.20 -y
conda activate labvla
```

如果 conda 不在 `/data/rbc/miniconda3`：

```bash
eval "$(conda shell.bash hook)"
conda create -n labvla python=3.10.20 -y
conda activate labvla
```

### 2. 安装 pinned requirements

```bash
cd /all-flash-data/LabVLA-final
python -m pip install -r requirements.txt
```

`requirements.txt` 顶部已经包含：

```text
--extra-index-url https://download.pytorch.org/whl/cu126
```

这是为了拿到和当前环境匹配的 CUDA 12.6 PyTorch wheel。

如果 `flash_attn==2.8.3` 在目标机器上源码编译失败，优先使用当前机器同架构的
wheel/conda 环境复制；其次确认 `torch==2.7.1` 已经安装，再单独执行：

```bash
python -m pip install flash_attn==2.8.3 --no-build-isolation
```

如果 `deepspeed==0.18.8` 编译自定义 op 失败，可以改用：

```bash
DS_BUILD_OPS=0 python -m pip install deepspeed==0.18.8
```

当前训练主路径使用 Accelerate + DeepSpeed ZeRO-2，不依赖手工编译的 DeepSpeed
自定义 op。

### 3. 离线模型检查

训练脚本默认离线运行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

启动训练前检查本地 Qwen3-VL 权重：

```bash
test -d /all-flash-data/vlm/Qwen3-VL-4B-Instruct
test -f /all-flash-data/vlm/Qwen3-VL-4B-Instruct/config.json
```

如果首次下载模型，需要临时取消 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE`，下载完
再恢复离线模式。

## 环境验证

进入项目根目录：

```bash
cd /all-flash-data/LabVLA-final
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate labvla
```

检查 Python、CUDA 和核心库版本：

```bash
python - <<'PY'
import sys
import torch
from importlib import import_module
from importlib.metadata import version

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())

for name in [
    "torchvision", "transformers", "accelerate", "deepspeed", "flash_attn",
    "datasets", "huggingface_hub", "safetensors", "draccus", "pyarrow",
    "pandas", "numpy", "PIL", "av", "websockets", "msgpack", "diffusers",
    "wandb", "einops",
]:
    mod = import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))

print("liger-kernel", version("liger-kernel"))
print("lerobot", version("lerobot"))
PY
```

检查 LabVLA 关键脚本能编译：

```bash
python -m py_compile \
  scripts/train.py \
  deployment/serve_labvla.py \
  src/policies/LabVLA/modeling_labvla.py \
  src/policies/LabVLA/dit_action_head.py
```

检查 `accelerate` / `deepspeed`：

```bash
accelerate env
python - <<'PY'
import deepspeed
print("deepspeed", deepspeed.__version__)
PY
```

## 训练启动示例

LabUtopia Level3 微调：

```bash
cd /all-flash-data/LabVLA-final
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate labvla

TASK=TransportBeaker \
PRETRAINED_CKPT=/path/to/base/checkpoint-50000 \
CUDA_VISIBLE_DEVICES_VAL=1,2,3,4,5,6,7 \
NUM_GPUS=7 \
bash launch/finetune/train_labutopia_level3.sh
```

KI 后训练多机启动时，每台机器都必须安装同一套 `labvla` 环境，并使用相同的
`MASTER_ADDR`、`MASTER_PORT`、`NUM_MACHINES`、`JOB_NAME`，不同的
`MACHINE_RANK`。示例见：

```bash
cat launch/README.md
```

## 部署启动示例

`deployment/deploy.sh` 仍允许用 `LABVLA_ROOT` 指定项目根。建议启动时显式指定：

```bash
cd /all-flash-data/LabVLA-final
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate labvla

LABVLA_ROOT=/all-flash-data/LabVLA-final \
PRETRAINED_PATH=/path/to/checkpoint \
VLM_PATH=/all-flash-data/vlm/Qwen3-VL-4B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
PORT=8000 \
bash deployment/deploy.sh
```

直接运行 Python 服务也可以：

```bash
python deployment/serve_labvla.py \
  --pretrained_path /path/to/checkpoint \
  --vlm_path /all-flash-data/vlm/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda
```

## 同步到 87 服务器

把代码和环境文件同步到 `117.148.167.87`：

```bash
rsync -avP --exclude='outputs' -e 'ssh -p 22' \
  /all-flash-data/LabVLA-final \
  root@117.148.167.87:/data1/LabVLA-code
```

同步后目标路径应为：

```text
/data1/LabVLA-code/LabVLA-final
```

在 87 服务器上安装环境：

```bash
cd /data1/LabVLA-code/LabVLA-final
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda create -n labvla python=3.10.20 -y
conda activate labvla
python -m pip install -r requirements.txt
```

如果 87 服务器上的 conda 不在 `/data/rbc/miniconda3`，用：

```bash
eval "$(conda shell.bash hook)"
```

## 常见问题

### `Qwen3VLProcessor` import 失败

说明 `transformers` 版本不对。必须保持：

```text
transformers==4.57.6
```

### annotation / VLM CE 路径报 `liger_kernel` 缺失

必须保持：

```text
liger_kernel==0.7.0
```

### tmux 或非交互 shell 中 conda 激活失败

优先使用：

```bash
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate labvla
```

如果脚本里用了 `set -u`，在 source conda 前加：

```bash
PS1="${PS1:-}"
```

### 多机训练某台机器 NCCL / CUDA 行为不一致

在每台机器分别执行：

```bash
conda activate labvla
python -m pip freeze --all | sort > /tmp/labvla.freeze.txt
```

把各机器的 `/tmp/labvla.freeze.txt` 做 diff。`torch`、`transformers`、
`accelerate`、`deepspeed`、`flash_attn`、`liger_kernel` 不一致时，不要直接开训。

