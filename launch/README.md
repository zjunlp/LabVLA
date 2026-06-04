# LabVLA final launch scripts

This directory keeps one clean entrypoint per training stage:

- `vlm_pretrain/train_vlm_pretrain.sh`
- `ki_posttrain/train_ki_posttrain.sh`
- `finetune/train_labutopia_level3.sh`

All scripts derive `PROJ_ROOT` from this checkout, so they can run from
`/all-flash-data/LabVLA-final` without editing paths.

Multi-node support differs per entry:

- `vlm_pretrain/` and `ki_posttrain/` are multi-node capable: launch the same
  script in tmux on every node with the same `MASTER_ADDR`, `MASTER_PORT`,
  `NUM_MACHINES`, and `JOB_NAME`, but a different `MACHINE_RANK`.
- `finetune/train_labutopia_level3.sh` is **single-machine and bf16-only**. It
  hardcodes `--num_machines 1` and `--mixed_precision bf16` and has no
  `MACHINE_RANK` / `MASTER_ADDR` plumbing. Its `DTYPE` knob sets only the model
  compute dtype (`--dtype`); it does not switch accelerate mixed precision off
  bf16.

Examples:

```bash
# Two-node KI posttrain, rank 0 on 10.1.23.9.
JOB_NAME=labvla_ki_final_$(date +%Y%m%d_%H%M%S) \
NUM_MACHINES=2 MACHINE_RANK=0 MASTER_ADDR=10.1.23.9 MASTER_PORT=29664 \
bash launch/ki_posttrain/train_ki_posttrain.sh

# Matching rank 1 on 10.1.23.8.
JOB_NAME=<same-job-name> \
NUM_MACHINES=2 MACHINE_RANK=1 MASTER_ADDR=10.1.23.9 MASTER_PORT=29664 \
bash launch/ki_posttrain/train_ki_posttrain.sh

# LabUtopia finetune. PRETRAINED_CKPT is required on purpose.
TASK=HeatLiquid PRETRAINED_CKPT=/path/to/checkpoint-50000 \
CUDA_VISIBLE_DEVICES_VAL=1,2,3,4,5,6,7 NUM_GPUS=7 \
bash launch/finetune/train_labutopia_level3.sh
```
