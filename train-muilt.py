"""
Usage:
Training:
python train.py --config-name=train_act_image_workspace
"""

import sys

import pytorch_lightning
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from policy.workspace.lightning_workspace import LightningWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

def _hydra_subprocess_cmd(local_rank: int):
    """
    Monkey patching for lightning.fabric.strategies.launchers.subprocess_script._hydra_subprocess_cmd
    Temporarily fixes the problem of unnecessarily creating log folders for DDP subprocesses in Hydra multirun/sweep.
    """
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    from hydra.core.hydra_config import HydraConfig
    from hydra.types import RunMode
    from hydra.utils import get_original_cwd, to_absolute_path

    # when user is using hydra find the absolute path
    if __main__.__spec__ is None:  # pragma: no-cover
        command = [sys.executable, to_absolute_path(sys.argv[0])]
    else:
        command = [sys.executable, "-m", __main__.__spec__.name]

    command += sys.argv[1:] + [f"hydra.job.name=train_ddp_process_{local_rank}", "hydra.output_subdir=null"]

    cwd = get_original_cwd()
    config = HydraConfig.get()

    if config.mode == RunMode.RUN:
        command += [f'hydra.run.dir="{config.run.dir}"']
    elif config.mode == RunMode.MULTIRUN:
        command += [f'hydra.sweep.dir="{config.sweep.dir}"']

    print(command)

    return command, cwd

pytorch_lightning.strategies.launchers.subprocess_script._hydra_subprocess_cmd = _hydra_subprocess_cmd

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'policy','config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    print(cfg._target_)

    cls = hydra.utils.get_class(cfg._target_)
    
    workspace: LightningWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
