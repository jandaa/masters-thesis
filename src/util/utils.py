import sys
import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch


def get_batch_offsets(batch_idxs, bs):
    """
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def get_cli_override_arguments():
    """Returns a list of config arguements being overriden from the current command line"""
    override_args = []
    for arg in sys.argv:
        arg.replace(" ", "")
        if "=" in arg:
            override_args.append(arg.split("=")[0])

    return override_args


def add_previous_override_args_to_cli(previous_cli_override):
    """Adds override arguments to the cli if they are not already overriden"""
    for override in previous_cli_override:
        override_key, override_value = override.split("=", 1)
        if override_key not in previous_cli_override:
            sys.argv.append(override)


def load_previous_config(previous_dir: Path, current_dir: Path) -> DictConfig:

    for copy_folder in [".hydra", "lightning_logs"]:
        src_dir = previous_dir / copy_folder
        dest_dir = current_dir / copy_folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)

    # Load overrides config and convert to Dict
    # with command line arguments taking precedence
    override_args = get_cli_override_arguments()

    overrides_cfg = OmegaConf.load(str(current_dir / ".hydra/overrides.yaml"))
    add_previous_override_args_to_cli(overrides_cfg)
    # save_current_overrides()

    cli_conf = OmegaConf.from_cli()
    main_cfg = OmegaConf.load(str(current_dir / ".hydra/config.yaml"))
    conf = OmegaConf.merge(main_cfg, cli_conf)
    print(OmegaConf.to_yaml(conf))


def load_previous_training(previous_dir: Path, current_dir: Path) -> DictConfig:

    for copy_folder in ["lightning_logs"]:
        src_dir = previous_dir / copy_folder
        dest_dir = current_dir / copy_folder
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)


def print_error(message, user_fault=False):
    sys.stderr.write("ERROR: " + str(message) + "\n")
    if user_fault:
        sys.exit(2)
    sys.exit(-1)
