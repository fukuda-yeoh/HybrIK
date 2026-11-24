"""Training CLI and experiment setup.

Importing this module has no side effects. The training scripts call
:func:`configure_experiment` from their ``__main__`` block. For backwards
compatibility, ``from hybrik.opt import opt, cfg, logger`` still works and
lazily configures the experiment on first attribute access.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import MethodType
from typing import Any

from .utils.config import update_config

__all__ = [
    "build_parser",
    "parse_args",
    "configure_experiment",
    "setup_logger",
]

_state: tuple[argparse.Namespace, Any, logging.Logger] | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser without reading ``sys.argv``."""
    parser = argparse.ArgumentParser(description="HybrIK Training")

    # ------------------------- Experiment options -------------------------
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--exp-id",
        default="default",
        type=str,
        help="Experiment ID",
    )

    # ---------------------------- General options -------------------------
    parser.add_argument(
        "--nThreads",
        default=10,
        type=int,
        help="Number of data loading threads",
    )
    parser.add_argument(
        "--snapshot",
        default=2,
        type=int,
        help="How often to take a snapshot of the model (0 = never)",
    )
    parser.add_argument(
        "--rank",
        default=-1,
        type=int,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        type=str,
        help="distributed backend",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )

    # ---------------------------- Training options ------------------------
    parser.add_argument(
        "--sync",
        default=False,
        dest="sync",
        help="Use Sync Batchnorm",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default=23333,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--dynamic-lr",
        default=False,
        dest="dynamic_lr",
        help="dynamic lr scheduler",
        action="store_true",
    )
    parser.add_argument(
        "--exp-lr",
        default=False,
        dest="exp_lr",
        help="Exponential lr scheduler",
        action="store_true",
    )

    # ----------------------------- Log options ----------------------------
    parser.add_argument(
        "--board",
        default=True,
        dest="board",
        help="Logging with tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        default=False,
        dest="debug",
        help="Visualization debug",
        action="store_true",
    )
    parser.add_argument(
        "--params",
        default=False,
        dest="params",
        help="Logging params",
        action="store_true",
    )
    parser.add_argument(
        "--map",
        default=True,
        dest="map",
        help="Evaluate mAP per epoch",
        action="store_true",
    )
    parser.add_argument(
        "--flip-test",
        default=True,
        dest="flip_test",
        help="flip test",
        action="store_true",
    )
    parser.add_argument(
        "--flip-shift",
        default=False,
        dest="flip_shift",
        help="flip shift",
        action="store_true",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training arguments, defaulting to ``sys.argv``."""
    return build_parser().parse_args(argv)


def _epoch_info(self, set, idx, loss, acc):
    self.info(
        "{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}".format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc,
        )
    )


def setup_logger(work_dir: str | os.PathLike[str], level: int = logging.INFO):
    """Create ``work_dir`` and return a configured ``hybrik`` logger."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hybrik")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if the experiment is configured more than once.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    filehandler = logging.FileHandler(work_dir / "training.log")
    streamhandler = logging.StreamHandler()
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.epochInfo = MethodType(_epoch_info, logger)
    return logger


def configure_experiment(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, Any, logging.Logger]:
    """Parse arguments, load the config and create the experiment work dir."""
    opt = parse_args(argv)
    cfg = update_config(opt.cfg)

    cfg["FILE_NAME"] = Path(opt.cfg).name
    cfg.TRAIN.DPG_STEP = [
        i - cfg.TRAIN.DPG_MILESTONE for i in cfg.TRAIN.DPG_STEP
    ]

    # Import torch only when an experiment is configured so that importing
    # ``hybrik.opt`` stays cheap and does not require PyTorch.
    import torch

    num_gpu = torch.cuda.device_count()
    if cfg.TRAIN.WORLD_SIZE > num_gpu:
        cfg.TRAIN.WORLD_SIZE = num_gpu
    opt.world_size = cfg.TRAIN.WORLD_SIZE

    opt.work_dir = os.path.join(
        "exp",
        cfg.DATASET.DATASET,
        f"{cfg.FILE_NAME}-{opt.exp_id}",
    ) + os.sep
    logger = setup_logger(opt.work_dir)

    return opt, cfg, logger


def __getattr__(name: str):
    """Lazily configure the experiment for legacy ``from hybrik.opt import ...``."""
    global _state
    if name not in {"opt", "cfg", "logger"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if _state is None:
        _state = configure_experiment()

    return {"opt": _state[0], "cfg": _state[1], "logger": _state[2]}[name]


def __dir__():
    return sorted(set(globals()) | {"opt", "cfg", "logger"})
