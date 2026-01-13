"""Open the dataloaders as set in the hydra config and log some data analyses."""

from typing import Any, Dict, List, Tuple, cast

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule
from lightning.pytorch.loggers import Logger, MLFlowLogger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mlflow.tracking.client import MlflowClient
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.to_class import to_one_integer_class

log = RankedLogger(__name__, rank_zero_only=True)


def get_frequencies(targets: torch.Tensor, encode=True):
    """Return the unique target class and their occurrences."""
    if encode:
        encoding = to_one_integer_class(targets)  # shape (B,)
    else:
        encoding = targets
    return torch.unique(encoding, return_counts=True)


def plot_histogram(logger: MLFlowLogger, frequencies: torch.Tensor, xlabel: str, name: str):
    fig, ax = cast(Tuple[Figure, Axes], plt.subplots())
    ax.bar(np.arange(len(frequencies)), frequencies.cpu().numpy() / frequencies.sum())
    ax.set_xlabel(xlabel)
    ax.set_xlabel("Frequency in the dataset")
    ax.set_title(f"Frequency of {xlabel} in the dataset.")
    cast(MlflowClient, logger.experiment).log_figure(
        cast(str, logger.run_id), fig, f"plots/{name}.png"
    )


@task_wrapper
def data_analysis(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "logger": logger,
    }

    log.info("Starting analysis!")
    datamodule.prepare_data()
    datamodule.setup("fit")

    NUM_STEM_CLASSES = 8
    train_val_class_frequencies = torch.zeros((NUM_STEM_CLASSES,), dtype=torch.int32)
    train_val_file_presence = torch.zeros(
        (datamodule.train_file_nb,), dtype=torch.int32  # type: ignore
    )
    test_class_frequencies = torch.zeros((NUM_STEM_CLASSES,), dtype=torch.int32)
    test_file_presence = torch.zeros((datamodule.test_file_nb,), dtype=torch.int32)  # type: ignore

    log.info(f"Input shape: {next(iter(datamodule.train_dataloader()))[0][0].shape}")

    for _, targets, source_positions in iter(datamodule.train_dataloader()):
        classes, counts = get_frequencies(targets)
        train_val_class_frequencies[classes] += counts
        classes, counts = get_frequencies(source_positions[:, 0], encode=False)
        train_val_file_presence[classes] += counts

    for _, targets, source_positions in iter(datamodule.val_dataloader()):
        classes, counts = get_frequencies(targets)
        train_val_class_frequencies[classes] += counts
        classes, counts = get_frequencies(source_positions[:, 0], encode=False)
        train_val_file_presence[classes] += counts

    mlflow_logger = cast(MLFlowLogger, logger[0])

    log.info(
        f"Got train/valid class distribution:\n{train_val_class_frequencies / train_val_class_frequencies.sum()}"
    )
    plot_histogram(
        mlflow_logger, train_val_class_frequencies, "output classes", "train_val_class_frequencies"
    )
    log.info(
        f"Got train/valid file presence:\n{train_val_file_presence / train_val_file_presence.sum()}"
    )
    plot_histogram(mlflow_logger, train_val_file_presence, "files", "train_val_file_presence")

    for _, targets, source_positions in iter(datamodule.test_dataloader()):
        classes, counts = get_frequencies(targets)
        test_class_frequencies[classes] += counts
        classes, counts = get_frequencies(source_positions[:, 0], encode=False)
        test_file_presence[classes] += counts

    log.info(
        f"Got test class distribution:\n{test_class_frequencies / test_class_frequencies.sum()}"
    )
    log.info(f"Got test file presence:\n{test_file_presence / test_file_presence.sum()}")
    plot_histogram(
        mlflow_logger, test_class_frequencies, "output classes", "test_class_frequencies"
    )
    plot_histogram(mlflow_logger, test_file_presence, "files", "test_file_presence")

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="data_analysis.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for the datamodule's analysis.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    data_analysis(cfg)


if __name__ == "__main__":
    main()
