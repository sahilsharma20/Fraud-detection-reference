"""Thin MLflow setup helper shared by the fraud and severity trainers.

Centralising this avoids copy-pasting tracking-URI boilerplate into both model
modules (and getting it subtly different — a classic source of "why are my runs
in two different folders?" confusion).
"""

from __future__ import annotations

import mlflow

from src.config import Config
from src.logger import get_logger

log = get_logger(__name__)


def setup_mlflow(cfg: Config, experiment: str) -> None:
    """Point MLflow at the local file store and select an experiment.

    Uses a ``file://`` URI built from an absolute path so it resolves identically
    whether launched from the repo root, pytest, or Docker.

    Args:
        cfg: Loaded config.
        experiment: Experiment name to create/activate.

    """
    tracking_dir = cfg.path("mlflow.tracking_uri")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment)
    log.info("MLflow tracking_uri=%s experiment=%s", tracking_dir.as_uri(), experiment)
