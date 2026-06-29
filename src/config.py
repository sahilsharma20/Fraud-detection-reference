"""Typed access to ``config.yaml``.

WHY (fresher pitfall):
    The common mistake is ``yaml.safe_load(open("config.yaml"))`` scattered in
    every module, each re-reading the file and each indexing raw dicts with
    string keys (``cfg["paths"]["raw_data"]``) that fail at runtime with a
    KeyError if you typo a key. Here we:
      * load once and cache (``functools.lru_cache``),
      * resolve every path relative to the project root so the code works no
        matter what CWD it's launched from (CI, Docker, your laptop),
      * expose dotted-key access with a clear error if a key is missing.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from src.exception import FraudDetectionError

# Project root = parent of the src/ package. Everything resolves against this so
# that `python -m src.train`, pytest, and the Docker CMD all agree on paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Read-only wrapper around the parsed YAML with dotted-key lookup."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = "__RAISE__") -> Any:
        """Fetch a nested value, e.g. ``cfg.get("paths.raw_data")``.

        Args:
            dotted_key: Dot-separated path into the config tree.
            default: Returned if the key is missing. If left as the sentinel,
                a missing key raises instead of returning silently-wrong None.

        Returns:
            The configured value.

        """
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default != "__RAISE__":
                return default
            else:
                raise FraudDetectionError(
                    f"Missing config key '{dotted_key}' (failed at '{part}')"
                )
        return node

    def path(self, dotted_key: str) -> Path:
        """Like :meth:`get`, but resolves the value to an absolute Path under root."""
        return (self.root / str(self.get(dotted_key))).resolve()

    @property
    def raw(self) -> dict[str, Any]:
        """The underlying dict (for whole-section access)."""
        return self._data


def read_inference_config(cfg: Config | None = None) -> dict[str, Any]:
    """Read the runtime inference config (chosen threshold, model metadata).

    This JSON is WRITTEN during training (by the threshold optimizer and severity
    trainer) and READ at serving time. It is the post-training source of truth for
    values that are computed, not authored — e.g. the cost-optimal threshold.

    Returns:
        The parsed dict, or an empty dict if training hasn't produced it yet.

    """
    import json

    cfg = cfg or load_config()
    path = cfg.path("paths.inference_config")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def update_inference_config(updates: dict[str, Any], cfg: Config | None = None) -> dict[str, Any]:
    """Merge ``updates`` into the runtime inference config and persist it.

    Args:
        updates: Keys to add/overwrite.
        cfg: Optional config.

    Returns:
        The merged config dict.

    """
    import json

    cfg = cfg or load_config()
    path = cfg.path("paths.inference_config")
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_inference_config(cfg)
    current.update(updates)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2)
    return current


@functools.lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Args:
        path: Optional override (used in tests). Defaults to ``<root>/config.yaml``.

    Returns:
        A cached :class:`Config` instance.

    """
    cfg_path = Path(path) if path else _CONFIG_PATH
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FraudDetectionError(f"config.yaml not found at {cfg_path}") from exc
    return Config(data=data, root=PROJECT_ROOT)
