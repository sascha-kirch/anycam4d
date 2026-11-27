"""Every config field is read outside ``anycam4d/config/``.

A field nothing reads is an inert knob. The check is token-based and deliberately generous: a name
appearing anywhere else counts as connected.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.strategy import StrategyConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.config.viewer import ViewerConfig


CONFIG_CLASSES = (
    StrategyConfig,
    TrainingConfig,
    ModelConfig,
    RenderingConfig,
    DatasetConfig,
    ViewerConfig,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _consumer_sources() -> str:
    """Every Python source that could consume a config field, excluding the config layer itself."""
    package_files = [p for p in (_PACKAGE_ROOT / "anycam4d").rglob("*.py") if "config" not in p.parts]
    script_files = list((_PACKAGE_ROOT / "scripts").rglob("*.py"))
    return "\n".join(p.read_text() for p in package_files + script_files)


def _is_referenced(name: str, sources: str) -> bool:
    """Whether ``name`` is read as an attribute or passed as a keyword anywhere in ``sources``."""
    return bool(re.search(rf"\.{name}\b", sources) or re.search(rf"\b{name}\s*=", sources))


def test_every_config_field_has_a_consumer() -> None:
    """No config dataclass declares a field that nothing outside the config layer reads."""
    sources = _consumer_sources()
    dead = {
        f"{cls.__name__}.{field.name}"
        for cls in CONFIG_CLASSES
        for field in dataclasses.fields(cls)
        if not _is_referenced(field.name, sources)
    }
    assert not dead, (
        f"Config fields with no consumer outside anycam4d/config/: {sorted(dead)}. Delete the field or wire it up."
    )
