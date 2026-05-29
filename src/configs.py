"""Assistant configuration — the unit of deployment.

An ``AssistantConfig`` fully describes one assistant: main model, system prompt
(inline content), and (optional) guardrail architecture.

Loaded from either:
- The ``configs/`` directory (dev): one YAML file per config. The filename stem
  is the ``config_id`` (e.g., ``configs/v4.yaml`` has ``config_id="v4"``).
  Prompt-file references inside the YAML are resolved to content at load time.
- An MLflow Model Registry version (production): the manifest is already
  self-contained, with prompts inlined as strings.

Convention: config files in ``configs/`` are append-only. To iterate, create a
new file with a new descriptive name (e.g., ``configs/v4_smaller_classifier.yaml``)
rather than editing an existing one. The Registry holds the immutable record
for promoted versions; local files are a development scratchpad.

The runtime AssistantConfig always carries prompts as strings, never as paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    temperature: float = 0.0


class ClassifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelSpec
    prompt: str  # inline prompt content


class GuardrailNone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["none"]


class GuardrailInputClassifier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_classifier"]
    classifier: ClassifierSpec


class GuardrailSandwich(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sandwich"]
    input_classifier: ClassifierSpec
    output_validator: ClassifierSpec


Guardrail = Annotated[
    Union[GuardrailNone, GuardrailInputClassifier, GuardrailSandwich],
    Field(discriminator="type"),
]


class AssistantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_id: str | None = None       # populated from filename stem by loaders
    display_name: str
    description: str
    model: ModelSpec
    system_prompt: str
    guardrail: Guardrail


def _read_prompt(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = base / p
    return p.read_text(encoding="utf-8")


def _inline_prompts(raw: dict, base: Path) -> dict:
    raw = dict(raw)
    if "system_prompt" in raw:
        raw["system_prompt"] = _read_prompt(raw["system_prompt"], base)
    g = dict(raw.get("guardrail", {}) or {})
    if g.get("type") == "input_classifier":
        cls = dict(g["classifier"])
        cls["prompt"] = _read_prompt(cls["prompt"], base)
        g["classifier"] = cls
    elif g.get("type") == "sandwich":
        ic = dict(g["input_classifier"])
        ic["prompt"] = _read_prompt(ic["prompt"], base)
        g["input_classifier"] = ic
        ov = dict(g["output_validator"])
        ov["prompt"] = _read_prompt(ov["prompt"], base)
        g["output_validator"] = ov
    raw["guardrail"] = g
    return raw


def load_configs(configs_dir: Path | str = "configs") -> dict[str, AssistantConfig]:
    """Load every ``*.yaml`` file in ``configs_dir`` as an AssistantConfig.

    Prompt file references in each YAML are resolved relative to the current
    working directory — run commands from the repo root.
    """
    configs_dir = Path(configs_dir)
    base = Path.cwd()
    result: dict[str, AssistantConfig] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        config_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = _inline_prompts(data, base)
        data["config_id"] = config_id
        result[config_id] = AssistantConfig.model_validate(data)
    return result


def load_config(
    config_id: str, configs_dir: Path | str = "configs"
) -> AssistantConfig:
    configs = load_configs(configs_dir)
    if config_id not in configs:
        raise KeyError(
            f"Config {config_id!r} not found in {configs_dir}. "
            f"Available: {sorted(configs)}"
        )
    return configs[config_id]


def load_config_from_mlflow(run_id: str) -> AssistantConfig:
    """Debug utility: fetch a deployment manifest directly from a specific run."""
    import mlflow

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="config.json"
    )
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AssistantConfig.model_validate(data)


def load_config_from_registry(name: str, alias: str) -> tuple[AssistantConfig, int]:
    """Resolve a Model Registry alias to an AssistantConfig.

    Returns (config, version_number). Raises if the alias is not set.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=name, alias=alias)
    local_path = mlflow.artifacts.download_artifacts(mv.source)
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AssistantConfig.model_validate(data), int(mv.version)
