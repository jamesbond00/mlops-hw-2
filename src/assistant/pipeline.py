"""Assistant pipeline — one shape with optional pre/post guardrails.

A request goes through up to three model calls:
    1. Optional input classifier (refuses fast if non-travel).
    2. Main assistant.
    3. Optional output validator (replaces output with canned refusal if it leaked).

The AssistantConfig's guardrail field controls which optional stages are
present. The response carries one ModelCall per stage executed, for
downstream Prometheus metrics and MLflow logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from src.assistant.types import AssistantResponse, ModelCall
from src.configs import (
    AssistantConfig,
    ClassifierSpec,
    GuardrailInputClassifier,
    GuardrailSandwich,
    ModelSpec,
)
from src.constants import CANNED_REFUSAL
from src.nebius_client import get_client


@dataclass(frozen=True)
class _Stage:
    model: ModelSpec
    system_prompt: str
    role: str
    enable_thinking: bool


def _call(client: OpenAI, stage: _Stage, user_content: str) -> tuple[str, ModelCall]:
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=stage.model.name,
        messages=[
            {"role": "system", "content": stage.system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=stage.model.temperature,
        # Nemotron is a reasoning model. The gates (classifier/validator) emit a
        # single terse label and must not burn tokens thinking — thinking off.
        # The main assistant needs to reason about answer-vs-refuse, so it keeps
        # thinking on; without it, a strict refusal prompt over-refuses real
        # travel questions.
        extra_body={"chat_template_kwargs": {"enable_thinking": stage.enable_thinking}},
    )
    elapsed = time.perf_counter() - start
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    call = ModelCall(
        model=stage.model.name,
        role=stage.role,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_seconds=elapsed,
    )
    return text, call


def _parse_input_category(raw: str) -> str:
    cleaned = raw.strip().strip('"\'').lower()
    for label in ("travel", "off_topic", "suspicious"):
        if cleaned == label or cleaned.startswith(label):
            return label
    return "suspicious"  # fail-closed: unparseable -> treat as suspicious


def _parse_output_verdict(raw: str) -> str:
    cleaned = raw.strip().strip('"\'').lower()
    for label in ("ok", "leaked"):
        if cleaned == label or cleaned.startswith(label):
            return label
    return "leaked"  # fail-closed: unparseable -> treat as leaked


class Pipeline:
    """Optional input classifier -> main assistant -> optional output validator."""

    def __init__(
        self,
        *,
        main: _Stage,
        input_classifier: _Stage | None,
        output_validator: _Stage | None,
        client: OpenAI,
    ):
        self._main = main
        self._input_classifier = input_classifier
        self._output_validator = output_validator
        self._client = client

    def respond(self, user_message: str) -> AssistantResponse:
        calls: list[ModelCall] = []
        category: str | None = None

        # 1. Input classifier (optional)
        if self._input_classifier is not None:
            raw, call = _call(self._client, self._input_classifier, user_message)
            calls.append(call)
            category = _parse_input_category(raw)
            if category != "travel":
                return AssistantResponse(
                    text=CANNED_REFUSAL,
                    refused=True,
                    input_category=category,
                    output_verdict=None,
                    model_calls=calls,
                )

        # 2. Main assistant
        raw_text, main_call = _call(self._client, self._main, user_message)
        calls.append(main_call)
        text = raw_text.strip()

        # 3. Output validator (optional)
        verdict: str | None = None
        if self._output_validator is not None:
            raw, val_call = _call(self._client, self._output_validator, text)
            calls.append(val_call)
            verdict = _parse_output_verdict(raw)
            if verdict == "leaked":
                return AssistantResponse(
                    text=CANNED_REFUSAL,
                    refused=True,
                    input_category=category,
                    output_verdict=verdict,
                    model_calls=calls,
                )

        return AssistantResponse(
            text=text,
            refused=text == CANNED_REFUSAL,
            input_category=category,
            output_verdict=verdict,
            model_calls=calls,
        )


def _classifier_stage(spec: ClassifierSpec, role: str) -> _Stage:
    return _Stage(
        model=spec.model,
        system_prompt=spec.prompt,
        role=role,
        enable_thinking=False,  # cheap one-label gate; reasoning would only add cost
    )


def build_pipeline(config: AssistantConfig) -> Pipeline:
    """Construct a Pipeline from an AssistantConfig. Prompts are taken from
    config.system_prompt and config.guardrail.*.prompt as strings — they
    were inlined at load time."""
    client = get_client()
    main = _Stage(
        model=config.model,
        system_prompt=config.system_prompt,
        role="main_assistant",
        enable_thinking=True,  # must reason about answer-vs-refuse on each request
    )

    g = config.guardrail
    input_classifier: _Stage | None = None
    output_validator: _Stage | None = None

    if isinstance(g, GuardrailInputClassifier):
        input_classifier = _classifier_stage(g.classifier, "input_classifier")
    elif isinstance(g, GuardrailSandwich):
        input_classifier = _classifier_stage(g.input_classifier, "input_classifier")
        output_validator = _classifier_stage(g.output_validator, "output_validator")
    # GuardrailNone: both stages remain None.

    return Pipeline(
        main=main,
        input_classifier=input_classifier,
        output_validator=output_validator,
        client=client,
    )
