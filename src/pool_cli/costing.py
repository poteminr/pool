from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None


DEFAULT_MODEL_PRICING = {
    "gpt-4.1-mini": ModelPricing(input_per_1m=0.40, output_per_1m=1.60, cached_input_per_1m=0.10),
    "gpt-4.1": ModelPricing(input_per_1m=2.00, output_per_1m=8.00, cached_input_per_1m=0.50)
}


def resolve_model_pricing(
    model: str,
    input_per_1m: float | None = None,
    cached_input_per_1m: float | None = None,
    output_per_1m: float | None = None,
) -> ModelPricing | None:
    base = DEFAULT_MODEL_PRICING.get(model.strip())
    if base is None and (input_per_1m is None or output_per_1m is None):
        return None

    if base is None:
        return ModelPricing(
            input_per_1m=float(input_per_1m),
            output_per_1m=float(output_per_1m),
            cached_input_per_1m=float(cached_input_per_1m) if cached_input_per_1m is not None else None,
        )

    return ModelPricing(
        input_per_1m=float(input_per_1m) if input_per_1m is not None else base.input_per_1m,
        output_per_1m=float(output_per_1m) if output_per_1m is not None else base.output_per_1m,
        cached_input_per_1m=(
            float(cached_input_per_1m)
            if cached_input_per_1m is not None
            else base.cached_input_per_1m
        ),
    )


def estimate_openai_cost_usd(
    usage: Mapping[str, int],
    pricing: ModelPricing | None,
) -> float | None:
    if pricing is None:
        return None

    input_tokens = max(0, int(usage.get("input_tokens", 0)))
    output_tokens = max(0, int(usage.get("output_tokens", 0)))
    cached_input_tokens = max(0, int(usage.get("cached_input_tokens", 0)))
    cached_input_tokens = min(cached_input_tokens, input_tokens)

    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    cached_rate = (
        pricing.cached_input_per_1m
        if pricing.cached_input_per_1m is not None
        else pricing.input_per_1m
    )

    input_cost = (uncached_input_tokens / 1_000_000.0) * pricing.input_per_1m
    cached_input_cost = (cached_input_tokens / 1_000_000.0) * cached_rate
    output_cost = (output_tokens / 1_000_000.0) * pricing.output_per_1m
    return float(input_cost + cached_input_cost + output_cost)


def format_cost_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1.0:
        return f"${value:.4f}"
    return f"${value:.2f}"
