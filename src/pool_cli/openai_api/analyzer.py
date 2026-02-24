from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image

from pool_cli.pools import PoolDefinition

from .prompts import (
    POOL_ACTION_SCHEMA_NAME,
    POOL_DEFINITIONS_PREFIX,
    USER_POOL_SCHEMA_NAME,
    build_pool_action_schema,
    build_pool_action_system_prompt,
    build_pool_catalog,
    build_user_pool_schema,
    build_user_pool_system_prompt,
)

from openai import OpenAI


@dataclass(slots=True)
class OpenAIUsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class OpenAIUserPoolMetadata:
    name: str
    description: str
    action_title: str
    why: str


@dataclass(slots=True)
class OpenAIPoolActionSuggestion:
    action_title: str
    why: str
    notes: str
    confidence: float


def is_openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _image_to_data_url(path: Path, max_side: int = 768) -> str | None:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((max_side, max_side))
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


class OpenAIPoolAnalyzer:
    def __init__(
        self,
        model: str,
        timeout_seconds: int,
        pools: list[PoolDefinition],
        log: Callable[[str], None],
    ) -> None:
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = model
        self._timeout_seconds = max(10, timeout_seconds)
        self._log = log
        self._stats_lock = threading.Lock()
        self.stats = OpenAIUsageStats()
        self._existing_pool_names: set[str] = set()
        self._pool_catalog_text = ""
        self._user_pool_system_prompt = build_user_pool_system_prompt()
        self._user_pool_text_format = {
            "format": {
                "type": "json_schema",
                "name": USER_POOL_SCHEMA_NAME,
                "strict": True,
                "schema": build_user_pool_schema(),
            }
        }
        self._pool_action_system_prompt = build_pool_action_system_prompt()
        self._pool_action_text_format = {
            "format": {
                "type": "json_schema",
                "name": POOL_ACTION_SCHEMA_NAME,
                "strict": True,
                "schema": build_pool_action_schema(),
            }
        }
        self.set_runtime_pools(pools)

    def set_runtime_pools(self, pools: list[PoolDefinition]) -> None:
        self._existing_pool_names = {pool.name for pool in pools}
        self._pool_catalog_text = POOL_DEFINITIONS_PREFIX + build_pool_catalog(pools)

    def _record_usage(self, response) -> None:
        if response.usage:
            details = response.usage.input_tokens_details
            with self._stats_lock:
                self.stats.input_tokens += response.usage.input_tokens
                self.stats.output_tokens += response.usage.output_tokens
                self.stats.cached_input_tokens += details.cached_tokens if details else 0

    def _request_json(
        self,
        *,
        system_prompt: str,
        content: list[dict],
        text_format: dict,
        include_pool_catalog: bool,
    ) -> dict | None:
        try:
            user_content: list[dict] = []
            if include_pool_catalog:
                user_content.append({"type": "input_text", "text": self._pool_catalog_text})
            user_content.extend(content)
            response = self._client.with_options(timeout=self._timeout_seconds).responses.create(
                model=self._model,
                temperature=0,
                text=text_format,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            )
        except Exception as exc:
            self._log(f"OpenAI request failed: {type(exc).__name__}: {exc}")
            return None
        self._record_usage(response)

        try:
            payload = json.loads(response.output_text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def suggest_user_pool(
        self,
        cluster_summary: str,
        sample_paths: list[Path],
    ) -> OpenAIUserPoolMetadata | None:
        content: list[dict] = [{"type": "input_text", "text": cluster_summary}]
        for path in sample_paths[:6]:
            image_data = _image_to_data_url(path)
            if image_data:
                content.append({"type": "input_image", "image_url": image_data})
        parsed = self._request_json(
            system_prompt=self._user_pool_system_prompt,
            content=content,
            text_format=self._user_pool_text_format,
            include_pool_catalog=True,
        )
        if parsed is None:
            return None

        return OpenAIUserPoolMetadata(
            name=str(parsed.get("name", "")),
            description=str(parsed.get("description", "")),
            action_title=str(parsed.get("action_title", "")),
            why=str(parsed.get("why", "")),
        )

    def suggest_pool_action(
        self,
        *,
        pool: PoolDefinition,
        match_count: int,
        sample_paths: list[Path],
    ) -> OpenAIPoolActionSuggestion | None:
        summary_lines = [
            f"Pool name: {pool.name}",
            f"Pool type: {pool.pool_type}",
            f"Pool description: {pool.description}",
            f"Pool matches: {max(0, int(match_count))}",
        ]

        content: list[dict] = [
            {"type": "input_text", "text": "\n".join(summary_lines)}
        ]
        for path in sample_paths[:6]:
            image_data = _image_to_data_url(path)
            if image_data:
                content.append({"type": "input_image", "image_url": image_data})
        parsed = self._request_json(
            system_prompt=self._pool_action_system_prompt,
            content=content,
            text_format=self._pool_action_text_format,
            include_pool_catalog=False,
        )
        if parsed is None:
            return None

        action_title = str(parsed.get("action_title", "")).strip()[:90]
        why = str(parsed.get("why", "")).strip()[:220]
        notes = str(parsed.get("notes", "")).strip()[:220]
        if not action_title or not why or not notes:
            return None
        confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
        return OpenAIPoolActionSuggestion(
            action_title=action_title,
            why=why,
            notes=notes,
            confidence=confidence,
        )
