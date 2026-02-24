from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from pool_cli.cache import CacheDB, CachedActionSuggestion, CachedAssignment
from pool_cli.openai_api import (
    OpenAIPoolAnalyzer,
    is_openai_available,
)
from pool_cli.pools import OTHER_POOL, PoolDefinition, predefined_pools
from pool_cli.siglip import classify_with_siglip_image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"}
ProgressCallback = Callable[[str, int, int, dict[str, int] | None], None]
StatusCallback = Callable[[str], None]


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    file_path: Path
    size: int
    mtime: float
    sha1: str
    file_key: str
    canonical_id: str | None = None


@dataclass(slots=True)
class MatchView:
    image_id: str
    file_path: str
    confidence: float
    why: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "confidence": round(self.confidence, 4),
            "why": self.why,
        }


@dataclass(slots=True)
class ActionSuggestion:
    title: str
    why: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "title": self.title,
            "why": self.why,
            "confidence": round(self.confidence, 4),
        }


@dataclass(slots=True)
class PoolSuggestion:
    pool_name: str
    pool_type: str
    match_count: int
    action: ActionSuggestion | None
    notes: str
    top_matches: list[MatchView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pool_name": self.pool_name,
            "pool_type": self.pool_type,
            "match_count": self.match_count,
            "action": self.action.to_dict() if self.action else None,
            "notes": self.notes,
            "top_matches": [match.to_dict() for match in self.top_matches],
        }


@dataclass(slots=True)
class RunReport:
    run_id: str
    input_path: str
    pipeline: str
    model: str
    image_count: int
    dedup_groups: int
    stage_timings: dict[str, float]
    openai_usage: dict[str, int]
    pools: list[PoolSuggestion]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_path": self.input_path,
            "pipeline": self.pipeline,
            "model": self.model,
            "image_count": self.image_count,
            "dedup_groups": self.dedup_groups,
            "stage_timings": {k: round(float(v), 4) for k, v in self.stage_timings.items()},
            "openai_usage": {k: int(v) for k, v in self.openai_usage.items()},
            "pools": [pool.to_dict() for pool in self.pools],
        }


@dataclass(slots=True)
class PipelineResult:
    report: RunReport
    predictions_by_path: dict[str, str]


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    cache_dir: Path
    openai_model: str
    openai_timeout_seconds: int
    max_images: int | None
    batch_size: int
    seed: int
    siglip2_max_patches: int
    image_min_confidence: float = 0.0
    image_min_margin: float = 0.0
    user_pools_enable: bool = True
    user_pools_max: int = 4
    user_pool_min_size: int = 12
    user_pool_min_score: float = 0.62
    user_pool_min_margin: float = 0.06
    user_pool_min_cohesion: float = 0.72
    user_pool_cluster_similarity: float = 0.80
    user_pool_prefilter_similarity: float = 0.72
    user_pool_seed_min_similarity: float = 0.60
    user_pool_sample_size: int = 12


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_key(path: Path, size: int, mtime: float, sha1: str) -> str:
    payload = f"{path.resolve()}|{size}|{int(mtime)}|{sha1}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _iter_image_paths(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def discover_images(input_path: Path, max_images: int | None) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in _iter_image_paths(input_path):
        stat = path.stat()
        digest = _hash_file(path)
        image_id = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        records.append(
            ImageRecord(
                image_id=image_id,
                file_path=path,
                size=int(stat.st_size),
                mtime=float(stat.st_mtime),
                sha1=digest,
                file_key=_file_key(path, int(stat.st_size), float(stat.st_mtime), digest),
            )
        )
        if max_images is not None and len(records) >= max_images:
            break
    return records


def apply_exact_dedup(records: list[ImageRecord]) -> tuple[list[ImageRecord], int]:
    canonical_by_sha: dict[str, str] = {}
    canonical_records: list[ImageRecord] = []
    for record in records:
        canonical_id = canonical_by_sha.setdefault(record.sha1, record.image_id)
        record.canonical_id = canonical_id
        if canonical_id == record.image_id:
            canonical_records.append(record)
    return canonical_records, len(canonical_records)


def _build_pool_notes(matches: list[MatchView]) -> str:
    if not matches:
        return "No strong matches."
    avg_confidence = sum(match.confidence for match in matches) / len(matches)
    return f"Consistent visual pattern (avg confidence {avg_confidence:.2f})."


def _action_cache_signature(matches: list[MatchView]) -> str:
    digest = hashlib.sha1()
    for match in matches:
        digest.update(match.image_id.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _suggest_action_with_openai(
    *,
    pool: PoolDefinition,
    matches: list[MatchView],
    match_count: int,
    openai_analyzer: OpenAIPoolAnalyzer | None,
    cache: CacheDB | None,
) -> tuple[ActionSuggestion, str] | None:
    if len(matches) < 2:
        return None

    signature = _action_cache_signature(matches)

    if cache is not None:
        cached = cache.get_action_suggestion(pool.name, signature)
        if cached is not None:
            action = ActionSuggestion(
                title=cached.action_title,
                why=cached.why,
                confidence=max(0.35, min(0.99, cached.confidence)),
            )
            return action, cached.notes

    if openai_analyzer is None:
        return None

    sample_paths: list[Path] = []
    for match in matches[:6]:
        path = Path(match.file_path)
        if path.exists():
            sample_paths.append(path)

    suggested = openai_analyzer.suggest_pool_action(
        pool=pool,
        match_count=match_count,
        sample_paths=sample_paths,
    )
    if suggested is None:
        return None

    if cache is not None:
        cache.put_action_suggestion(
            pool_name=pool.name,
            signature=signature,
            suggestion=CachedActionSuggestion(
                action_title=suggested.action_title,
                why=suggested.why,
                notes=suggested.notes,
                confidence=suggested.confidence,
            ),
        )

    action = ActionSuggestion(
        title=suggested.action_title,
        why=suggested.why,
        confidence=max(0.35, min(0.99, suggested.confidence)),
    )
    return action, suggested.notes


def _build_report(
    config: PipelineConfig,
    records: list[ImageRecord],
    dedup_groups: int,
    assignments: dict[str, CachedAssignment],
    pools: list[PoolDefinition],
    openai_analyzer: OpenAIPoolAnalyzer | None = None,
    cache: CacheDB | None = None,
) -> RunReport:
    pool_map = {pool.name: pool for pool in pools}
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        assigned = assignments.get(record.image_id)
        pool_name = assigned.pool_name if assigned else OTHER_POOL
        grouped[pool_name].append(record)

    _default = CachedAssignment(OTHER_POOL, 0.0, "")
    pool_names = [pool.name for pool in pools] + [OTHER_POOL]
    suggestions: list[PoolSuggestion] = []
    for pool_name in pool_names:
        items = grouped.get(pool_name, [])
        items_sorted = sorted(
            items,
            key=lambda rec: assignments.get(rec.image_id, _default).confidence,
            reverse=True,
        )
        matches: list[MatchView] = []
        for record in items_sorted[:10]:
            assignment = assignments.get(record.image_id, _default)
            matches.append(
                MatchView(
                    image_id=record.image_id,
                    file_path=str(record.file_path),
                    confidence=assignment.confidence,
                    why=assignment.why,
                )
            )

        action = None
        notes = _build_pool_notes(matches)
        if pool_name != OTHER_POOL and pool_name in pool_map:
            suggested = _suggest_action_with_openai(
                pool=pool_map[pool_name],
                matches=matches,
                match_count=len(items),
                openai_analyzer=openai_analyzer,
                cache=cache,
            )
            if suggested is not None:
                action, notes = suggested

        suggestions.append(
            PoolSuggestion(
                pool_name=pool_name,
                pool_type=pool_map[pool_name].pool_type if pool_name in pool_map else "system",
                match_count=len(items),
                action=action,
                notes=notes,
                top_matches=matches,
            )
        )

    suggestions.sort(key=lambda item: (item.match_count, item.pool_name), reverse=True)
    return RunReport(
        run_id=hashlib.sha1(f"{time.time()}:{config.input_path}".encode("utf-8")).hexdigest()[:12],
        input_path=str(config.input_path),
        pipeline="siglip2-image",
        model=config.openai_model,
        image_count=len(records),
        dedup_groups=dedup_groups,
        stage_timings={},
        openai_usage={},
        pools=suggestions,
    )


def run_pipeline(
    config: PipelineConfig,
    log: Callable[[str], None],
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> PipelineResult:
    np.random.seed(config.seed)
    pools = predefined_pools()
    stage_timings: dict[str, float] = {}

    with CacheDB(config.cache_dir / "cache.sqlite3") as cache:
        start = time.perf_counter()
        if status is not None:
            status("Discovering images…")
        records = discover_images(input_path=config.input_path, max_images=config.max_images)
        if not records:
            raise RuntimeError("No supported images found in input folder.")
        canonical_records, dedup_groups = apply_exact_dedup(records)
        stage_timings["discovery"] = time.perf_counter() - start
        log(f"Discovered images: {len(records)} (canonical: {len(canonical_records)})")

        openai_analyzer: OpenAIPoolAnalyzer | None = None
        if is_openai_available():
            openai_analyzer = OpenAIPoolAnalyzer(
                model=config.openai_model,
                timeout_seconds=config.openai_timeout_seconds,
                pools=pools,
                log=log,
            )
            log("OpenAI analyzer: enabled")
        else:
            log("OpenAI analyzer: disabled (no OPENAI_API_KEY)")

        canonical_assignments, siglip_timings, runtime_pools = classify_with_siglip_image(
            config=config,
            cache=cache,
            canonical_records=canonical_records,
            pools=pools,
            log=log,
            progress=progress,
            openai_analyzer=openai_analyzer,
            status=status,
        )
        pools = runtime_pools
        stage_timings.update(siglip_timings)

        start = time.perf_counter()
        if status is not None:
            status("Generating suggestions…")
        canonical_by_id = {r.image_id: r for r in canonical_records}
        assignments: dict[str, CachedAssignment] = {}
        for record in records:
            canonical_id = record.canonical_id or record.image_id
            canonical_record = canonical_by_id.get(canonical_id, record)
            assignments[record.image_id] = canonical_assignments.get(
                canonical_record.image_id,
                CachedAssignment(OTHER_POOL, 0.30, "No canonical assignment"),
            )

        report = _build_report(
            config=config,
            records=records,
            dedup_groups=dedup_groups,
            assignments=assignments,
            pools=pools,
            openai_analyzer=openai_analyzer,
            cache=cache,
        )
        openai_usage = openai_analyzer.stats.to_dict() if openai_analyzer is not None else {}
        report.openai_usage = openai_usage
        log(
            f"OpenAI usage tokens: "
            f"in={openai_usage.get('input_tokens', 0)}, "
            f"cached={openai_usage.get('cached_input_tokens', 0)}, "
            f"out={openai_usage.get('output_tokens', 0)}"
        )
        stage_timings["suggesting"] = time.perf_counter() - start

        report.stage_timings = {
            **stage_timings,
            "analyzing": sum(
                stage_timings.get(k, 0.0)
                for k in ("discovery", "embedding", "classification")
            ),
            "suggesting": stage_timings.get("suggesting", 0.0),
        }

        predictions_by_path = {
            str(record.file_path.resolve()): assignments[record.image_id].pool_name
            for record in records
            if record.image_id in assignments
        }
        return PipelineResult(report=report, predictions_by_path=predictions_by_path)


def write_json_report(report: RunReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    minutes, left = divmod(total, 60)
    return f"{minutes}m{left:02d}s"
