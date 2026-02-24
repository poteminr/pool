from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from pool_cli.cache import CacheDB, CachedAssignment
from pool_cli.openai_api import OpenAIPoolAnalyzer
from pool_cli.pools import OTHER_POOL, PoolDefinition, pool_prompt
from pool_cli.user_pools import discover_user_pools_from_other

from .ops import build_siglip2_backend, normalize_rows

if TYPE_CHECKING:
    from .ops import SigLIP2EmbeddingBackend


ProgressCallback = Callable[[str, int, int, dict[str, int] | None], None]


def _read_cached_embeddings(
    records: list[Any],
    cache: CacheDB,
) -> tuple[dict[str, list[float]], list[Any]]:
    results: dict[str, list[float]] = {}
    missing: list[Any] = []
    for record in records:
        cached = cache.get_embedding(record.file_key)
        if cached is not None:
            results[record.image_id] = cached
        else:
            missing.append(record)
    return results, missing


def _compute_missing_embeddings(
    missing: list[Any],
    results: dict[str, list[float]],
    cache: CacheDB,
    backend: SigLIP2EmbeddingBackend,
    batch_size: int,
    total: int,
    progress: ProgressCallback | None = None,
) -> dict[str, list[float]]:
    completed = total - len(missing)
    step = max(1, int(batch_size))
    for start in range(0, len(missing), step):
        chunk = missing[start : start + step]
        paths = [record.file_path for record in chunk]
        vectors = backend.embed(paths=paths, batch_size=step)
        for record, vector in zip(chunk, vectors, strict=False):
            results[record.image_id] = vector
            cache.put_embedding(record.file_key, vector)
        completed += len(chunk)
        if progress is not None and total > 0:
            progress("embedding", completed, total, None)
    return results


def _assignment_from_scores(
    scores: list[float] | np.ndarray | None,
    pool_names: list[str],
    min_confidence: float,
    min_margin: float,
) -> CachedAssignment:
    if scores is None or len(scores) != len(pool_names):
        return CachedAssignment(
            pool_name=OTHER_POOL,
            confidence=0.35,
            why="SigLIP2 image score unavailable",
        )

    clipped = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    if clipped.size == 0:
        return CachedAssignment(
            pool_name=OTHER_POOL,
            confidence=0.35,
            why="SigLIP2 image score unavailable",
        )

    ranked_idx = np.argsort(clipped)[::-1]
    top_idx = int(ranked_idx[0])
    second_idx = int(ranked_idx[1]) if ranked_idx.size > 1 else top_idx
    top_pool = pool_names[top_idx]
    top_score = float(clipped[top_idx])
    second_score = float(clipped[second_idx])
    margin = max(0.0, top_score - second_score)
    confidence = max(0.30, min(0.99, 0.55 * top_score + 0.45 * margin + 0.25))

    if top_score < min_confidence or margin < min_margin:
        return CachedAssignment(
            pool_name=OTHER_POOL,
            confidence=min(confidence, 0.60),
            why=f"Per-image SigLIP2 low confidence ({top_pool}:{top_score:.2f}, margin:{margin:.2f})",
        )

    return CachedAssignment(
        pool_name=top_pool,
        confidence=confidence,
        why=f"Per-image SigLIP2 prompt match ({top_pool}:{top_score:.2f}, margin:{margin:.2f})",
    )


StatusCallback = Callable[[str], None]


def classify_with_siglip_image(
    config: Any,
    cache: CacheDB,
    canonical_records: list[Any],
    pools: list[PoolDefinition],
    log: Callable[[str], None],
    progress: ProgressCallback | None = None,
    openai_analyzer: OpenAIPoolAnalyzer | None = None,
    status: StatusCallback | None = None,
) -> tuple[dict[str, CachedAssignment], dict[str, float], list[PoolDefinition]]:
    stage_timings: dict[str, float] = {}
    start = time.perf_counter()

    # Phase 1: check caches (no model needed)
    canonical_embeddings, embedding_missing = _read_cached_embeddings(
        records=canonical_records,
        cache=cache,
    )

    ordered_records = [record for record in canonical_records if record.image_id in canonical_embeddings]
    canonical_assignments: dict[str, CachedAssignment] = {}
    assignment_missing: list[Any] = []
    for record in ordered_records:
        cached = cache.get_assignment(record.file_key)
        if cached is None:
            assignment_missing.append(record)
        else:
            canonical_assignments[record.image_id] = cached

    # Phase 2: load model only if needed
    needs_model = bool(embedding_missing) or bool(assignment_missing)
    embed_backend: SigLIP2EmbeddingBackend | None = None

    if needs_model:
        if status is not None:
            status("Loading vision model…")
        embed_backend = build_siglip2_backend(siglip2_max_patches=config.siglip2_max_patches, log=log)

        # Phase 3: compute missing embeddings
        if embedding_missing:
            if progress is not None:
                progress("embedding", 0, len(canonical_records), None)
            canonical_embeddings = _compute_missing_embeddings(
                missing=embedding_missing,
                results=canonical_embeddings,
                cache=cache,
                backend=embed_backend,
                batch_size=config.batch_size,
                total=len(canonical_records),
                progress=progress,
            )
            ordered_records = [record for record in canonical_records if record.image_id in canonical_embeddings]
            for record in ordered_records:
                if record.image_id not in canonical_assignments:
                    cached = cache.get_assignment(record.file_key)
                    if cached is None:
                        if record not in assignment_missing:
                            assignment_missing.append(record)
                    else:
                        canonical_assignments[record.image_id] = cached

    stage_timings["embedding"] = time.perf_counter() - start

    if not ordered_records:
        raise RuntimeError("No embeddings produced; cannot continue.")

    # Phase 4: classify missing assignments
    start = time.perf_counter()
    pool_names = [pool.name for pool in pools]
    min_confidence = config.image_min_confidence
    min_margin = config.image_min_margin

    if assignment_missing and embed_backend is not None:
        total_assignments = len(ordered_records)
        completed_assignments = total_assignments - len(assignment_missing)
        if progress is not None:
            progress("classifying", completed_assignments, total_assignments, None)

        prompts = [pool_prompt(pool) for pool in pools]
        label_vectors = embed_backend.embed_text(prompts, batch_size=min(32, config.batch_size))
        label_matrix = normalize_rows(np.asarray(label_vectors, dtype=np.float32))

        image_matrix = np.asarray(
            [canonical_embeddings[record.image_id] for record in assignment_missing],
            dtype=np.float32,
        )
        normalized_images = normalize_rows(image_matrix)
        similarities = normalized_images @ label_matrix.T
        score_rows = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
        for record, row in zip(assignment_missing, score_rows.tolist(), strict=False):
            assignment = _assignment_from_scores(
                scores=row,
                pool_names=pool_names,
                min_confidence=min_confidence,
                min_margin=min_margin,
            )
            canonical_assignments[record.image_id] = assignment
            cache.put_assignment(record.file_key, assignment)
            completed_assignments += 1
            if progress is not None and total_assignments > 0:
                progress("classifying", completed_assignments, total_assignments, None)

    # Phase 5: discover user pools
    canonical_assignments, runtime_pools = discover_user_pools_from_other(
        config=config,
        cache=cache,
        ordered_records=ordered_records,
        canonical_embeddings=canonical_embeddings,
        canonical_assignments=canonical_assignments,
        pools=pools,
        backend=embed_backend,
        openai_analyzer=openai_analyzer,
        log=log,
    )
    stage_timings["classification"] = time.perf_counter() - start
    return canonical_assignments, stage_timings, runtime_pools
