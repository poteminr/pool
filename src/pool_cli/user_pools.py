from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from pool_cli.cache import CacheDB, CachedAssignment, CachedUserPoolMetadata
from pool_cli.openai_api import OpenAIPoolAnalyzer
from pool_cli.pools import OTHER_POOL, PoolDefinition, pool_prompt
from pool_cli.siglip.ops import normalize_rows

if TYPE_CHECKING:
    from pool_cli.siglip.ops import SigLIP2EmbeddingBackend


@dataclass(slots=True, frozen=True)
class _CandidateCluster:
    indices: tuple[int, ...]
    cohesion: float

    @property
    def size(self) -> int:
        return len(self.indices)


def _cluster_signature(file_keys: list[str]) -> str:
    digest = hashlib.sha1()
    for value in sorted(file_keys):
        digest.update(value.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _cluster_cohesion(
    normalized_embeddings: np.ndarray,
    indices: tuple[int, ...],
) -> float:
    if len(indices) < 2:
        return 1.0
    subset = normalized_embeddings[np.asarray(indices, dtype=np.int32)]
    similarities = subset @ subset.T
    upper = similarities[np.triu_indices(len(indices), k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.mean(np.clip(upper, -1.0, 1.0)))


def _clusters_from_labels(labels: np.ndarray) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    for label in sorted(int(value) for value in set(labels.tolist()) if int(value) >= 0):
        members = tuple(int(index) for index, value in enumerate(labels.tolist()) if int(value) == label)
        if members:
            output.append(members)
    return output


def _cluster_with_hdbscan(
    normalized_embeddings: np.ndarray,
    min_cluster_size: int,
) -> tuple[list[tuple[int, ...]], str | None]:
    try:
        import hdbscan
    except Exception:
        return [], "missing"

    try:
        min_samples = max(2, min_cluster_size // 3)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(normalized_embeddings)
    except Exception:
        return [], "failed"
    return _clusters_from_labels(np.asarray(labels, dtype=np.int32)), None


def _cluster_with_similarity_graph(
    normalized_embeddings: np.ndarray,
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[tuple[int, ...]]:
    count = int(normalized_embeddings.shape[0])
    if count < min_cluster_size:
        return []

    similarities = normalized_embeddings @ normalized_embeddings.T
    visited = np.zeros(count, dtype=bool)
    clusters: list[tuple[int, ...]] = []
    for seed in range(count):
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            neighbors = np.where(similarities[current] >= similarity_threshold)[0].tolist()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) >= min_cluster_size:
            clusters.append(tuple(sorted(component)))
    return clusters


def _discover_clusters(
    normalized_embeddings: np.ndarray,
    min_cluster_size: int,
    min_cohesion: float,
    similarity_threshold: float,
    log: Callable[[str], None],
) -> list[_CandidateCluster]:
    hdbscan_clusters, hdbscan_state = _cluster_with_hdbscan(
        normalized_embeddings=normalized_embeddings,
        min_cluster_size=min_cluster_size,
    )
    if hdbscan_clusters:
        log(f"User-pool discovery: HDBSCAN produced {len(hdbscan_clusters)} candidate clusters")
        raw_clusters = hdbscan_clusters
    else:
        if hdbscan_state == "missing":
            log("User-pool discovery: hdbscan not installed, using similarity-graph fallback")
        elif hdbscan_state == "failed":
            log("User-pool discovery: HDBSCAN failed, using similarity-graph fallback")
        raw_clusters = _cluster_with_similarity_graph(
            normalized_embeddings=normalized_embeddings,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
        )
        log(f"User-pool discovery: fallback produced {len(raw_clusters)} candidate clusters")

    result: list[_CandidateCluster] = []
    for indices in raw_clusters:
        cohesion = _cluster_cohesion(normalized_embeddings, indices)
        if cohesion < min_cohesion:
            continue
        result.append(_CandidateCluster(indices=indices, cohesion=cohesion))
    result.sort(key=lambda item: (item.size, item.cohesion), reverse=True)
    return result


def _cluster_centroid(
    normalized_embeddings: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    subset = normalized_embeddings[np.asarray(indices, dtype=np.int32)]
    centroid = np.mean(subset, axis=0, keepdims=True)
    return normalize_rows(centroid)[0]


def _select_cluster_samples(
    normalized_embeddings: np.ndarray,
    indices: tuple[int, ...],
    sample_size: int,
) -> tuple[list[int], np.ndarray]:
    centroid = _cluster_centroid(normalized_embeddings=normalized_embeddings, indices=indices)
    subset = normalized_embeddings[np.asarray(indices, dtype=np.int32)]
    similarities = subset @ centroid
    order = np.argsort(similarities)[::-1].tolist()
    target = max(1, min(len(order), sample_size))
    front = max(1, int(round(target * 0.7)))
    back = max(0, target - front)

    selected_local: list[int] = []
    for value in order[:front]:
        if value not in selected_local:
            selected_local.append(value)
    if back > 0:
        for value in order[-back:]:
            if value not in selected_local:
                selected_local.append(value)
    selected_local = selected_local[:target]
    selected_indices = [indices[value] for value in selected_local]
    return selected_indices, centroid


def _score_prompt_matches(
    records: list[Any],
    prompts: list[str],
    backend: SigLIP2EmbeddingBackend,
    embeddings_by_image_id: dict[str, list[float]],
    batch_size: int,
) -> list[list[float]]:
    if not records or not prompts:
        return []

    label_vectors = backend.embed_text(prompts, batch_size=min(32, batch_size))
    label_matrix = normalize_rows(np.asarray(label_vectors, dtype=np.float32))
    image_matrix = np.asarray(
        [embeddings_by_image_id[record.image_id] for record in records],
        dtype=np.float32,
    )
    normalized_images = normalize_rows(image_matrix)
    similarities = normalized_images @ label_matrix.T
    return np.clip((similarities + 1.0) / 2.0, 0.0, 1.0).tolist()


def _sanitize_metadata(raw: CachedUserPoolMetadata) -> CachedUserPoolMetadata | None:
    name = " ".join(raw.name.split()).strip()
    description = " ".join(raw.description.split()).strip()
    action_title = " ".join(raw.action_title.split()).strip()
    why = " ".join(raw.why.split()).strip()
    if not name or not description or not action_title or not why:
        return None
    if name.casefold() in {"other", "misc", "random"}:
        return None
    return CachedUserPoolMetadata(
        name=name[:64],
        description=description[:220],
        action_title=action_title[:80],
        why=why[:220],
    )


def _build_cluster_summary(cluster_size: int, cohesion: float) -> str:
    return (
        "These screenshots were assigned to Other by predefined pools. "
        f"They form one cohesive theme (size={cluster_size}, cohesion={cohesion:.2f}). "
        "Suggest one user-specific pool."
    )


def _assign_with_siglip(
    *,
    candidate_records: list[Any],
    assignments: dict[str, CachedAssignment],
    discovered_pools: list[PoolDefinition],
    centroids: list[np.ndarray],
    backend: SigLIP2EmbeddingBackend,
    canonical_embeddings: dict[str, list[float]],
    batch_size: int,
    min_score: float,
    min_margin: float,
    prefilter_similarity: float,
) -> tuple[dict[str, CachedAssignment], int]:
    pending = [
        record
        for record in candidate_records
        if assignments.get(record.image_id, CachedAssignment(OTHER_POOL, 0.0, "")).pool_name == OTHER_POOL
    ]
    if not pending:
        return dict(assignments), 0

    prompts = [pool_prompt(pool) for pool in discovered_pools]
    rows = _score_prompt_matches(
        records=pending,
        prompts=prompts,
        backend=backend,
        embeddings_by_image_id=canonical_embeddings,
        batch_size=batch_size,
    )

    record_vectors = normalize_rows(
        np.asarray(
            [canonical_embeddings[record.image_id] for record in pending],
            dtype=np.float32,
        )
    )
    centroid_matrix = np.asarray(centroids, dtype=np.float32)
    centroid_similarities = np.clip((record_vectors @ centroid_matrix.T + 1.0) / 2.0, 0.0, 1.0)

    updated = dict(assignments)
    reassigned = 0
    pool_names = [pool.name for pool in discovered_pools]
    for row_index, record in enumerate(pending):
        if row_index >= len(rows):
            break
        row = rows[row_index]
        if len(row) != len(pool_names):
            continue
        scores = np.clip(np.asarray(row, dtype=np.float32), 0.0, 1.0)
        scores = np.where(
            centroid_similarities[row_index] >= prefilter_similarity,
            scores,
            0.0,
        )
        if scores.size == 0 or float(np.max(scores)) <= 0.0:
            continue

        ranked = np.argsort(scores)[::-1]
        top = int(ranked[0])
        second = int(ranked[1]) if ranked.size > 1 else top
        top_score = float(scores[top])
        second_score = float(scores[second])
        margin = max(0.0, top_score - second_score)
        if top_score < min_score or margin < min_margin:
            continue

        confidence = max(0.35, min(0.99, 0.55 * top_score + 0.45 * margin + 0.25))
        pool_name = pool_names[top]
        updated[record.image_id] = CachedAssignment(
            pool_name=pool_name,
            confidence=confidence,
            why=f"User-specific prompt match ({pool_name}:{top_score:.2f}, margin:{margin:.2f})",
        )
        reassigned += 1
    return updated, reassigned


def _seed_from_source_clusters(
    *,
    assignments: dict[str, CachedAssignment],
    discovered_pools: list[PoolDefinition],
    source_cluster_indices: list[tuple[int, ...]],
    other_records: list[Any],
    normalized_other: np.ndarray,
    centroids: list[np.ndarray],
    min_similarity: float,
) -> tuple[dict[str, CachedAssignment], int, int]:
    if not discovered_pools or not source_cluster_indices:
        return dict(assignments), 0, 0

    updated = dict(assignments)
    seeded = 0
    filtered_low_similarity = 0
    for pool_index, pool in enumerate(discovered_pools):
        if pool_index >= len(source_cluster_indices) or pool_index >= len(centroids):
            continue
        indices = source_cluster_indices[pool_index]
        centroid = centroids[pool_index]
        for local_index in indices:
            if local_index < 0 or local_index >= len(other_records):
                continue
            record = other_records[local_index]
            current = updated.get(record.image_id, CachedAssignment(OTHER_POOL, 0.0, ""))
            if current.pool_name != OTHER_POOL:
                continue

            similarity = float(np.dot(normalized_other[local_index], centroid))
            similarity01 = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            if similarity01 < min_similarity:
                filtered_low_similarity += 1
                continue
            confidence = max(0.55, min(0.99, 0.45 + 0.50 * similarity01))
            updated[record.image_id] = CachedAssignment(
                pool_name=pool.name,
                confidence=confidence,
                why=f"User-specific seed cluster match ({pool.name}, sim:{similarity01:.2f})",
            )
            seeded += 1
    return updated, seeded, filtered_low_similarity


def discover_user_pools_from_other(
    *,
    config: Any,
    cache: CacheDB,
    ordered_records: list[Any],
    canonical_embeddings: dict[str, list[float]],
    canonical_assignments: dict[str, CachedAssignment],
    pools: list[PoolDefinition],
    backend: SigLIP2EmbeddingBackend | None,
    openai_analyzer: OpenAIPoolAnalyzer | None,
    log: Callable[[str], None],
) -> tuple[dict[str, CachedAssignment], list[PoolDefinition]]:
    if not config.user_pools_enable or config.user_pools_max <= 0:
        return canonical_assignments, list(pools)

    max_pools = config.user_pools_max
    min_cluster_size = config.user_pool_min_size
    min_cluster_cohesion = config.user_pool_min_cohesion
    min_score = config.user_pool_min_score
    min_margin = config.user_pool_min_margin
    similarity_threshold = config.user_pool_cluster_similarity
    prefilter_similarity = config.user_pool_prefilter_similarity
    seed_min_similarity = config.user_pool_seed_min_similarity
    sample_size = config.user_pool_sample_size

    other_records = [
        record
        for record in ordered_records
        if record.image_id in canonical_embeddings
        and canonical_assignments.get(record.image_id, CachedAssignment(OTHER_POOL, 0.0, "")).pool_name
        == OTHER_POOL
    ]
    if len(other_records) < min_cluster_size:
        log(
            "User-pool discovery skipped: "
            f"Other residual too small ({len(other_records)} < {min_cluster_size})"
        )
        return canonical_assignments, list(pools)

    other_matrix = np.asarray(
        [canonical_embeddings[record.image_id] for record in other_records],
        dtype=np.float32,
    )
    normalized_other = normalize_rows(other_matrix)
    candidates = _discover_clusters(
        normalized_embeddings=normalized_other,
        min_cluster_size=min_cluster_size,
        min_cohesion=min_cluster_cohesion,
        similarity_threshold=similarity_threshold,
        log=log,
    )
    if not candidates:
        log("User-pool discovery: no clusters passed size/cohesion thresholds")
        return canonical_assignments, list(pools)

    used_names = {pool.name.casefold() for pool in pools}
    discovered: list[PoolDefinition] = []
    centroids: list[np.ndarray] = []
    source_cluster_indices: list[tuple[int, ...]] = []

    for cluster in candidates:
        if len(discovered) >= max_pools:
            break

        cluster_file_keys = [other_records[index].file_key for index in cluster.indices]
        signature = _cluster_signature(cluster_file_keys)
        metadata = cache.get_user_cluster_metadata(signature=signature)
        centroid = _cluster_centroid(normalized_embeddings=normalized_other, indices=cluster.indices)

        if metadata is None:
            if openai_analyzer is None:
                continue
            sample_indices, centroid = _select_cluster_samples(
                normalized_embeddings=normalized_other,
                indices=cluster.indices,
                sample_size=sample_size,
            )
            sample_paths = [other_records[index].file_path for index in sample_indices]
            suggested = openai_analyzer.suggest_user_pool(
                cluster_summary=_build_cluster_summary(
                    cluster_size=cluster.size,
                    cohesion=cluster.cohesion,
                ),
                sample_paths=sample_paths,
            )
            if suggested is None:
                continue
            metadata = CachedUserPoolMetadata(
                name=suggested.name,
                description=suggested.description,
                action_title=suggested.action_title,
                why=suggested.why,
            )
            metadata = _sanitize_metadata(metadata)
            if metadata is None:
                continue
            cache.put_user_cluster_metadata(signature=signature, metadata=metadata)
        else:
            metadata = _sanitize_metadata(metadata)
            if metadata is None:
                continue

        if metadata.name.casefold() in used_names:
            continue

        discovered.append(
            PoolDefinition(
                name=metadata.name,
                pool_type="user",
                description=metadata.description,
            )
        )
        centroids.append(centroid)
        source_cluster_indices.append(cluster.indices)
        used_names.add(metadata.name.casefold())

    if not discovered:
        log("User-pool discovery: no valid metadata candidates")
        return canonical_assignments, list(pools)

    updated_assignments = dict(canonical_assignments)
    updated_assignments, seeded_from_clusters, seed_filtered = _seed_from_source_clusters(
        assignments=updated_assignments,
        discovered_pools=discovered,
        source_cluster_indices=source_cluster_indices,
        other_records=other_records,
        normalized_other=normalized_other,
        centroids=centroids,
        min_similarity=seed_min_similarity,
    )
    reassigned_siglip = 0
    if backend is not None:
        updated_assignments, reassigned_siglip = _assign_with_siglip(
            candidate_records=other_records,
            assignments=updated_assignments,
            discovered_pools=discovered,
            centroids=centroids,
            backend=backend,
            canonical_embeddings=canonical_embeddings,
            batch_size=config.batch_size,
            min_score=min_score,
            min_margin=min_margin,
            prefilter_similarity=prefilter_similarity,
        )

    reassigned_total = seeded_from_clusters + reassigned_siglip
    log(
        "User-pool discovery: "
        f"created={len(discovered)}, reassigned_from_other={reassigned_total}, "
        f"other_total={len(other_records)}, "
        f"seeded={seeded_from_clusters}, seed_filtered={seed_filtered}, "
        f"seed_min_similarity={seed_min_similarity:.2f}, reassigned_siglip={reassigned_siglip}"
    )
    return updated_assignments, list(pools) + discovered
