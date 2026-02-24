from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pool_cli.pools import OTHER_POOL, all_pool_names


@dataclass(slots=True)
class EvalResult:
    images_scored: int
    accuracy: float
    macro_f1: float
    macro_f0_5: float
    macro_f0_5_no_other: float
    macro_precision: float
    coverage_non_other: float
    avg_pool_purity_non_other: float
    per_class: dict[str, dict[str, float]]
    cluster_purity: dict[str, dict[str, float | int | str]]
    missing_predictions: int
    unused_predictions: int

    def to_dict(self) -> dict:
        return {
            "images_scored": self.images_scored,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "macro_f0_5": round(self.macro_f0_5, 4),
            "macro_f0_5_no_other": round(self.macro_f0_5_no_other, 4),
            "macro_precision": round(self.macro_precision, 4),
            "coverage_non_other": round(self.coverage_non_other, 4),
            "avg_pool_purity_non_other": round(self.avg_pool_purity_non_other, 4),
            "per_class": {
                key: {
                    "precision": round(value["precision"], 4),
                    "recall": round(value["recall"], 4),
                    "f1": round(value["f1"], 4),
                    "f0_5": round(value["f0_5"], 4),
                    "support": int(value["support"]),
                }
                for key, value in self.per_class.items()
            },
            "cluster_purity": {
                key: {
                    "size": int(value["size"]),
                    "strict_precision": round(float(value["strict_precision"]), 4),
                    "strict_recall": round(float(value["strict_recall"]), 4),
                    "strict_f1": round(float(value["strict_f1"]), 4),
                    "strict_f0_5": round(float(value["strict_f0_5"]), 4),
                    "garbage_rate": round(float(value["garbage_rate"]), 4),
                    "majority_label": str(value["majority_label"]),
                    "majority_purity": round(float(value["majority_purity"]), 4),
                    "correct_count": int(value["correct_count"]),
                    "gt_total": int(value["gt_total"]),
                }
                for key, value in self.cluster_purity.items()
            },
            "missing_predictions": self.missing_predictions,
            "unused_predictions": self.unused_predictions,
        }


def normalize_pool_label(raw: str, strict: bool) -> str | None:
    value = raw.strip()
    if not value:
        return None
    allowed = all_pool_names(include_other=True)
    if value in allowed:
        return value
    allowed_lower = {item.lower(): item for item in allowed}
    lower = value.lower()
    if lower in allowed_lower:
        return allowed_lower[lower]
    if strict:
        raise ValueError(f"Unknown pool label: {raw}")
    return None


def load_eval_labels(labels_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return labels

    header = [column.strip().lower() for column in rows[0]]
    has_header = "file_path" in header and "pool" in header
    if has_header:
        file_idx = header.index("file_path")
        pool_idx = header.index("pool")
        rows_iter = rows[1:]
    else:
        file_idx = 0
        pool_idx = 1
        rows_iter = rows

    for row in rows_iter:
        if len(row) <= max(file_idx, pool_idx):
            continue
        file_raw = row[file_idx].strip()
        pool_raw = row[pool_idx].strip()
        if not file_raw or not pool_raw:
            continue
        path_obj = Path(file_raw).expanduser()
        if not path_obj.is_absolute():
            path_obj = (labels_path.parent / path_obj).resolve()
        pool_name = normalize_pool_label(pool_raw, strict=True)
        if pool_name is None:
            continue
        labels[str(path_obj)] = pool_name
    return labels


def infer_eval_labels_from_folders(
    predictions_by_path: dict[str, str],
    input_root: Path,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    root = input_root.resolve()
    for file_path in predictions_by_path:
        path_obj = Path(file_path).resolve()
        try:
            relative = path_obj.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        folder_name = relative.parts[0]
        normalized = normalize_pool_label(folder_name, strict=False)
        if normalized is None:
            continue
        labels[str(path_obj)] = normalized
    return labels


def compute_eval(
    labels_by_path: dict[str, str],
    predictions_by_path: dict[str, str],
) -> EvalResult:
    common_paths = sorted(set(labels_by_path).intersection(predictions_by_path))
    if not common_paths:
        raise ValueError("No overlapping files between labels and predictions.")

    total = len(common_paths)
    correct = sum(1 for path in common_paths if labels_by_path[path] == predictions_by_path[path])
    classes = sorted(
        set(labels_by_path[path] for path in common_paths)
        | set(predictions_by_path[path] for path in common_paths)
    )

    def _fbeta(precision: float, recall: float, beta: float) -> float:
        if precision <= 0.0 and recall <= 0.0:
            return 0.0
        beta_sq = beta * beta
        denom = (beta_sq * precision) + recall
        if denom <= 0.0:
            return 0.0
        return (1.0 + beta_sq) * precision * recall / denom

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    f0_5_values: list[float] = []
    precision_values: list[float] = []
    f0_5_by_class: dict[str, float] = {}
    for class_name in classes:
        tp = sum(
            1
            for path in common_paths
            if labels_by_path[path] == class_name and predictions_by_path[path] == class_name
        )
        fp = sum(
            1
            for path in common_paths
            if labels_by_path[path] != class_name and predictions_by_path[path] == class_name
        )
        fn = sum(
            1
            for path in common_paths
            if labels_by_path[path] == class_name and predictions_by_path[path] != class_name
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _fbeta(precision, recall, beta=1.0)
        f0_5 = _fbeta(precision, recall, beta=0.5)
        precision_values.append(precision)
        f1_values.append(f1)
        f0_5_values.append(f0_5)
        f0_5_by_class[class_name] = f0_5
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f0_5": f0_5,
            "support": float(sum(1 for path in common_paths if labels_by_path[path] == class_name)),
        }

    cluster_purity: dict[str, dict[str, float | int | str]] = {}
    predicted_pools = sorted(set(predictions_by_path[path] for path in common_paths))
    label_support: dict[str, int] = {}
    for path in common_paths:
        label = labels_by_path[path]
        label_support[label] = label_support.get(label, 0) + 1
    for pool_name in predicted_pools:
        members = [path for path in common_paths if predictions_by_path[path] == pool_name]
        size = len(members)
        if size == 0:
            continue
        label_counts: dict[str, int] = {}
        for path in members:
            label = labels_by_path[path]
            label_counts[label] = label_counts.get(label, 0) + 1
        majority_label, majority_count = max(
            label_counts.items(),
            key=lambda item: (item[1], int(item[0] == pool_name), item[0]),
        )
        strict_tp = label_counts.get(pool_name, 0)
        strict_precision = strict_tp / size
        support = label_support.get(pool_name, 0)
        strict_recall = strict_tp / support if support else 0.0
        strict_f1 = _fbeta(strict_precision, strict_recall, beta=1.0)
        strict_f0_5 = _fbeta(strict_precision, strict_recall, beta=0.5)
        majority_purity = majority_count / size
        cluster_purity[pool_name] = {
            "size": size,
            "strict_precision": strict_precision,
            "strict_recall": strict_recall,
            "strict_f1": strict_f1,
            "strict_f0_5": strict_f0_5,
            "garbage_rate": 1.0 - strict_precision,
            "majority_label": majority_label,
            "majority_purity": majority_purity,
            "correct_count": strict_tp,
            "gt_total": support,
        }

    non_other_f0_5_values = [
        value for class_name, value in f0_5_by_class.items() if class_name != OTHER_POOL
    ]
    macro_f0_5_no_other = (
        sum(non_other_f0_5_values) / len(non_other_f0_5_values)
        if non_other_f0_5_values
        else 0.0
    )
    non_other_predictions = sum(
        1 for path in common_paths if predictions_by_path[path] != OTHER_POOL
    )
    coverage_non_other = non_other_predictions / total if total else 0.0
    non_other_pool_purity_values = [
        float(stats["strict_precision"])
        for pool_name, stats in cluster_purity.items()
        if pool_name != OTHER_POOL
    ]
    avg_pool_purity_non_other = (
        sum(non_other_pool_purity_values) / len(non_other_pool_purity_values)
        if non_other_pool_purity_values
        else 0.0
    )

    return EvalResult(
        images_scored=total,
        accuracy=(correct / total) if total else 0.0,
        macro_f1=(sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        macro_f0_5=(sum(f0_5_values) / len(f0_5_values)) if f0_5_values else 0.0,
        macro_f0_5_no_other=macro_f0_5_no_other,
        macro_precision=(sum(precision_values) / len(precision_values)) if precision_values else 0.0,
        coverage_non_other=coverage_non_other,
        avg_pool_purity_non_other=avg_pool_purity_non_other,
        per_class=per_class,
        cluster_purity=cluster_purity,
        missing_predictions=len(set(labels_by_path) - set(predictions_by_path)),
        unused_predictions=len(set(predictions_by_path) - set(labels_by_path)),
    )



__all__ = [
    "EvalResult",
    "normalize_pool_label",
    "load_eval_labels",
    "infer_eval_labels_from_folders",
    "compute_eval",
]
