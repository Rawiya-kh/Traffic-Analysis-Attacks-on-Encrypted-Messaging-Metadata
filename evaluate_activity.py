#!/usr/bin/env python3
"""Evaluate activity-classification baselines on fixed R1 session splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_COLUMNS = (
    "bytes_sent",
    "bytes_received",
    "packets_sent",
    "packets_received",
    "mean_iat_seconds",
    "std_iat_seconds",
    "duration_seconds",
)
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEEDS = (11, 23, 42, 57, 101)


def _read_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        expected = [*FEATURE_COLUMNS, "label"]
        if reader.fieldnames != expected:
            raise ValueError(
                f"{path}: expected columns {expected}, got {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: split is empty")
    try:
        features = np.asarray(
            [[float(row[name]) for name in FEATURE_COLUMNS] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: R1 feature value is not numeric") from error
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError(f"{path}: features contain a negative or non-finite value")
    if any(not label for label in labels):
        raise ValueError(f"{path}: empty activity label")
    return features, labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_models(seed: int, xgboost_class: Any, class_count: int) -> dict[str, Any]:
    return {
        "B0-majority": None,
        "B2-random-forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "B2-bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                max_features="sqrt",
                random_state=seed,
            ),
            n_estimators=200,
            max_samples=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
        "B3-xgboost": xgboost_class(
            objective="multi:softprob",
            num_class=class_count,
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _evaluate(
    model_name: str,
    seed: int,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )
    )
    run_metrics = {
        "model": model_name,
        "seed": seed,
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    per_class = [
        {
            "model": model_name,
            "seed": seed,
            "split": split_name,
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(classes)
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    confusion_rows = [
        {
            "model": model_name,
            "seed": seed,
            "split": split_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[true_index, predicted_index]),
        }
        for true_index, true_label in enumerate(classes)
        for predicted_index, predicted_label in enumerate(classes)
    ]
    return run_metrics, per_class, confusion_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["split"]), []).append(row)

    results = []
    for (model, split), runs in sorted(groups.items()):
        result: dict[str, Any] = {
            "model": model,
            "split": split,
            "seed_count": len(runs),
            "seeds": ";".join(str(row["seed"]) for row in runs),
        }
        for metric in metrics:
            values = [row[metric] for row in runs]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        results.append(result)
    return results


def run_evaluation(
    splits_dir: Path, output_dir: Path, seeds: tuple[int, ...]
) -> None:
    split_data = {
        name: _read_split(splits_dir / f"{name}.csv") for name in SPLIT_NAMES
    }
    classes = sorted(set(split_data["train"][1]))
    if len(classes) != 8:
        raise ValueError(
            f"Expected eight activity classes for multiclass XGBoost, found {len(classes)}"
        )
    for split_name in ("val", "test"):
        observed = set(split_data[split_name][1])
        if observed != set(classes):
            raise ValueError(
                f"{split_name} labels differ from training labels: "
                f"missing={sorted(set(classes) - observed)}, "
                f"unexpected={sorted(observed - set(classes))}"
            )

    try:
        from xgboost import XGBClassifier
        import xgboost
    except ImportError as error:
        raise RuntimeError(
            "B3 requires XGBoost. Install dependencies with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from error

    x_train, y_train = split_data["train"]
    x_val, y_val = split_data["val"]
    class_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    class_to_index = {label: index for index, label in enumerate(classes)}
    y_train_indexed = np.asarray([class_to_index[label] for label in y_train])
    y_val_indexed = np.asarray([class_to_index[label] for label in y_val])
    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for seed in seeds:
        models = _build_models(seed, XGBClassifier, len(classes))
        majority_label = Counter(y_train).most_common(1)[0][0]
        for model_name, model in models.items():
            if model is None:
                val_prediction = np.asarray([majority_label] * len(y_val), dtype=str)
                test_prediction = np.asarray(
                    [majority_label] * len(split_data["test"][1]), dtype=str
                )
            elif model_name == "B3-xgboost":
                model.fit(
                    x_train,
                    y_train_indexed,
                    sample_weight=class_weights,
                    eval_set=[(x_val, y_val_indexed)],
                    verbose=False,
                )
                val_prediction = np.asarray(classes)[model.predict(x_val).astype(int)]
                test_prediction = np.asarray(classes)[
                    model.predict(split_data["test"][0]).astype(int)
                ]
            else:
                model.fit(x_train, y_train)
                val_prediction = model.predict(x_val).astype(str)
                test_prediction = model.predict(split_data["test"][0]).astype(str)

            for split_name, y_true, y_pred in (
                ("val", y_val, val_prediction),
                ("test", split_data["test"][1], test_prediction),
            ):
                run, per_class, matrix_rows = _evaluate(
                    model_name, seed, split_name, y_true, y_pred, classes
                )
                run_rows.append(run)
                class_rows.extend(per_class)
                confusion_rows.extend(matrix_rows)
        print(f"Completed seed {seed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "activity_metrics_by_seed.csv",
        run_rows,
        ["model", "seed", "split", "accuracy", "macro_precision", "macro_recall", "macro_f1"],
    )
    summary_rows = _aggregate_runs(run_rows)
    summary_fields = [
        "model",
        "split",
        "seed_count",
        "seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]
    _write_csv(output_dir / "activity_summary.csv", summary_rows, summary_fields)
    _write_csv(
        output_dir / "activity_per_class.csv",
        class_rows,
        ["model", "seed", "split", "label", "precision", "recall", "f1", "support"],
    )
    _write_csv(
        output_dir / "activity_confusion_matrices.csv",
        confusion_rows,
        [
            "model",
            "seed",
            "split",
            "true_label",
            "predicted_label",
            "count",
        ],
    )

    metadata = {
        "task": "activity classification",
        "correlation_metrics": "deferred: no session peer/pair ground-truth labels",
        "split_policy": "Existing train/val/test files reused unchanged for every model and seed",
        "split_files": {
            f"{name}.csv": {
                "sha256": _sha256(splits_dir / f"{name}.csv"),
                "sessions": len(split_data[name][1]),
            }
            for name in SPLIT_NAMES
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": classes,
        "seeds": list(seeds),
        "models": {
            "B0-majority": "Most frequent training label",
            "B2-random-forest": {
                "n_estimators": 400,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
            "B2-bagging": {
                "n_estimators": 200,
                "base_estimator": "balanced decision tree, max_features=sqrt",
                "max_samples": 0.8,
            },
            "B3-xgboost": {
                "n_estimators": 600,
                "max_depth": 5,
                "learning_rate": 0.05,
                "early_stopping_rounds": 30,
                "validation_split": "val",
                "training_class_weights": "balanced",
            },
        },
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (output_dir / "activity_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for row in summary_rows:
        if row["split"] == "test":
            print(
                f'{row["model"]}: test macro-F1 '
                f'{row["macro_f1_mean"]:.4f} +/- {row["macro_f1_std"]:.4f}, '
                f'macro-precision {row["macro_precision_mean"]:.4f}, '
                f'macro-recall {row["macro_recall_mean"]:.4f}'
            )
    print(f"Structured activity results written to {output_dir}")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate B0, B2, and B3 activity baselines on fixed CSV splits."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=project_dir / "processed_r1",
        help="Folder containing the existing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "activity",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Use 3-5 distinct seeds; splits are never regenerated.",
    )
    args = parser.parse_args()
    if not 3 <= len(args.seeds) <= 5:
        parser.error("provide 3-5 seeds")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct non-negative integers")
    run_evaluation(args.splits_dir, args.output_dir, tuple(args.seeds))


if __name__ == "__main__":
    main()
