#!/usr/bin/env python3
"""Step 5: Correlation baselines (SIMULATED PAIRS)."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from step4_simulated_paired_flows import (
    DELAY_MEAN,
    DIRECTION_SWAP_PAIRS,
    FEATURE_COLUMNS,
    SEED,
    _load_split,
    make_group,
    make_view_B,
)


# Step 5: SIMULATED PAIRS evaluation settings.
SEEDS = (42, 43, 44)
GROUP_SIZES = (10, 50, 100)
PAIR_BUDGET = 5000
MIN_GROUPS = 5
TOP_K = 5
FPR_TARGETS = (0.2, 1e-2, 1e-3)
ROC_FPR_GRID = np.logspace(-5, 0, 200)
BATCH_SIZE = 256
TRIPLET_MARGIN = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 40
EMBEDDING_BATCH_SIZE = 4096

METHODS = ("B0-timing", "B1-rules", "B4-siamese")
ROC_DATA: dict[tuple[int, str], dict[str, Any]] = {}


class SiameseEncoder(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), p=2, dim=1)


def _read_splits(splits_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep the existing lowercase x_train/y_train naming. Validation remains unused.
    x_train, y_train = _load_split(splits_dir / "train.csv")
    x_test, y_test = _load_split(splits_dir / "test.csv")
    if x_train.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("SIMULATED PAIRS feature columns do not match Step 4")
    if len(x_test) < max(GROUP_SIZES):
        raise ValueError("Test split is smaller than the largest requested group")
    return x_train, y_train, x_test, y_test


def _fit_siamese(
    x_train: np.ndarray, seed: int
) -> tuple[SiameseEncoder, StandardScaler, list[float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    noise_rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    x_train_log = np.log1p(x_train)
    x_train_scaled = scaler.fit_transform(x_train_log).astype(np.float32)

    model = SiameseEncoder(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    loss_history: list[float] = []

    for epoch in range(EPOCHS):
        # A fresh SIMULATED PAIRS view is generated from train sessions each epoch.
        x_partner = make_view_B(x_train, noise_rng)
        x_partner_scaled = scaler.transform(np.log1p(x_partner)).astype(np.float32)
        permutation = noise_rng.permutation(len(x_train))
        epoch_loss = 0.0
        examples_seen = 0

        for start in range(0, len(permutation), BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            if len(batch_indices) < 2:
                continue
            anchors = torch.from_numpy(x_train_scaled[batch_indices])
            positives = torch.from_numpy(x_partner_scaled[batch_indices])

            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            similarities = anchor_embeddings @ positive_embeddings.T
            positive_similarity = similarities.diagonal()
            non_matching = ~torch.eye(len(batch_indices), dtype=torch.bool)
            hardest_negative = similarities.masked_fill(
                ~non_matching, -torch.inf
            ).max(dim=1).values
            loss = F.relu(
                TRIPLET_MARGIN + hardest_negative - positive_similarity
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = len(batch_indices)
            epoch_loss += float(loss.detach()) * batch_count
            examples_seen += batch_count

        if examples_seen == 0:
            raise RuntimeError("SIMULATED PAIRS training produced no usable batches")
        loss_history.append(epoch_loss / examples_seen)

    model.eval()
    return model, scaler, loss_history


def _embed(
    model: SiameseEncoder,
    scaler: StandardScaler,
    features: np.ndarray,
) -> np.ndarray:
    transformed = scaler.transform(np.log1p(features)).astype(np.float32)
    batches = []
    with torch.no_grad():
        for start in range(0, len(transformed), EMBEDDING_BATCH_SIZE):
            tensor = torch.from_numpy(
                transformed[start : start + EMBEDDING_BATCH_SIZE]
            )
            batches.append(model(tensor).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _score_timing(group: dict[str, np.ndarray]) -> np.ndarray:
    delta = group["tB"][None, :] - group["tA"][:, None]
    return -np.abs(delta - DELAY_MEAN)


def _relative_difference_matrix(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    denominator = np.maximum(
        np.maximum(np.abs(left), np.abs(right)), np.finfo(np.float64).eps
    )
    return np.abs(left - right) / denominator


def _score_rules(
    group: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    XA, XB = group["XA"], group["XB"]
    time_difference = np.abs(group["tB"][None, :] - group["tA"][:, None])
    duration_difference = _relative_difference_matrix(
        XA[:, 6, None], XB[None, :, 6]
    )
    bytes_a = XA[:, 0] + XA[:, 1]
    bytes_b = XB[:, 0] + XB[:, 1]
    bytes_difference = _relative_difference_matrix(bytes_a[:, None], bytes_b[None, :])

    passed_rules = (
        (time_difference < 1.0).astype(np.float64)
        + (duration_difference < 0.2).astype(np.float64)
        + (bytes_difference < 0.2).astype(np.float64)
    )
    return passed_rules + rng.uniform(0.0, 1e-7, size=passed_rules.shape)


def _score_siamese(
    group: dict[str, np.ndarray],
    model: SiameseEncoder,
    scaler: StandardScaler,
) -> np.ndarray:
    embeddings_a = _embed(model, scaler, group["XA"])
    embeddings_b = _embed(model, scaler, group["XB"])
    return embeddings_a @ embeddings_b.T


def _roc_metrics(
    scores: np.ndarray, y_true: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    metrics = {
        "tpr_fpr_0.2": float(np.interp(0.2, fpr, tpr)),
        "tpr_fpr_0.01": float(np.interp(1e-2, fpr, tpr)),
        "tpr_fpr_0.001": float(np.interp(1e-3, fpr, tpr)),
    }
    return metrics, fpr, tpr


def _evaluate_seed(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model: SiameseEncoder,
    scaler: StandardScaler,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    seed_rows: list[dict[str, Any]] = []
    roc_curves: dict[tuple[int, str], np.ndarray] = {}
    siamese_true_sum = 0.0
    siamese_true_count = 0
    siamese_random_sum = 0.0
    siamese_random_count = 0

    for group_size in GROUP_SIZES:
        group_count = max(MIN_GROUPS, PAIR_BUDGET // group_size)
        score_parts = {method: [] for method in METHODS}
        label_parts = []

        for _ in range(group_count):
            indices = rng.choice(len(x_test), size=group_size, replace=False)
            group = make_group(x_test, y_test, indices, rng)
            true_pair_mask = np.eye(group_size, dtype=bool)
            labels = true_pair_mask.ravel().astype(np.uint8)
            label_parts.append(labels)

            group_scores = {
                "B0-timing": _score_timing(group),
                "B1-rules": _score_rules(group, rng),
                "B4-siamese": _score_siamese(group, model, scaler),
            }
            for method, scores in group_scores.items():
                score_parts[method].append(scores)

            siamese_scores = group_scores["B4-siamese"]
            siamese_true_sum += float(siamese_scores[true_pair_mask].sum())
            siamese_true_count += int(true_pair_mask.sum())
            negative_scores = siamese_scores[~true_pair_mask]
            siamese_random_sum += float(negative_scores.sum())
            siamese_random_count += len(negative_scores)

        all_labels = np.concatenate(label_parts)
        for method in METHODS:
            all_scores = np.concatenate(
                [scores.ravel() for scores in score_parts[method]]
            )
            metrics, fpr, tpr = _roc_metrics(all_scores, all_labels)
            group_top1 = []
            group_top5 = []
            for scores in score_parts[method]:
                group_top1.append(
                    float(np.mean(np.argmax(scores, axis=1) == np.arange(group_size)))
                )
                k = min(TOP_K, group_size)
                candidate_indices = np.argpartition(
                    scores, kth=group_size - k, axis=1
                )[:, -k:]
                group_top5.append(
                    float(
                        np.mean(
                            np.any(
                                candidate_indices
                                == np.arange(group_size)[:, None],
                                axis=1,
                            )
                        )
                    )
                )
            metrics["top1"] = statistics.mean(group_top1)
            metrics["top5"] = statistics.mean(group_top5)
            seed_rows.append(
                {
                    "seed": seed,
                    "N": group_size,
                    "method": method,
                    **metrics,
                }
            )
            roc_curves[(group_size, method)] = np.interp(
                ROC_FPR_GRID, fpr, tpr
            )

    sanity = {
        "true_pair_mean": siamese_true_sum / siamese_true_count,
        "random_pair_mean": siamese_random_sum / siamese_random_count,
    }
    return seed_rows, roc_curves, sanity


def _aggregate(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_names = (
        "top1",
        "top5",
        "tpr_fpr_0.2",
        "tpr_fpr_0.01",
        "tpr_fpr_0.001",
    )
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        groups.setdefault((row["N"], row["method"]), []).append(row)

    table = []
    for (group_size, method), rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "N": group_size,
            "method": method,
        }
        for metric in metric_names:
            values = [row[metric] for row in rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        table.append(summary)
    return table


def _write_results(
    output_dir: Path,
    table: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    roc_curves_by_seed: dict[int, dict[tuple[int, str], np.ndarray]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "table_correlation.csv"
    table_fields = list(table[0])
    with table_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=table_fields)
        writer.writeheader()
        writer.writerows(table)

    seed_path = output_dir / "correlation_metrics_by_seed.csv"
    seed_fields = ["seed", "N", "method", "top1", "top5", *[
        f"tpr_fpr_{target:g}" for target in FPR_TARGETS
    ]]
    with seed_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=seed_fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow(
                {
                    **row,
                    "tpr_fpr_0.2": row["tpr_fpr_0.2"],
                    "tpr_fpr_0.01": row["tpr_fpr_0.01"],
                    "tpr_fpr_0.001": row["tpr_fpr_0.001"],
                }
            )

    ROC_DATA.clear()
    roc_path = output_dir / "roc_correlation.csv"
    with roc_path.open("w", newline="", encoding="utf-8") as destination:
        fields = ["N", "method", "fpr", "tpr_mean", "tpr_std", *[
            f"tpr_seed_{seed}" for seed in SEEDS
        ]]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for group_size in GROUP_SIZES:
            for method in METHODS:
                seed_curves = np.stack(
                    [
                        roc_curves_by_seed[seed][(group_size, method)]
                        for seed in SEEDS
                    ]
                )
                tpr_mean = seed_curves.mean(axis=0)
                tpr_std = seed_curves.std(axis=0, ddof=1)
                ROC_DATA[(group_size, method)] = {
                    "fpr_grid": ROC_FPR_GRID.copy(),
                    "tpr_by_seed": seed_curves,
                    "tpr_mean": tpr_mean,
                    "tpr_std": tpr_std,
                }
                for index, fpr_value in enumerate(ROC_FPR_GRID):
                    writer.writerow(
                        {
                            "N": group_size,
                            "method": method,
                            "fpr": fpr_value,
                            "tpr_mean": tpr_mean[index],
                            "tpr_std": tpr_std[index],
                            **{
                                f"tpr_seed_{seed}": seed_curves[seed_index, index]
                                for seed_index, seed in enumerate(SEEDS)
                            },
                        }
                    )


def run_evaluation(splits_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    x_train, y_train, x_test, y_test = _read_splits(splits_dir)
    del y_train
    if DIRECTION_SWAP_PAIRS:
        print(
            "SIMULATED PAIRS warning: DIRECTION_SWAP_PAIRS is non-empty; "
            "expected empty for application-name labels."
        )
    else:
        print("SIMULATED PAIRS: DIRECTION_SWAP_PAIRS=[] (application labels)")

    all_seed_rows: list[dict[str, Any]] = []
    roc_curves_by_seed: dict[int, dict[tuple[int, str], np.ndarray]] = {}
    for seed in SEEDS:
        model, scaler, loss_history = _fit_siamese(x_train, seed)
        print(
            f"SIMULATED PAIRS: seed={seed} B4 training loss "
            f"epoch 1={loss_history[0]:.6f}, "
            f"epoch 20={loss_history[19]:.6f}, "
            f"epoch 40={loss_history[39]:.6f}"
        )
        seed_rows, seed_curves, sanity = _evaluate_seed(
            x_test, y_test, model, scaler, seed
        )
        if sanity["true_pair_mean"] <= sanity["random_pair_mean"]:
            print(
                "SIMULATED PAIRS warning: B4 mean true-pair score "
                f"({sanity['true_pair_mean']:.6f}) is not higher than "
                f"random-pair score ({sanity['random_pair_mean']:.6f}) "
                f"for seed={seed}."
            )
        else:
            print(
                f"SIMULATED PAIRS: seed={seed} B4 score sanity - "
                f"true pairs={sanity['true_pair_mean']:.6f}, "
                f"random pairs={sanity['random_pair_mean']:.6f}"
            )
        all_seed_rows.extend(seed_rows)
        roc_curves_by_seed[seed] = seed_curves
        del model

    table = _aggregate(all_seed_rows)
    _write_results(output_dir, table, all_seed_rows, roc_curves_by_seed)

    print("SIMULATED PAIRS: correlation results (mean +/- std across 3 seeds)")
    print(
        "N\tmethod\ttop-1\ttop-5\tTPR@0.2\tTPR@1e-2\tTPR@1e-3"
    )
    for row in table:
        print(
            f"{row['N']}\t{row['method']}\t"
            f"{row['top1_mean']:.4f} +/- {row['top1_std']:.4f}\t"
            f"{row['top5_mean']:.4f} +/- {row['top5_std']:.4f}\t"
            f"{row['tpr_fpr_0.2_mean']:.4f} +/- {row['tpr_fpr_0.2_std']:.4f}\t"
            f"{row['tpr_fpr_0.01_mean']:.4f} +/- {row['tpr_fpr_0.01_std']:.4f}\t"
            f"{row['tpr_fpr_0.001_mean']:.4f} +/- "
            f"{row['tpr_fpr_0.001_std']:.4f}"
        )
    print(f"SIMULATED PAIRS: table saved to {output_dir / 'table_correlation.csv'}")
    print(f"SIMULATED PAIRS: ROC data saved to {output_dir / 'roc_correlation.csv'}")
    print(
        "SIMULATED PAIRS: ROC_DATA is available as "
        "ROC_DATA[(N, method)]['fpr_grid'/'tpr_mean'/'tpr_std'/'tpr_by_seed']."
    )
    print("SIMULATED PAIRS: validation set was not loaded or used.")
    return table


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate SIMULATED PAIRS correlation baselines."
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=project_dir / "processed_r1"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "correlation",
    )
    args = parser.parse_args()
    if SEED not in SEEDS:
        raise ValueError("The existing Step 4 seed must be included in SEEDS")
    run_evaluation(args.splits_dir, args.output_dir)


if __name__ == "__main__":
    main()
