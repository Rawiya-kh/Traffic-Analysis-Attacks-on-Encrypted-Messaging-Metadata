#!/usr/bin/env python3
"""Step 6: Stage C fusion (SIMULATED PAIRS)."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

from step4_simulated_paired_flows import (
    DELAY_MEAN,
    DIRECTION_SWAP_PAIRS,
    FEATURE_COLUMNS,
    SEED,
    _load_split,
    make_group,
)
from step5_correlation_baselines import (
    GROUP_SIZES,
    METHODS as BASELINE_METHODS,
    PAIR_BUDGET,
    ROC_FPR_GRID,
    SEEDS,
    SiameseEncoder,
    _embed,
    _fit_siamese,
    _score_rules,
    _score_siamese,
    _score_timing,
)


# Step 6: SIMULATED PAIRS fusion settings.
VALIDATION_GROUP_COUNT = 30
VALIDATION_GROUP_SIZE = 50
TOP_K = 5
FPR_TARGETS = (0.2, 1e-2, 1e-3)
TIE_BREAK_EPSILON = 1e-7
LOGISTIC_MAX_ITER = 2000
STAGE_A_F1_TOLERANCE = 0.01
FUSION_FEATURE_NAMES = (
    "f0_embedding_cosine",
    "f1_timing_log_error",
    "f2_duration_log_difference",
    "f3_bytes_log_difference",
    "f4_application_consistency",
)
FUSION_VARIANTS = {
    "IM-Link-full": (0, 1, 2, 3, 4),
    "IM-Link-no-consistency": (0, 1, 2, 3),
    "IM-Link-no-pairwise": (0, 4),
    "IM-Link-no-siamese": (1, 2, 3, 4),
    "IM-Link-no-timing": (0, 2, 3, 4),
}
BASELINE_METHODS = tuple(BASELINE_METHODS)
FUSION_METHOD = "IM-Link-full"
TABLE_METHODS = (*BASELINE_METHODS, FUSION_METHOD)
ROC_DATA: dict[tuple[int, str], dict[str, Any]] = {}


def _stage_a_probabilities(
    bundle: dict[str, Any], features: np.ndarray
) -> np.ndarray:
    model = bundle["model"]
    feature_columns = tuple(bundle["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Stage A feature columns do not match the existing R1 feature order"
        )

    if hasattr(model, "feature_names_in_"):
        import pandas as pd

        model_input: Any = pd.DataFrame(features, columns=feature_columns)
    else:
        model_input = features
    probabilities = np.asarray(model.predict_proba(model_input), dtype=np.float64)
    classes = np.asarray(bundle["classes_"], dtype=str)
    if probabilities.shape != (len(features), len(classes)):
        raise ValueError("Stage A predict_proba returned an unexpected shape")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Stage A returned invalid application probabilities")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Stage A probabilities do not sum to one")
    return probabilities


def _prior_stage_a_test_f1(
    results_path: Path, model_name: str, seed: int
) -> float:
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        matches = [
            row
            for row in rows
            if row["model"] == model_name
            and row["split"] == "test"
            and int(row["seed"]) == seed
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one prior Stage A TEST result for {model_name}, seed {seed}"
        )
    return float(matches[0]["macro_f1"])


def _pair_features(
    group: dict[str, np.ndarray],
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    siamese: SiameseEncoder,
    siamese_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    XA, XB = group["XA"], group["XB"]
    embeddings_a = _embed(siamese, siamese_scaler, XA)
    embeddings_b = _embed(siamese, siamese_scaler, XB)
    f0 = embeddings_a @ embeddings_b.T

    time_error = np.abs(
        (group["tB"][None, :] - group["tA"][:, None]) - DELAY_MEAN
    )
    f1 = np.log(time_error + 0.01)

    duration_a = np.log(XA[:, 6, None] + 1e-3)
    duration_b = np.log(XB[None, :, 6] + 1e-3)
    f2 = np.abs(duration_a - duration_b)

    total_bytes_a = XA[:, 0] + XA[:, 1]
    total_bytes_b = XB[:, 0] + XB[:, 1]
    bytes_a = np.log(total_bytes_a[:, None] + 1.0)
    bytes_b = np.log(total_bytes_b[None, :] + 1.0)
    f3 = np.abs(bytes_a - bytes_b)

    consistency = probabilities_a @ probabilities_b.T
    f4 = np.log(consistency + 1e-3)
    features = np.stack((f0, f1, f2, f3, f4), axis=2)
    if not np.isfinite(features).all():
        raise ValueError("SIMULATED PAIRS fusion features are non-finite")
    return features, f0


def _fit_fusion_variants(
    x_val: np.ndarray,
    y_val: np.ndarray,
    stage_a: dict[str, Any],
    siamese: SiameseEncoder,
    siamese_scaler: StandardScaler,
    seed: int,
) -> dict[str, tuple[StandardScaler, LogisticRegression]]:
    rng = np.random.default_rng(seed)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for _ in range(VALIDATION_GROUP_COUNT):
        indices = rng.choice(
            len(x_val), size=VALIDATION_GROUP_SIZE, replace=False
        )
        group = make_group(x_val, y_val, indices, rng)
        probabilities_a = _stage_a_probabilities(stage_a, group["XA"])
        probabilities_b = _stage_a_probabilities(stage_a, group["XB"])
        features, _ = _pair_features(
            group,
            probabilities_a,
            probabilities_b,
            siamese,
            siamese_scaler,
        )
        feature_parts.append(features.reshape(-1, len(FUSION_FEATURE_NAMES)))
        label_parts.append(np.eye(VALIDATION_GROUP_SIZE, dtype=np.uint8).ravel())

    x_pair_val = np.concatenate(feature_parts, axis=0)
    y_pair_val = np.concatenate(label_parts)
    fitted: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    for name, selected_features in FUSION_VARIANTS.items():
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_pair_val[:, selected_features])
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=seed,
        )
        classifier.fit(x_scaled, y_pair_val)
        fitted[name] = (scaler, classifier)
    return fitted


def _pair_scores(
    features: np.ndarray,
    fitted: dict[str, tuple[StandardScaler, LogisticRegression]],
    tie_rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    group_size = features.shape[0]
    flattened = features.reshape(-1, len(FUSION_FEATURE_NAMES))
    scores: dict[str, np.ndarray] = {}
    for name, selected_features in FUSION_VARIANTS.items():
        scaler, classifier = fitted[name]
        transformed = scaler.transform(flattened[:, selected_features])
        probabilities = classifier.predict_proba(transformed)[:, 1]
        matrix = probabilities.reshape(group_size, group_size)
        matrix += tie_rng.uniform(
            0.0, TIE_BREAK_EPSILON, size=matrix.shape
        )
        scores[name] = matrix
    return scores


def _aggregate(
    seed_rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    group_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    metric_names = (
        "top1",
        "top5",
        "tpr_fpr_0.2",
        "tpr_fpr_0.01",
        "tpr_fpr_0.001",
    )
    rows = []
    for group_size in group_sizes:
        for method in methods:
            samples = [
                row
                for row in seed_rows
                if row["N"] == group_size and row["method"] == method
            ]
            if not samples:
                raise ValueError(f"No SIMULATED PAIRS metrics for {group_size}/{method}")
            aggregate: dict[str, Any] = {"N": group_size, "method": method}
            for metric in metric_names:
                values = [sample[metric] for sample in samples]
                aggregate[f"{metric}_mean"] = statistics.mean(values)
                aggregate[f"{metric}_std"] = statistics.stdev(values)
            rows.append(aggregate)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _joint_rows(
    per_seed: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_names = (
        "stage_a_application_macro_f1",
        "imlink_application_macro_f1",
        "b4_peer_top1",
        "imlink_peer_top1",
        "b4_peer_plus_stage_a_joint_accuracy",
        "imlink_peer_plus_combined_app_joint_accuracy",
    )
    rows = []
    for metric in metric_names:
        values = [per_seed[seed][metric] for seed in SEEDS]
        mean, std = _mean_std(values)
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                **{f"seed_{seed}": per_seed[seed][metric] for seed in SEEDS},
            }
        )
    return rows


def _write_outputs(
    output_dir: Path,
    fusion_table: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    all_seed_rows: list[dict[str, Any]],
    roc_by_seed: dict[int, dict[tuple[int, str], np.ndarray]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "table_fusion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fusion_table[0]))
        writer.writeheader()
        writer.writerows(fusion_table)

    with (output_dir / "table_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)

    with (output_dir / "table_joint.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)

    with (output_dir / "fusion_metrics_by_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(all_seed_rows[0]))
        writer.writeheader()
        writer.writerows(all_seed_rows)

    global ROC_DATA
    ROC_DATA.clear()
    roc_methods = (*BASELINE_METHODS, *FUSION_VARIANTS.keys())
    with (output_dir / "roc_fusion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        fields = ["N", "method", "fpr", "tpr_mean", "tpr_std", *[
            f"tpr_seed_{seed}" for seed in SEEDS
        ]]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for group_size in GROUP_SIZES:
            for method in roc_methods:
                curves = np.stack(
                    [
                        roc_by_seed[seed][(group_size, method)]
                        for seed in SEEDS
                    ]
                )
                mean_tpr = curves.mean(axis=0)
                std_tpr = curves.std(axis=0, ddof=1)
                ROC_DATA[(group_size, method)] = {
                    "fpr_grid": ROC_FPR_GRID.copy(),
                    "tpr_by_seed": curves,
                    "tpr_mean": mean_tpr,
                    "tpr_std": std_tpr,
                }
                for index, fpr_value in enumerate(ROC_FPR_GRID):
                    writer.writerow(
                        {
                            "N": group_size,
                            "method": method,
                            "fpr": fpr_value,
                            "tpr_mean": mean_tpr[index],
                            "tpr_std": std_tpr[index],
                            **{
                                f"tpr_seed_{seed}": curves[seed_index, index]
                                for seed_index, seed in enumerate(SEEDS)
                            },
                        }
                    )


def run_fusion(
    project_dir: Path,
    stage_a_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_a_bundle = joblib.load(stage_a_path)
    required = {"model", "classes_", "feature_columns"}
    if not isinstance(stage_a_bundle, dict) or not required.issubset(stage_a_bundle):
        raise ValueError(f"{stage_a_path}: invalid Stage A joblib bundle")
    if DIRECTION_SWAP_PAIRS:
        raise ValueError(
            "SIMULATED PAIRS application labels require DIRECTION_SWAP_PAIRS=[]"
        )

    x_test, y_test = _load_split(project_dir / "processed_r1" / "test.csv")
    classes = np.asarray(stage_a_bundle["classes_"], dtype=str)
    if set(classes) != set(y_test):
        raise ValueError("Stage A classes do not match the saved test labels")

    print("SIMULATED PAIRS: Stage A application classifier loaded; no refit.")
    print(
        f"SIMULATED PAIRS: Stage A model type="
        f"{type(stage_a_bundle['model']).__name__} "
        f"({stage_a_bundle.get('model_name')}), seed={stage_a_bundle.get('seed')}"
    )
    print(f"SIMULATED PAIRS: Stage A classes_ order={classes.tolist()}")
    model_classes = getattr(stage_a_bundle["model"], "classes_", None)
    if model_classes is not None:
        print(
            "SIMULATED PAIRS: estimator classes_="
            f"{np.asarray(model_classes).tolist()}"
        )
    print("SIMULATED PAIRS: DIRECTION_SWAP_PAIRS=[]; consistency means same app.")
    stage_a_test_probabilities = _stage_a_probabilities(stage_a_bundle, x_test)
    stage_a_test_predictions = classes[np.argmax(stage_a_test_probabilities, axis=1)]
    stage_a_test_macro_f1 = f1_score(
        y_test,
        stage_a_test_predictions,
        labels=classes.tolist(),
        average="macro",
        zero_division=0,
    )
    prior_f1 = _prior_stage_a_test_f1(
        project_dir / "results" / "activity" / "activity_metrics_by_seed.csv",
        str(stage_a_bundle.get("model_name")),
        int(stage_a_bundle.get("seed")),
    )
    difference = abs(stage_a_test_macro_f1 - prior_f1)
    print(
        f"SIMULATED PAIRS: Stage A TEST macro-F1="
        f"{stage_a_test_macro_f1:.8f}; earlier activity-baseline TEST "
        f"macro-F1={prior_f1:.8f}; absolute difference={difference:.8f}"
    )
    if difference > STAGE_A_F1_TOLERANCE:
        raise SystemExit(
            "SIMULATED PAIRS STOP: Stage A TEST macro-F1 differs from the "
            f"earlier result by more than {STAGE_A_F1_TOLERANCE:.3f}; "
            "no pair features or fusion models were built."
        )
    print("SIMULATED PAIRS: Stage A TEST macro-F1 matches; continuing.")

    x_train, y_train = _load_split(project_dir / "processed_r1" / "train.csv")
    x_val, y_val = _load_split(project_dir / "processed_r1" / "val.csv")
    if set(classes) != set(y_train):
        raise ValueError("Stage A classes do not match the saved train labels")
    if len(x_val) < VALIDATION_GROUP_SIZE:
        raise ValueError("Validation split is smaller than the fusion group size")

    all_seed_rows: list[dict[str, Any]] = []
    roc_by_seed: dict[int, dict[tuple[int, str], np.ndarray]] = {}
    joint_by_seed: dict[int, dict[str, float]] = {}

    for seed in SEEDS:
        siamese, siamese_scaler, loss_history = _fit_siamese(x_train, seed)
        print(
            f"SIMULATED PAIRS: seed={seed} B4 loss "
            f"epoch 1={loss_history[0]:.6f}, "
            f"epoch 20={loss_history[19]:.6f}, "
            f"epoch 40={loss_history[39]:.6f}"
        )
        fitted_variants = _fit_fusion_variants(
            x_val, y_val, stage_a_bundle, siamese, siamese_scaler, seed
        )
        full_classifier = fitted_variants[FUSION_METHOD][1]
        full_coefficients = full_classifier.coef_[0]
        coefficient_order = np.argsort(np.abs(full_coefficients))[::-1]
        print(f"SIMULATED PAIRS: seed={seed} standardized full-model coefficients")
        for feature_index in coefficient_order:
            print(
                f"  {FUSION_FEATURE_NAMES[feature_index]}: "
                f"{full_coefficients[feature_index]:+.6f}"
            )

        # A fresh test RNG follows Step 5's group sampling and B1 tie-noise order.
        test_rng = np.random.default_rng(seed)
        tie_rng = np.random.default_rng(seed + 100_000)
        seed_joint_records: dict[str, list[Any]] = {
            "labels": [],
            "stage_a_app": [],
            "imlink_app": [],
            "b4_peer_hit": [],
            "imlink_peer_hit": [],
            "b4_joint": [],
            "imlink_joint": [],
        }
        seed_roc: dict[tuple[int, str], np.ndarray] = {}

        for group_size in GROUP_SIZES:
            group_count = max(5, PAIR_BUDGET // group_size)
            scores_by_method = {
                method: [] for method in (*BASELINE_METHODS, *FUSION_VARIANTS)
            }
            for _ in range(group_count):
                indices = test_rng.choice(
                    len(x_test), size=group_size, replace=False
                )
                group = make_group(x_test, y_test, indices, test_rng)

                # Recompute Step 5 baselines on the same simulated test groups.
                scores_by_method["B0-timing"].append(_score_timing(group))
                scores_by_method["B1-rules"].append(
                    _score_rules(group, test_rng)
                )
                b4_scores = _score_siamese(group, siamese, siamese_scaler)
                scores_by_method["B4-siamese"].append(b4_scores)

                probabilities_a = _stage_a_probabilities(
                    stage_a_bundle, group["XA"]
                )
                probabilities_b = _stage_a_probabilities(
                    stage_a_bundle, group["XB"]
                )
                pair_features, _ = _pair_features(
                    group,
                    probabilities_a,
                    probabilities_b,
                    siamese,
                    siamese_scaler,
                )
                fusion_scores = _pair_scores(
                    pair_features, fitted_variants, tie_rng
                )
                for method, matrix in fusion_scores.items():
                    scores_by_method[method].append(matrix)

                true_rows = np.arange(group_size)
                b4_partner = np.argmax(b4_scores, axis=1)
                imlink_partner = np.argmax(fusion_scores[FUSION_METHOD], axis=1)
                selected_b_probabilities = probabilities_b[imlink_partner]
                combined = probabilities_a * selected_b_probabilities
                normalizers = combined.sum(axis=1, keepdims=True)
                if (normalizers <= 0).any() or not np.isfinite(normalizers).all():
                    raise ValueError(
                        "SIMULATED PAIRS combined application belief cannot be normalized"
                    )
                combined /= normalizers
                stage_a_app_prediction = classes[
                    np.argmax(probabilities_a, axis=1)
                ]
                imlink_app_prediction = classes[np.argmax(combined, axis=1)]
                true_apps = group["y"].astype(str)

                seed_joint_records["labels"].extend(true_apps.tolist())
                seed_joint_records["stage_a_app"].extend(
                    stage_a_app_prediction.tolist()
                )
                seed_joint_records["imlink_app"].extend(
                    imlink_app_prediction.tolist()
                )
                b4_hit = b4_partner == true_rows
                imlink_hit = imlink_partner == true_rows
                stage_a_correct = stage_a_app_prediction == true_apps
                imlink_app_correct = imlink_app_prediction == true_apps
                seed_joint_records["b4_peer_hit"].extend(b4_hit.tolist())
                seed_joint_records["imlink_peer_hit"].extend(imlink_hit.tolist())
                seed_joint_records["b4_joint"].extend(
                    (b4_hit & stage_a_correct).tolist()
                )
                seed_joint_records["imlink_joint"].extend(
                    (imlink_hit & imlink_app_correct).tolist()
                )

            for method, matrices in scores_by_method.items():
                seed_metrics, fpr, tpr = _metrics_for_groups(matrices)
                all_seed_rows.append(
                    {
                        "seed": seed,
                        "N": group_size,
                        "method": method,
                        **seed_metrics,
                    }
                )
                seed_roc[(group_size, method)] = np.interp(
                    ROC_FPR_GRID, fpr, tpr
                )

        joint_labels = np.asarray(seed_joint_records["labels"], dtype=str)
        joint_app_predictions = np.asarray(
            seed_joint_records["imlink_app"], dtype=str
        )
        joint_stage_a_predictions = np.asarray(
            seed_joint_records["stage_a_app"], dtype=str
        )
        joint_by_seed[seed] = {
            "stage_a_application_macro_f1": float(stage_a_test_macro_f1),
            "imlink_application_macro_f1": float(
                f1_score(
                    joint_labels,
                    joint_app_predictions,
                    labels=classes.tolist(),
                    average="macro",
                    zero_division=0,
                )
            ),
            "b4_peer_top1": float(
                np.mean(seed_joint_records["b4_peer_hit"])
            ),
            "imlink_peer_top1": float(
                np.mean(seed_joint_records["imlink_peer_hit"])
            ),
            "b4_peer_plus_stage_a_joint_accuracy": float(
                np.mean(seed_joint_records["b4_joint"])
            ),
            "imlink_peer_plus_combined_app_joint_accuracy": float(
                np.mean(seed_joint_records["imlink_joint"])
            ),
        }
        roc_by_seed[seed] = seed_roc
        del siamese

    methods_in_full_table = TABLE_METHODS
    full_table = _aggregate(
        all_seed_rows, methods_in_full_table, GROUP_SIZES
    )
    ablation_rows = []
    ablation_methods = tuple(FUSION_VARIANTS)
    for summary in _aggregate(
        all_seed_rows, ablation_methods, (50, 100)
    ):
        ablation_rows.append(
            {
                "variant": summary["method"],
                "N": summary["N"],
                "top1_mean": summary["top1_mean"],
                "top1_std": summary["top1_std"],
                "tpr_fpr_0.01_mean": summary["tpr_fpr_0.01_mean"],
                "tpr_fpr_0.01_std": summary["tpr_fpr_0.01_std"],
            }
        )
    joint_rows = _joint_rows(joint_by_seed)
    _write_outputs(
        output_dir,
        full_table,
        ablation_rows,
        joint_rows,
        all_seed_rows,
        roc_by_seed,
    )

    print("SIMULATED PAIRS: table_fusion.csv (mean +/- std over 3 seeds)")
    _print_metric_table(full_table)
    print("SIMULATED PAIRS: table_ablation.csv (N=50 and N=100)")
    print("variant\tN\ttop-1\tTPR@1e-2")
    for row in ablation_rows:
        print(
            f"{row['variant']}\t{row['N']}\t"
            f"{row['top1_mean']:.4f} +/- {row['top1_std']:.4f}\t"
            f"{row['tpr_fpr_0.01_mean']:.4f} +/- "
            f"{row['tpr_fpr_0.01_std']:.4f}"
        )
    print("SIMULATED PAIRS: table_joint.csv (mean +/- std over 3 seeds)")
    for row in joint_rows:
        print(f"{row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f}")

    fusion_by_n = {
        row["N"]: row for row in full_table if row["method"] == FUSION_METHOD
    }
    baseline_by_n = {
        (row["N"], row["method"]): row
        for row in full_table
        if row["method"] in ("B0-timing", "B4-siamese")
    }
    for group_size in GROUP_SIZES:
        stronger_baseline = max(
            baseline_by_n[(group_size, "B0-timing")]["top1_mean"],
            baseline_by_n[(group_size, "B4-siamese")]["top1_mean"],
        )
        fusion_top1 = fusion_by_n[group_size]["top1_mean"]
        if fusion_top1 >= stronger_baseline:
            print(
                f"SIMULATED PAIRS: fusion top-1 check N={group_size} PASS "
                f"({fusion_top1:.4f} >= {stronger_baseline:.4f})"
            )
        else:
            print(
                f"SIMULATED PAIRS warning: fusion top-1 check N={group_size} FAIL "
                f"({fusion_top1:.4f} < {stronger_baseline:.4f}); "
                "no test-set tuning was performed."
            )
    print(
        f"SIMULATED PAIRS: outputs saved under {output_dir}; "
        "ROC_DATA[(N, method)] contains the interpolated curves."
    )
    print("SIMULATED PAIRS: validation set was used only to fit fusion models.")
    return full_table, ablation_rows, joint_rows


def _metrics_for_groups(
    matrices: list[np.ndarray],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    group_size = matrices[0].shape[0]
    top1_values = []
    top5_values = []
    for matrix in matrices:
        top1_values.append(
            float(np.mean(np.argmax(matrix, axis=1) == np.arange(group_size)))
        )
        k = min(TOP_K, group_size)
        candidates = np.argpartition(
            matrix, kth=group_size - k, axis=1
        )[:, -k:]
        top5_values.append(
            float(
                np.mean(
                    np.any(candidates == np.arange(group_size)[:, None], axis=1)
                )
            )
        )
    pooled_labels = np.tile(
        np.eye(group_size, dtype=np.uint8).ravel(), len(matrices)
    )
    pooled_scores = np.concatenate([matrix.ravel() for matrix in matrices])
    fpr, tpr, _ = roc_curve(pooled_labels, pooled_scores)
    metrics = {
        "top1": statistics.mean(top1_values),
        "top5": statistics.mean(top5_values),
        "tpr_fpr_0.2": float(np.interp(0.2, fpr, tpr)),
        "tpr_fpr_0.01": float(np.interp(1e-2, fpr, tpr)),
        "tpr_fpr_0.001": float(np.interp(1e-3, fpr, tpr)),
    }
    return metrics, fpr, tpr


def _print_metric_table(table: list[dict[str, Any]]) -> None:
    print("N\tmethod\ttop-1\ttop-5\tTPR@0.2\tTPR@1e-2\tTPR@1e-3")
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


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate SIMULATED PAIRS Stage C fusion."
    )
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=project_dir / "models" / "stageA.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "fusion",
    )
    args = parser.parse_args()
    if SEED not in SEEDS:
        raise ValueError("Step 4's configured seed must be included in SEEDS")
    run_fusion(project_dir, args.stage_a, args.output_dir)


if __name__ == "__main__":
    main()
