#!/usr/bin/env python3
"""Step 7: Abstention and defences (SIMULATED PAIRS)."""

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
    _load_split,
    make_group,
)
from step5_correlation_baselines import (
    GROUP_SIZES,
    PAIR_BUDGET,
    ROC_FPR_GRID,
    SEEDS,
    TOP_K,
    _fit_siamese,
    _read_splits,
    _score_rules,
    _score_siamese,
    _score_timing,
)
from step6_imlink_fusion import (
    FUSION_FEATURE_NAMES,
    FUSION_METHOD,
    FUSION_VARIANTS,
    LOGISTIC_MAX_ITER,
    STAGE_A_F1_TOLERANCE,
    _pair_features,
    _prior_stage_a_test_f1,
    _stage_a_probabilities,
)


# Step 7: SIMULATED PAIRS evaluation settings.
ABSTAIN_GROUP_SIZES = (50, 100)
ABSTAIN_COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3)
DEFENCE_GROUP_SIZE = 50
DEFENCE_GROUP_COUNT = 40
VALIDATION_GROUP_COUNT = 30
VALIDATION_GROUP_SIZE = 50
TIE_BREAK_EPSILON = 1e-7
CLEAN_FUSION_RECIPE = {
    "validation_group_count": VALIDATION_GROUP_COUNT,
    "validation_group_size": VALIDATION_GROUP_SIZE,
    "feature_names": FUSION_FEATURE_NAMES,
    "logistic_class_weight": "balanced",
    "logistic_max_iter": LOGISTIC_MAX_ITER,
}

DEFENCES = (
    {"name": "none", "padding_bytes": None, "jitter_seconds": 0.0},
    {"name": "padding_256", "padding_bytes": 256, "jitter_seconds": 0.0},
    {"name": "padding_1024", "padding_bytes": 1024, "jitter_seconds": 0.0},
    {"name": "padding_4096", "padding_bytes": 4096, "jitter_seconds": 0.0},
    {"name": "jitter_0.5", "padding_bytes": None, "jitter_seconds": 0.5},
    {"name": "jitter_2.0", "padding_bytes": None, "jitter_seconds": 2.0},
    {"name": "combined_1024_0.5", "padding_bytes": 1024, "jitter_seconds": 0.5},
)
ATTACKER_SETTINGS = ("clean", "adaptive")
ABSTAIN_CURVES: dict[tuple[int, str], dict[str, np.ndarray]] = {}


def _check_stage_a_checkpoint(
    bundle: dict[str, Any],
    project_dir: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    classes = np.asarray(bundle["classes_"], dtype=str)
    if set(classes) != set(y_test):
        raise ValueError("Stage A classes do not match test application labels")
    probabilities = _stage_a_probabilities(bundle, x_test)
    predictions = classes[np.argmax(probabilities, axis=1)]
    score = float(
        f1_score(
            y_test,
            predictions,
            labels=classes.tolist(),
            average="macro",
            zero_division=0,
        )
    )
    prior = _prior_stage_a_test_f1(
        project_dir / "results" / "activity" / "activity_metrics_by_seed.csv",
        str(bundle["model_name"]),
        int(bundle["seed"]),
    )
    difference = abs(score - prior)
    print(
        f"SIMULATED PAIRS: Stage A {type(bundle['model']).__name__}, "
        f"seed={bundle['seed']}, classes_={classes.tolist()}"
    )
    print(
        f"SIMULATED PAIRS: Stage A test macro-F1={score:.8f}, "
        f"saved baseline={prior:.8f}, difference={difference:.8f}"
    )
    if difference > STAGE_A_F1_TOLERANCE:
        raise RuntimeError(
            "Stage A test macro-F1 differs from its saved baseline by more "
            f"than {STAGE_A_F1_TOLERANCE:.3f}; stopping before Step 7."
        )
    return classes, score


def _fit_clean_fusion(
    x_val: np.ndarray,
    y_val: np.ndarray,
    stage_a: dict[str, Any],
    siamese: Any,
    siamese_scaler: StandardScaler,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    rng = np.random.default_rng(seed)
    feature_parts = []
    label_parts = []
    for _ in range(VALIDATION_GROUP_COUNT):
        indices = rng.choice(len(x_val), size=VALIDATION_GROUP_SIZE, replace=False)
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

    x_pairs = np.concatenate(feature_parts, axis=0)
    y_pairs = np.concatenate(label_parts)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_pairs)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=seed,
    )
    classifier.fit(x_scaled, y_pairs)
    return scaler, classifier


def _clean_fusion_path(project_dir: Path, seed: int) -> Path:
    return project_dir / "models" / f"fusion_clean_seed{seed}.joblib"


def _load_or_reconstruct_clean_fusion(
    project_dir: Path,
    x_val: np.ndarray,
    y_val: np.ndarray,
    stage_a: dict[str, Any],
    siamese: Any,
    siamese_scaler: StandardScaler,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    path = _clean_fusion_path(project_dir, seed)
    if path.exists():
        bundle = joblib.load(path)
        if (
            not isinstance(bundle, dict)
            or bundle.get("seed") != seed
            or bundle.get("recipe") != CLEAN_FUSION_RECIPE
            or not isinstance(bundle.get("scaler"), StandardScaler)
            or not isinstance(bundle.get("classifier"), LogisticRegression)
        ):
            raise ValueError(
                f"{path}: existing clean fusion artifact is invalid or uses a "
                "different Step 6 recipe; refusing to refit it"
            )
        print(
            f"SIMULATED PAIRS: reusing saved clean fusion model for seed={seed} "
            f"from {path}"
        )
        return bundle["scaler"], bundle["classifier"]

    scaler, classifier = _fit_clean_fusion(
        x_val, y_val, stage_a, siamese, siamese_scaler, seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "seed": seed,
            "recipe": CLEAN_FUSION_RECIPE,
            "reconstructed_from_step6_recipe": True,
            "reason": "Step 6 did not save its fitted clean fusion model.",
        },
        path,
        compress=3,
    )
    print(
        f"SIMULATED PAIRS: reconstructed clean fusion model from the Step 6 "
        f"recipe because Step 6 did not save it; saved seed={seed} to {path}"
    )
    return scaler, classifier


def _apply_defence(
    group: dict[str, np.ndarray],
    defence: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], float, float]:
    defended = {key: value.copy() for key, value in group.items()}
    original_bytes = float(
        group["XA"][:, :2].sum() + group["XB"][:, :2].sum()
    )
    padding_bytes = defence["padding_bytes"]
    if padding_bytes is not None:
        for view_name in ("XA", "XB"):
            byte_values = defended[view_name][:, :2]
            positive = byte_values > 0
            byte_values[positive] = (
                np.ceil(byte_values[positive] / padding_bytes) * padding_bytes
            )
    padded_bytes = float(
        defended["XA"][:, :2].sum() + defended["XB"][:, :2].sum()
    )
    overhead_pct = (
        100.0 * (padded_bytes / original_bytes - 1.0)
        if original_bytes > 0
        else 0.0
    )

    jitter_seconds = float(defence["jitter_seconds"])
    if jitter_seconds:
        time_offsets_a = rng.uniform(
            0.0, jitter_seconds, size=len(defended["tA"])
        )
        time_offsets_b = rng.uniform(
            0.0, jitter_seconds, size=len(defended["tB"])
        )
        defended["tA"] += time_offsets_a
        defended["tB"] += time_offsets_b
        for view_name in ("XA", "XB"):
            timing = defended[view_name][:, 4:6]
            timing *= 1.0 + rng.uniform(0.0, jitter_seconds, size=timing.shape)
        added_delay = float(
            (time_offsets_a.sum() + time_offsets_b.sum())
            / (len(time_offsets_a) + len(time_offsets_b))
        )
    else:
        added_delay = 0.0
    return defended, overhead_pct, added_delay


def _fusion_score(
    group: dict[str, np.ndarray],
    stage_a: dict[str, Any],
    siamese: Any,
    siamese_scaler: StandardScaler,
    fusion_scaler: StandardScaler,
    fusion_classifier: LogisticRegression,
    tie_rng: np.random.Generator,
    consume_all_variant_ties: bool,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities_a = _stage_a_probabilities(stage_a, group["XA"])
    probabilities_b = _stage_a_probabilities(stage_a, group["XB"])
    features, _ = _pair_features(
        group,
        probabilities_a,
        probabilities_b,
        siamese,
        siamese_scaler,
    )
    flat = features.reshape(-1, len(FUSION_FEATURE_NAMES))
    scaled = fusion_scaler.transform(flat)
    group_size = len(group["XA"])
    scores = fusion_classifier.predict_proba(scaled)[:, 1].reshape(
        group_size, group_size
    )
    scores += tie_rng.uniform(0.0, TIE_BREAK_EPSILON, size=scores.shape)

    if consume_all_variant_ties:
        for _ in range(len(FUSION_VARIANTS) - 1):
            tie_rng.uniform(0.0, TIE_BREAK_EPSILON, size=scores.shape)
    return scores, probabilities_a


def _top1_and_confidence(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted_partner = np.argmax(scores, axis=1)
    ordered = np.partition(scores, kth=scores.shape[1] - 2, axis=1)
    confidence = ordered[:, -1] - ordered[:, -2]
    correct = predicted_partner == np.arange(scores.shape[0])
    return confidence, correct


def _curve_from_groups(
    score_groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = []
    correctness = []
    for scores in score_groups:
        confidence, correct = _top1_and_confidence(scores)
        confidences.append(confidence)
        correctness.append(correct)
    confidence = np.concatenate(confidences)
    correct = np.concatenate(correctness).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    cumulative_accuracy = np.cumsum(correct[order]) / np.arange(
        1, len(order) + 1
    )
    coverage_grid = np.arange(1, len(order) + 1, dtype=np.float64) / len(order)
    return coverage_grid, cumulative_accuracy, order


def _abstain_summary(
    per_seed: dict[int, tuple[np.ndarray, np.ndarray]],
    group_size: int,
    method: str,
) -> list[dict[str, Any]]:
    coverage_grid = per_seed[SEEDS[0]][0]
    accuracy_by_seed = np.stack(
        [per_seed[seed][1] for seed in SEEDS], axis=0
    )
    ABSTAIN_CURVES[(group_size, method)] = {
        "coverage": coverage_grid,
        "accuracy_by_seed": accuracy_by_seed,
        "accuracy_mean": accuracy_by_seed.mean(axis=0),
        "accuracy_std": accuracy_by_seed.std(axis=0, ddof=1),
    }

    rows = []
    session_count = len(coverage_grid)
    for coverage in ABSTAIN_COVERAGE_LEVELS:
        answered = max(1, int(np.ceil(coverage * session_count)))
        seed_accuracy = accuracy_by_seed[:, answered - 1]
        rows.append(
            {
                "N": group_size,
                "method": method,
                "coverage": coverage,
                "accuracy_mean": float(seed_accuracy.mean()),
                "accuracy_std": float(seed_accuracy.std(ddof=1)),
            }
        )
    return rows


def _pooled_binary_metrics(
    score_groups: list[np.ndarray],
) -> tuple[float, float]:
    group_size = score_groups[0].shape[0]
    top1 = statistics.mean(
        float(np.mean(np.argmax(scores, axis=1) == np.arange(group_size)))
        for scores in score_groups
    )
    labels = np.tile(np.eye(group_size, dtype=np.uint8).ravel(), len(score_groups))
    pooled_scores = np.concatenate([scores.ravel() for scores in score_groups])
    fpr, tpr, _ = roc_curve(labels, pooled_scores)
    return top1, float(np.interp(1e-2, fpr, tpr))


def _fit_adaptive_fusion(
    x_val: np.ndarray,
    y_val: np.ndarray,
    stage_a: dict[str, Any],
    siamese: Any,
    siamese_scaler: StandardScaler,
    defence: dict[str, Any],
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    rng = np.random.default_rng(seed)
    feature_parts = []
    label_parts = []
    for _ in range(VALIDATION_GROUP_COUNT):
        indices = rng.choice(len(x_val), size=VALIDATION_GROUP_SIZE, replace=False)
        group = make_group(x_val, y_val, indices, rng)
        defended, _, _ = _apply_defence(group, defence, rng)
        probabilities_a = _stage_a_probabilities(stage_a, defended["XA"])
        probabilities_b = _stage_a_probabilities(stage_a, defended["XB"])
        features, _ = _pair_features(
            defended,
            probabilities_a,
            probabilities_b,
            siamese,
            siamese_scaler,
        )
        feature_parts.append(features.reshape(-1, len(FUSION_FEATURE_NAMES)))
        label_parts.append(np.eye(VALIDATION_GROUP_SIZE, dtype=np.uint8).ravel())

    x_pairs = np.concatenate(feature_parts, axis=0)
    y_pairs = np.concatenate(label_parts)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_pairs)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=LOGISTIC_MAX_ITER,
        random_state=seed,
    )
    classifier.fit(x_scaled, y_pairs)
    return scaler, classifier


def _write_abstain_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["N", "method", "coverage", "accuracy_mean", "accuracy_std"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_curve_data(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["N", "method", "coverage", "accuracy_mean", "accuracy_std"],
        )
        writer.writeheader()
        for (group_size, method), curve in sorted(ABSTAIN_CURVES.items()):
            for index, coverage in enumerate(curve["coverage"]):
                writer.writerow(
                    {
                        "N": group_size,
                        "method": method,
                        "coverage": coverage,
                        "accuracy_mean": curve["accuracy_mean"][index],
                        "accuracy_std": curve["accuracy_std"][index],
                    }
                )


def _aggregate_defence_rows(
    per_seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "imlink_peer_top1",
        "imlink_tpr_fpr_0.01",
        "b0_timing_top1_clean_reference",
        "b1_rules_top1_clean_reference",
        "b4_siamese_top1_clean_reference",
        "stage_a_application_macro_f1",
        "overhead_pct",
        "mean_added_delay_s",
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        grouped.setdefault((row["defence"], row["attacker"]), []).append(row)

    summary = []
    for (defence, attacker), rows in grouped.items():
        result: dict[str, Any] = {"defence": defence, "attacker": attacker}
        for field in fields:
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"] = statistics.mean(values)
            result[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(result)
    return summary


def _write_defence_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_abstain_table(rows: list[dict[str, Any]]) -> None:
    print("SIMULATED PAIRS: abstention table (top-1 among answered)")
    print("N\tmethod\tcoverage\taccuracy mean +/- std")
    for row in rows:
        print(
            f"{row['N']}\t{row['method']}\t{row['coverage']:.1f}\t"
            f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )


def _print_defence_table(rows: list[dict[str, Any]]) -> None:
    print("SIMULATED PAIRS: defence table (mean +/- std over seeds)")
    print(
        "defence\tattacker\tIM-Link top-1\tIM-Link TPR@1e-2\t"
        "B0 top-1\tB1 top-1\tB4 top-1\tStage A macro-F1\t"
        "overhead %\tadded delay s"
    )
    for row in rows:
        print(
            f"{row['defence']}\t{row['attacker']}\t"
            f"{row['imlink_peer_top1_mean']:.4f} +/- "
            f"{row['imlink_peer_top1_std']:.4f}\t"
            f"{row['imlink_tpr_fpr_0.01_mean']:.4f} +/- "
            f"{row['imlink_tpr_fpr_0.01_std']:.4f}\t"
            f"{row['b0_timing_top1_clean_reference_mean']:.4f} +/- "
            f"{row['b0_timing_top1_clean_reference_std']:.4f}\t"
            f"{row['b1_rules_top1_clean_reference_mean']:.4f} +/- "
            f"{row['b1_rules_top1_clean_reference_std']:.4f}\t"
            f"{row['b4_siamese_top1_clean_reference_mean']:.4f} +/- "
            f"{row['b4_siamese_top1_clean_reference_std']:.4f}\t"
            f"{row['stage_a_application_macro_f1_mean']:.4f} +/- "
            f"{row['stage_a_application_macro_f1_std']:.4f}\t"
            f"{row['overhead_pct_mean']:.4f} +/- "
            f"{row['overhead_pct_std']:.4f}\t"
            f"{row['mean_added_delay_s_mean']:.4f} +/- "
            f"{row['mean_added_delay_s_std']:.4f}"
        )


def run_step7(
    project_dir: Path,
    stage_a_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if DIRECTION_SWAP_PAIRS:
        raise ValueError(
            "SIMULATED PAIRS application labels require DIRECTION_SWAP_PAIRS=[]"
        )
    stage_a = joblib.load(stage_a_path)
    required = {"model", "classes_", "feature_columns", "model_name", "seed"}
    if not isinstance(stage_a, dict) or not required.issubset(stage_a):
        raise ValueError(f"{stage_a_path}: invalid Stage A checkpoint")
    print(
        "SIMULATED PAIRS: clean fusion models are Step 6 recipe "
        "reconstructions because Step 6 did not save its fitted models."
    )

    x_train, _, x_test, y_test = _read_splits(project_dir / "processed_r1")
    x_val, y_val = _load_split(project_dir / "processed_r1" / "val.csv")
    classes, _ = _check_stage_a_checkpoint(
        stage_a, project_dir, x_test, y_test
    )
    print("SIMULATED PAIRS: Stage A is loaded and predict_proba-only.")
    print("SIMULATED PAIRS: DIRECTION_SWAP_PAIRS=[] (application labels).")

    clean_seed_runs: dict[int, dict[str, Any]] = {}
    abstain_seed_curves: dict[
        tuple[int, str], dict[int, tuple[np.ndarray, np.ndarray]]
    ] = {}

    # Part A uses the clean full fusion model and B4 embeddings from Steps 5/6.
    for seed in SEEDS:
        siamese, siamese_scaler, _ = _fit_siamese(x_train, seed)
        fusion_scaler, fusion_classifier = _load_or_reconstruct_clean_fusion(
            project_dir,
            x_val,
            y_val,
            stage_a,
            siamese,
            siamese_scaler,
            seed,
        )
        test_rng = np.random.default_rng(seed)
        tie_rng = np.random.default_rng(seed + 100_000)
        per_n: dict[int, dict[str, list[np.ndarray]]] = {
            n: {"IM-Link-full": [], "B4-siamese": []}
            for n in ABSTAIN_GROUP_SIZES
        }

        for group_size in GROUP_SIZES:
            group_count = max(5, PAIR_BUDGET // group_size)
            for _ in range(group_count):
                indices = test_rng.choice(
                    len(x_test), size=group_size, replace=False
                )
                group = make_group(x_test, y_test, indices, test_rng)
                _score_timing(group)
                _score_rules(group, test_rng)
                b4_scores = _score_siamese(group, siamese, siamese_scaler)
                probabilities_a = _stage_a_probabilities(stage_a, group["XA"])
                probabilities_b = _stage_a_probabilities(stage_a, group["XB"])
                pair_features, _ = _pair_features(
                    group,
                    probabilities_a,
                    probabilities_b,
                    siamese,
                    siamese_scaler,
                )
                flat_features = pair_features.reshape(
                    -1, len(FUSION_FEATURE_NAMES)
                )
                scaled_features = fusion_scaler.transform(flat_features)
                fusion_scores = fusion_classifier.predict_proba(
                    scaled_features
                )[:, 1].reshape(group_size, group_size)
                fusion_scores += tie_rng.uniform(
                    0.0, TIE_BREAK_EPSILON, size=fusion_scores.shape
                )
                for _ in range(len(FUSION_VARIANTS) - 1):
                    tie_rng.uniform(
                        0.0, TIE_BREAK_EPSILON, size=fusion_scores.shape
                    )

                if group_size in per_n:
                    per_n[group_size]["IM-Link-full"].append(fusion_scores)
                    per_n[group_size]["B4-siamese"].append(b4_scores)

        clean_seed_runs[seed] = {
            "model": siamese,
            "scaler": siamese_scaler,
            "fusion_scaler": fusion_scaler,
            "fusion_classifier": fusion_classifier,
        }
        for group_size, methods in per_n.items():
            for method, scores in methods.items():
                coverage, accuracy, _ = _curve_from_groups(scores)
                abstain_seed_curves.setdefault(
                    (group_size, method), {}
                )[seed] = (coverage, accuracy)

    abstain_rows = []
    for (group_size, method), curves in sorted(abstain_seed_curves.items()):
        abstain_rows.extend(_abstain_summary(curves, group_size, method))

    step6_path = project_dir / "results" / "fusion" / "table_fusion.csv"
    with step6_path.open(newline="", encoding="utf-8") as source:
        saved_fusion_rows = {
            int(row["N"]): float(row["top1_mean"])
            for row in csv.DictReader(source)
            if row["method"] == FUSION_METHOD
            and int(row["N"]) in ABSTAIN_GROUP_SIZES
        }
    if set(saved_fusion_rows) != set(ABSTAIN_GROUP_SIZES):
        raise ValueError(
            f"Expected Step 6 IM-Link-full rows for N={ABSTAIN_GROUP_SIZES}"
        )
    print("SIMULATED PAIRS: pre-defence clean fusion checks against Step 6")
    for group_size in ABSTAIN_GROUP_SIZES:
        clean_top1 = float(
            np.mean(
                ABSTAIN_CURVES[(group_size, "IM-Link-full")][
                    "accuracy_by_seed"
                ][:, -1]
            )
        )
        step6_top1 = saved_fusion_rows[group_size]
        difference = abs(clean_top1 - step6_top1)
        print(
            f"  N={group_size}: reconstructed={clean_top1:.6f}, "
            f"Step 6={step6_top1:.6f}, difference={difference:.6f}"
        )
        if difference > 0.01:
            raise RuntimeError(
                f"SIMULATED PAIRS stop before defence evaluation: clean "
                f"N={group_size} top-1 differs from Step 6 by more than 0.01."
            )
    print("SIMULATED PAIRS: clean N=50 and N=100 checks PASS.")

    # Part B: one training/evaluation run per seed and defence. The clean
    # Siamese and Stage A remain fixed. Adaptive mode refits only the fusion LR.
    defence_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        siamese = clean_seed_runs[seed]["model"]
        siamese_scaler = clean_seed_runs[seed]["scaler"]
        clean_scaler = clean_seed_runs[seed]["fusion_scaler"]
        clean_classifier = clean_seed_runs[seed]["fusion_classifier"]
        for defence in DEFENCES:
            adaptive_scaler, adaptive_classifier = _fit_adaptive_fusion(
                x_val,
                y_val,
                stage_a,
                siamese,
                siamese_scaler,
                defence,
                seed,
            )
            test_rng = np.random.default_rng(seed)
            clean_tie_rng = np.random.default_rng(seed + 100_000)
            adaptive_tie_rng = np.random.default_rng(seed + 200_000)
            for _ in range(max(5, PAIR_BUDGET // 10)):
                indices = test_rng.choice(len(x_test), size=10, replace=False)
                clean_group = make_group(x_test, y_test, indices, test_rng)
                _score_timing(clean_group)
                _score_rules(clean_group, test_rng)
                for _ in FUSION_VARIANTS:
                    clean_tie_rng.uniform(
                        0.0, TIE_BREAK_EPSILON, size=(10, 10)
                    )
            attacker_scores = {
                "clean": {"top": [], "roc": []},
                "adaptive": {"top": [], "roc": []},
            }
            baseline_scores = {
                "b0": [],
                "b1": [],
                "b4": [],
            }
            stage_a_labels: list[str] = []
            stage_a_predictions: list[str] = []
            overhead_values: list[float] = []
            delay_values: list[float] = []

            for _ in range(DEFENCE_GROUP_COUNT):
                indices = test_rng.choice(
                    len(x_test), size=DEFENCE_GROUP_SIZE, replace=False
                )
                clean_group = make_group(x_test, y_test, indices, test_rng)
                group, overhead_pct, added_delay = _apply_defence(
                    clean_group, defence, test_rng
                )
                overhead_values.append(overhead_pct)
                delay_values.append(added_delay)

                b0_scores = _score_timing(group)
                b1_scores = _score_rules(group, test_rng)
                b4_scores = _score_siamese(group, siamese, siamese_scaler)
                baseline_scores["b0"].append(b0_scores)
                baseline_scores["b1"].append(b1_scores)
                baseline_scores["b4"].append(b4_scores)

                # Preserve Step 6's tie-noise stream: the full model consumes
                # the first matrix, followed by four ablation-model tie draws.
                for attacker, scaler, classifier in (
                    ("clean", clean_scaler, clean_classifier),
                    ("adaptive", adaptive_scaler, adaptive_classifier),
                ):
                    scores, probabilities_a = _fusion_score(
                        group,
                        stage_a,
                        siamese,
                        siamese_scaler,
                        scaler,
                        classifier,
                        clean_tie_rng
                        if attacker == "clean"
                        else adaptive_tie_rng,
                        consume_all_variant_ties=(attacker == "clean"),
                    )
                    attacker_scores[attacker]["top"].append(scores)
                    attacker_scores[attacker]["roc"].append(scores)
                    if attacker == "clean":
                        app_predictions = classes[
                            np.argmax(probabilities_a, axis=1)
                        ]
                        stage_a_labels.extend(group["y"].astype(str).tolist())
                        stage_a_predictions.extend(app_predictions.tolist())

            baseline_metrics = {
                key: _pooled_binary_metrics(matrices)[0]
                for key, matrices in baseline_scores.items()
            }
            stage_a_f1 = float(
                f1_score(
                    stage_a_labels,
                    stage_a_predictions,
                    labels=classes.tolist(),
                    average="macro",
                    zero_division=0,
                )
            )
            for attacker in ATTACKER_SETTINGS:
                imlink_top1, imlink_tpr = _pooled_binary_metrics(
                    attacker_scores[attacker]["top"]
                )
                overhead_mean = float(np.mean(overhead_values))
                if defence["padding_bytes"] is None:
                    overhead_mean = 0.0
                defence_seed_rows.append(
                    {
                        "seed": seed,
                        "defence": defence["name"],
                        "attacker": attacker,
                        "imlink_peer_top1": imlink_top1,
                        "imlink_tpr_fpr_0.01": imlink_tpr,
                        "b0_timing_top1_clean_reference": baseline_metrics["b0"],
                        "b1_rules_top1_clean_reference": baseline_metrics["b1"],
                        "b4_siamese_top1_clean_reference": baseline_metrics["b4"],
                        "stage_a_application_macro_f1": stage_a_f1,
                        "overhead_pct": overhead_mean,
                        "mean_added_delay_s": float(np.mean(delay_values)),
                    }
                )
    none_step6 = [
        row for row in defence_seed_rows
        if row["defence"] == "none" and row["attacker"] == "clean"
    ]
    none_top1_mean = statistics.mean(
        row["imlink_peer_top1"] for row in none_step6
    )
    step6_top1 = saved_fusion_rows[DEFENCE_GROUP_SIZE]
    difference = abs(none_top1_mean - step6_top1)
    print(
        f"SIMULATED PAIRS: none defence N=50 clean IM-Link top-1="
        f"{none_top1_mean:.6f}; Step 6={step6_top1:.6f}; "
        f"difference={difference:.6f}"
    )
    if difference > 0.01:
        raise RuntimeError(
            "SIMULATED PAIRS verification failed: none defence differs from "
            "Step 6 by more than 0.01. No Step 7 outputs were written; check "
            "that the clean validation groups, test sampling, and tie-noise "
            "stream match Step 6."
        )

    for row in defence_seed_rows:
        if row["defence"] in ("none", "jitter_0.5", "jitter_2.0"):
            if row["overhead_pct"] != 0.0:
                raise AssertionError(
                    f"{row['defence']} must have exactly zero byte overhead"
                )

    defence_summary = _aggregate_defence_rows(defence_seed_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_abstain_table(output_dir / "table_abstain.csv", abstain_rows)
    _write_curve_data(output_dir / "coverage_curves.csv")
    _write_defence_table(output_dir / "table_defence.csv", defence_summary)

    for defence in DEFENCES:
        if defence["name"] in ("none", "jitter_0.5", "jitter_2.0"):
            print(
                f"SIMULATED PAIRS: overhead check {defence['name']} = 0% PASS"
            )
    print(
        "SIMULATED PAIRS: none defence reproduces Step 6 within 0.01 PASS "
        f"(difference={difference:.6f})."
    )
    _print_abstain_table(abstain_rows)
    _print_defence_table(defence_summary)
    print(
        f"SIMULATED PAIRS: files saved in {output_dir}; "
        "ABSTAIN_CURVES[(N, method)] is available for plotting."
    )
    return abstain_rows, defence_summary


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate SIMULATED PAIRS abstention and traffic defences."
    )
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=project_dir / "models" / "stageA.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "step7",
    )
    args = parser.parse_args()
    run_step7(project_dir, args.stage_a, args.output_dir)


if __name__ == "__main__":
    main()
