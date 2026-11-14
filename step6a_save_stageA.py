#!/usr/bin/env python3
"""Save the selected Stage A activity classifier without editing prior results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from evaluate_activity import FEATURE_COLUMNS, _read_split


SEED = 42
MODEL_NAMES = {
    "B2-random-forest": "Random Forest",
    "B3-xgboost": "XGBoost",
}


def _validation_f1(summary_path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["model"]
            if row["split"] == "val" and name in MODEL_NAMES:
                scores[name] = float(row["macro_f1_mean"])
    missing = set(MODEL_NAMES) - set(scores)
    if missing:
        raise ValueError(
            f"{summary_path}: missing validation results for {sorted(missing)}"
        )
    return scores


def _make_model(model_name: str, class_count: int) -> Any:
    if model_name == "B2-random-forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    if model_name == "B3-xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
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
            random_state=SEED,
        )
    raise ValueError(f"Unsupported Stage A model: {model_name}")


def save_stage_a(
    project_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    scores = _validation_f1(
        project_dir / "results" / "activity" / "activity_summary.csv"
    )
    model_name = max(scores, key=scores.get)
    classes = sorted(
        set(
            _read_split(project_dir / "processed_r1" / "train.csv")[1].tolist()
        )
    )
    x_train, y_train = _read_split(project_dir / "processed_r1" / "train.csv")
    if len(classes) != 8:
        raise ValueError(f"Expected eight application classes, found {len(classes)}")

    model = _make_model(model_name, len(classes))
    if model_name == "B3-xgboost":
        class_to_index = {label: index for index, label in enumerate(classes)}
        y_train_encoded = np.asarray(
            [class_to_index[label] for label in y_train], dtype=np.int64
        )
        sample_weights = compute_sample_weight(
            class_weight="balanced", y=y_train
        )
        # Use train only for both fitting and early stopping; validation is not used.
        model.fit(
            x_train,
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=[(x_train, y_train_encoded)],
            verbose=False,
        )
    else:
        model.fit(x_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes_": np.asarray(classes, dtype=str),
            "feature_columns": tuple(FEATURE_COLUMNS),
            "model_name": model_name,
            "seed": SEED,
            "validation_macro_f1": scores,
        },
        output_path,
        compress=3,
    )
    print("SIMULATED PAIRS: Stage A application classifier checkpoint")
    print(
        f"Selected {MODEL_NAMES[model_name]} using existing validation macro-F1: "
        + ", ".join(
            f"{MODEL_NAMES[name]}={score:.6f}" for name, score in scores.items()
        )
    )
    print(f"Seed: {SEED}; train sessions: {len(y_train)}")
    print(f"Feature columns: {', '.join(FEATURE_COLUMNS)}")
    print(f"Saved class order (classes_): {classes}")
    print(f"Saved checkpoint to {output_path}")
    print("Validation set was not loaded or used for fitting.")
    return {"model": model, "classes_": np.asarray(classes, dtype=str)}


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Save the best existing Stage A classifier as a joblib bundle."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "models" / "stageA.joblib",
    )
    args = parser.parse_args()
    save_stage_a(project_dir, args.output)


if __name__ == "__main__":
    main()
