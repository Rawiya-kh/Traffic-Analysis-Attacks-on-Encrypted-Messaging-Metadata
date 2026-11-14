#!/usr/bin/env python3
"""Step 4: Simulated paired flows (SIMULATED PAIRS only)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


# Step 4: Simulated paired flows — SIMULATED PAIRS configuration.
NOISE_SIGMA = 0.30
DUR_NOISE = 1.0
WINDOW = 30.0
DELAY_MEAN = 0.15
DELAY_STD = 0.08
SEED = 42
PACKET_LOSS_MAX = 0.03
TIMING_NOISE_SIGMA = 0.15
RANDOM_GROUP_COUNT = 20
TIMING_GROUP_SIZES = (10, 50, 100)
DIRECTION_SWAP_PAIRS = []

FEATURE_COLUMNS = (
    "bytes_sent",
    "bytes_received",
    "packets_sent",
    "packets_received",
    "mean_iat_seconds",
    "std_iat_seconds",
    "duration_seconds",
)
MIRROR_WORDS = {
    "sent": "received",
    "received": "sent",
    "send": "receive",
    "receive": "send",
}


def _column_indices(column_names: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(column_names)}


def make_view_B(
    X: np.ndarray,
    rng: np.random.Generator,
    sigma: float = NOISE_SIGMA,
) -> np.ndarray:
    """Make a noisy simulated second-vantage view without mutating view A."""
    view_a = np.asarray(X, dtype=np.float64)
    if view_a.ndim != 2 or view_a.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected a 2D feature matrix with {len(FEATURE_COLUMNS)} columns"
        )
    if not np.isfinite(view_a).all() or (view_a < 0).any():
        raise ValueError("SIMULATED PAIRS input must be finite and non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not 0 <= PACKET_LOSS_MAX <= 1:
        raise ValueError("PACKET_LOSS_MAX must be between zero and one")
    if TIMING_NOISE_SIGMA < 0 or DUR_NOISE < 0:
        raise ValueError("SIMULATED PAIRS noise scales must be non-negative")

    view_b = view_a.copy()
    indices = _column_indices(FEATURE_COLUMNS)
    byte_columns = [indices["bytes_sent"], indices["bytes_received"]]
    packet_columns = [indices["packets_sent"], indices["packets_received"]]
    timing_columns = [indices["mean_iat_seconds"], indices["std_iat_seconds"]]
    duration_column = indices["duration_seconds"]

    if packet_columns and not np.equal(
        view_b[:, packet_columns], np.floor(view_b[:, packet_columns])
    ).all():
        raise ValueError("SIMULATED PAIRS packet counts must be integer-valued")

    # Apply row-level packet loss; packet counts use actual binomial losses.
    loss_rates = rng.uniform(0.0, PACKET_LOSS_MAX, size=(len(view_b), 1))
    if len(view_b):
        packet_counts = view_b[:, packet_columns].astype(np.int64)
        lost_packets = rng.binomial(packet_counts, loss_rates)
        view_b[:, packet_columns] = packet_counts - lost_packets
        view_b[:, byte_columns] *= 1.0 - loss_rates

    if len(view_b):
        view_b[:, byte_columns] *= rng.lognormal(
            mean=0.0, sigma=sigma, size=(len(view_b), len(byte_columns))
        )
        view_b[:, timing_columns] *= rng.lognormal(
            mean=0.0,
            sigma=TIMING_NOISE_SIGMA,
            size=(len(view_b), len(timing_columns)),
        )
        view_b[:, duration_column] += np.abs(
            rng.normal(0.0, DUR_NOISE, size=len(view_b))
        )

    for left_name, right_name in DIRECTION_SWAP_PAIRS:
        if left_name not in indices or right_name not in indices:
            raise ValueError(
                f"Unknown feature in DIRECTION_SWAP_PAIRS: "
                f"{left_name!r}, {right_name!r}"
            )
        left, right = indices[left_name], indices[right_name]
        view_b[:, [left, right]] = view_b[:, [right, left]]
    return view_b


def make_group(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Create SIMULATED PAIRS and same-group timing distractors."""
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    indices = np.asarray(idx, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("X and y must contain the same number of sessions")
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("idx must contain at least one session index")
    if (indices < 0).any() or (indices >= len(features)).any():
        raise IndexError("idx contains a session index outside X and y")

    # Copy before view generation so the source split remains unchanged.
    XA = features[indices].copy()
    group_labels = labels[indices].copy()
    XB = make_view_B(XA, rng)
    tA = rng.uniform(0.0, WINDOW, size=len(indices))
    delay = np.clip(
        rng.normal(DELAY_MEAN, DELAY_STD, size=len(indices)), 0.01, None
    )
    tB = tA + delay
    return {"XA": XA, "XB": XB, "tA": tA, "tB": tB, "y": group_labels}


def mirror_label(label: str) -> str:
    """Swap direction words in a class label while preserving their casing."""
    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        mirrored = MIRROR_WORDS[word.lower()]
        if word.isupper():
            return mirrored.upper()
        if word.istitle():
            return mirrored.title()
        return mirrored

    return re.sub(
        r"\b(?:sent|received|send|receive)\b",
        replace,
        label,
        flags=re.IGNORECASE,
    )


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected_columns = [*FEATURE_COLUMNS, "label"]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, got {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: split is empty")
    x = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([row["label"] for row in rows], dtype=str)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{path}: found negative or non-finite feature values")
    return x, y


def _relative_difference(original: np.ndarray, simulated: np.ndarray) -> float:
    denominator = np.maximum.reduce(
        (
            np.abs(original),
            np.abs(simulated),
            np.full_like(original, np.finfo(np.float64).eps),
        )
    )
    return float(np.mean(np.abs(simulated - original) / denominator))


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    splits_dir = project_dir / "processed_r1"

    # Keep the existing evaluator's lowercase variable names and fixed splits.
    x_train, y_train = _load_split(splits_dir / "train.csv")
    x_val, y_val = _load_split(splits_dir / "val.csv")
    x_test, y_test = _load_split(splits_dir / "test.csv")

    rng = np.random.default_rng(SEED)
    test_indices = rng.choice(len(x_test), size=50, replace=False)
    check_group = make_group(x_test, y_test, test_indices, rng)
    XA, XB = check_group["XA"], check_group["XB"]
    shape_match = XA.shape == XB.shape
    finite = bool(np.isfinite(XA).all() and np.isfinite(XB).all())
    nonnegative_sizes_duration = bool(
        (XA[:, [0, 1, 6]] >= 0).all() and (XB[:, [0, 1, 6]] >= 0).all()
    )
    byte_total_a = XA[:, 0] + XA[:, 1]
    byte_total_b = XB[:, 0] + XB[:, 1]

    print("Step 4: Simulated paired flows - SIMULATED PAIRS")
    print(
        f"SIMULATED PAIRS: XA shape={XA.shape}, XB shape={XB.shape}, "
        f"shapes match={shape_match}"
    )
    print(
        "SIMULATED PAIRS: no NaN/inf="
        f"{finite}, no negative byte sizes/durations="
        f"{nonnegative_sizes_duration}"
    )
    print(
        "SIMULATED PAIRS: N=50 mean symmetric relative difference - "
        f"bytes={_relative_difference(byte_total_a, byte_total_b):.6f}, "
        f"duration={_relative_difference(XA[:, 6], XB[:, 6]):.6f}"
    )
    delays = check_group["tB"] - check_group["tA"]
    print(
        "SIMULATED PAIRS: N=50 tB-tA seconds - "
        f"mean={np.mean(delays):.6f}, std={np.std(delays):.6f}"
    )

    classes = sorted(set(y_train))
    mirror_map = {label: mirror_label(label) for label in classes}
    print("SIMULATED PAIRS: mirror-label map")
    for label, mirrored in mirror_map.items():
        print(f"  {label} -> {mirrored}")

    print(
        "SIMULATED PAIRS: timing-only nearest-partner top-1 "
        f"({RANDOM_GROUP_COUNT} random groups per N)"
    )
    print("N\tmean accuracy\tstd accuracy")
    for group_size in TIMING_GROUP_SIZES:
        group_accuracies = []
        for _ in range(RANDOM_GROUP_COUNT):
            indices = rng.choice(len(x_test), size=group_size, replace=False)
            group = make_group(x_test, y_test, indices, rng)
            distances = np.abs(group["tA"][:, None] - group["tB"][None, :])
            nearest_b = np.argmin(distances, axis=1)
            group_accuracies.append(float(np.mean(nearest_b == np.arange(group_size))))
        print(
            f"{group_size}\t{np.mean(group_accuracies):.4f}\t"
            f"{np.std(group_accuracies):.4f}"
        )

    if not (
        shape_match
        and finite
        and nonnegative_sizes_duration
        and np.isfinite(delays).all()
        and (delays >= 0.01 - 1e-12).all()
    ):
        raise AssertionError("SIMULATED PAIRS verification checks failed")


if __name__ == "__main__":
    main()
