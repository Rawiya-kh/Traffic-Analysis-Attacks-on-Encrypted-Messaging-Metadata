#!/usr/bin/env python3
"""Create IM-Link result figures and the paper-comparison table from saved outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

from step4_simulated_paired_flows import _load_split


METHOD_COLORS = {
    "B0-timing": "#7f7f7f",
    "B1-rules": "#ff7f0e",
    "B4-siamese": "#1f77b4",
    "IM-Link-full": "#d62728",
}
ACTIVITY_COLORS = {
    "B0-majority": "#7f7f7f",
    "B2-bagging": "#2ca02c",
    "B2-random-forest": "#9467bd",
    "B3-xgboost": "#d62728",
}
FIGURE_FILES = (
    "fig1_stageA_f1.png",
    "fig2_confusion_matrix.png",
    "fig3_roc.png",
    "fig4_vs_N.png",
    "fig5_joint.png",
    "fig6_coverage.png",
    "fig7_ablation.png",
    "fig8_defence.png",
)
COMPARISON_COLUMNS = (
    "Study",
    "Task",
    "Dataset and setting",
    "Reported result (as stated in the paper)",
    "Our comparable result",
    "Directly comparable?",
    "Why not / caveats",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved results file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader)


def _index(rows: list[dict[str, str]], *fields: str) -> dict[tuple[str, ...], dict[str, str]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate row key {key} in saved results")
        result[key] = row
    return result


def _number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite saved metric {field}={row[field]}")
    return value


def _mean_std(row: dict[str, str], metric: str) -> str:
    return f"{_number(row, metric + '_mean'):.4f} +/- {_number(row, metric + '_std'):.4f}"


def _save_figure(
    fig: plt.Figure, path: Path, layout_rect: tuple[float, float, float, float] | None = None
) -> None:
    if layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=layout_rect)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _configure_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _load_roc(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    rows = _read_csv(path)
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["N"]), row["method"])
        grouped.setdefault(key, []).append(row)
    curves = {}
    for key, group in grouped.items():
        group.sort(key=lambda row: float(row["fpr"]))
        curves[key] = {
            "fpr": np.asarray([float(row["fpr"]) for row in group]),
            "tpr": np.asarray([float(row["tpr_mean"]) for row in group]),
            "std": np.asarray([float(row["tpr_std"]) for row in group]),
        }
    return curves


def _make_fig1(summary: dict[tuple[str, str], dict[str, str]], path: Path) -> None:
    names = (
        "B0-majority",
        "B2-bagging",
        "B2-random-forest",
        "B3-xgboost",
    )
    labels = ("B0 Majority", "B2 Bagging", "B2 Random Forest", "B3 XGBoost")
    values = [summary[(name, "test")] for name in names]
    means = [_number(row, "macro_f1_mean") for row in values]
    stds = [_number(row, "macro_f1_std") for row in values]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
        color=[ACTIVITY_COLORS[name] for name in names],
    )
    ax.set_ylabel("Test macro-F1 (unitless)")
    ax.set_title("Stage A application identification (test macro-F1)")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                label="Mean ± std over five seeds",
            )
        ],
        loc="upper left",
    )
    _configure_axes(ax)
    _save_figure(fig, path)


def _make_fig2(project_dir: Path, path: Path) -> None:
    bundle_path = project_dir / "models" / "stageA.joblib"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Saved Stage A model is missing: {bundle_path}")
    bundle = joblib.load(bundle_path)
    required = {"model", "classes_", "feature_columns"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"{bundle_path}: invalid Stage A joblib bundle")
    x_test, y_test = _load_split(project_dir / "processed_r1" / "test.csv")
    classes = np.asarray(bundle["classes_"], dtype=str)
    if tuple(bundle["feature_columns"]) != tuple(
        ("bytes_sent", "bytes_received", "packets_sent", "packets_received",
         "mean_iat_seconds", "std_iat_seconds", "duration_seconds")
    ):
        raise ValueError("Stage A feature order differs from the saved test split")
    probabilities = np.asarray(bundle["model"].predict_proba(x_test))
    if probabilities.shape != (len(y_test), len(classes)):
        raise ValueError("Stage A predict_proba returned an unexpected shape")
    predictions = classes[np.argmax(probabilities, axis=1)]
    raw = confusion_matrix(y_test, predictions, labels=classes.tolist())
    row_sums = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw,
        row_sums,
        out=np.zeros_like(raw, dtype=np.float64),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, pad=0.04)
    colorbar.set_label("Row-normalised fraction (unitless)", labelpad=12)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    ax.set_xlabel("Predicted application")
    ax.set_ylabel("True application")
    ax.set_title("Stage A test confusion matrix (saved model, predict only)")
    fig.suptitle("Row-normalised: each true-application row sums to 1.0")
    threshold = normalized.max() / 2 if normalized.size else 0
    for true_index in range(len(classes)):
        for predicted_index in range(len(classes)):
            value = normalized[true_index, predicted_index]
            ax.text(
                predicted_index,
                true_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    _configure_axes(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_fig3(roc: dict[tuple[int, str], dict[str, np.ndarray]], path: Path) -> None:
    methods = ("B0-timing", "B1-rules", "B4-siamese", "IM-Link-full")
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        curve = roc[(50, method)]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=METHOD_COLORS[method],
            label=method,
            linewidth=2,
        )
        lower = np.clip(curve["tpr"] - curve["std"], 0, 1)
        upper = np.clip(curve["tpr"] + curve["std"], 0, 1)
        ax.fill_between(
            curve["fpr"],
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.15,
        )
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate (unitless, log scale)")
    ax.set_ylabel("True-positive rate (unitless)")
    ax.set_title("SIMULATED PAIRS ROC curves (N=50; mean ± std over seeds)")
    ax.legend(loc="lower right")
    _configure_axes(ax)
    _save_figure(fig, path)


def _make_fig4(
    fusion: dict[tuple[int, str], dict[str, str]], path: Path
) -> None:
    methods = ("B0-timing", "B1-rules", "B4-siamese", "IM-Link-full")
    group_sizes = (10, 50, 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel, title in (
        (axes[0], "top1", "Top-1 peer accuracy (unitless)", "Top-1 vs. group size"),
        (
            axes[1],
            "tpr_fpr_0.01",
            "TPR at FPR = 1e-2 (unitless)",
            "TPR at FPR = 1e-2 vs. group size",
        ),
    ):
        for method in methods:
            rows = [fusion[(str(n), method)] for n in group_sizes]
            ax.errorbar(
                group_sizes,
                [_number(row, metric + "_mean") for row in rows],
                yerr=[_number(row, metric + "_std") for row in rows],
                marker="o",
                capsize=4,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_xticks(group_sizes)
        ax.set_xlabel("Group size N (sessions)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        _configure_axes(ax)
    fig.suptitle("SIMULATED PAIRS correlation performance")
    _save_figure(fig, path)


def _make_fig5(joint: dict[str, dict[str, str]], path: Path) -> None:
    metric_groups = (
        (
            "Application macro-F1",
            ("stage_a_application_macro_f1", "imlink_application_macro_f1"),
            ("Stage A alone", "IM-Link combined"),
        ),
        (
            "Peer top-1",
            ("b4_peer_top1", "imlink_peer_top1"),
            ("B4", "IM-Link"),
        ),
        (
            "Joint accuracy",
            (
                "b4_peer_plus_stage_a_joint_accuracy",
                "imlink_peer_plus_combined_app_joint_accuracy",
            ),
            ("B4 peer + Stage A", "IM-Link peer + combined app"),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    pair_colors = ("#1f77b4", "#d62728")
    for ax, (title, metric_names, labels) in zip(axes, metric_groups):
        values = [joint[(name,)] for name in metric_names]
        means = [_number(row, "mean") for row in values]
        stds = [_number(row, "std") for row in values]
        x = np.arange(2)
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=pair_colors,
            label="Mean ± std over seeds",
        )
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("Score (unitless)")
        ax.set_title(title)
        ax.set_ylim(0, 1.12)
        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.legend(loc="upper left")
        _configure_axes(ax)
    fig.suptitle("SIMULATED PAIRS joint evaluation (mean ± std over seeds)")
    _save_figure(fig, path)


def _make_fig6(coverage_rows: list[dict[str, str]], path: Path) -> None:
    methods = ("B4-siamese", "IM-Link-full")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, group_size in zip(axes, (50, 100)):
        for method in methods:
            rows = sorted(
                (
                    row
                    for row in coverage_rows
                    if int(row["N"]) == group_size and row["method"] == method
                ),
                key=lambda row: float(row["coverage"]),
            )
            if not rows:
                raise ValueError(
                    f"Saved coverage curve is missing N={group_size}, {method}"
                )
            coverage = np.asarray([float(row["coverage"]) for row in rows])
            accuracy = np.asarray(
                [float(row["accuracy_mean"]) for row in rows]
            )
            std = np.asarray([float(row["accuracy_std"]) for row in rows])
            ax.plot(
                coverage,
                accuracy,
                color=METHOD_COLORS[method],
                label=method,
            )
            ax.fill_between(
                coverage,
                np.clip(accuracy - std, 0, 1),
                np.clip(accuracy + std, 0, 1),
                color=METHOD_COLORS[method],
                alpha=0.15,
            )
        ax.set_title(f"N={group_size} sessions")
        ax.set_xlabel("Coverage (answered fraction)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend()
        _configure_axes(ax)
    axes[0].set_ylabel("Top-1 accuracy among answered sessions (unitless)")
    fig.suptitle("SIMULATED PAIRS abstention curves (mean ± std over seeds)")
    _save_figure(fig, path)


def _make_fig7(
    ablation: dict[tuple[str, str], dict[str, str]], path: Path
) -> None:
    variants = (
        "IM-Link-full",
        "IM-Link-no-consistency",
        "IM-Link-no-pairwise",
        "IM-Link-no-siamese",
        "IM-Link-no-timing",
    )
    labels = (
        "Full",
        "No consistency",
        "No pairwise",
        "No Siamese",
        "No timing",
    )
    y = np.arange(len(variants))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, group_size, color in (
        (-height / 2, 50, "#1f77b4"),
        (height / 2, 100, "#d62728"),
    ):
        rows = [ablation[(variant, str(group_size))] for variant in variants]
        ax.barh(
            y + offset,
            [_number(row, "top1_mean") for row in rows],
            height,
            xerr=[_number(row, "top1_std") for row in rows],
            capsize=4,
            color=color,
            label=f"N={group_size}",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Top-1 peer accuracy (unitless)")
    ax.set_ylabel("Fusion variant")
    ax.set_title("SIMULATED PAIRS fusion ablations (mean ± std over seeds)")
    ax.legend()
    _configure_axes(ax)
    _save_figure(fig, path)


def _make_fig8(
    defence: dict[tuple[str, str], dict[str, str]], path: Path
) -> None:
    padding_names = (
        "none",
        "padding_256",
        "padding_1024",
        "padding_4096",
        "combined_1024_0.5",
    )
    jitter_names = ("none", "jitter_0.5", "jitter_2.0", "combined_1024_0.5")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    left = axes[0]
    clean_padding = [defence[(name, "clean")] for name in padding_names]
    adaptive_padding = [defence[(name, "adaptive")] for name in padding_names]
    overhead = np.asarray(
        [_number(row, "overhead_pct_mean") for row in clean_padding]
    )
    overhead_std = np.asarray(
        [_number(row, "overhead_pct_std") for row in clean_padding]
    )
    order = np.argsort(overhead, kind="stable")
    sorted_names = [padding_names[index] for index in order]
    left.errorbar(
        overhead[order],
        [_number(clean_padding[i], "stage_a_application_macro_f1_mean") for i in order],
        xerr=overhead_std[order],
        yerr=[
            _number(clean_padding[i], "stage_a_application_macro_f1_std")
            for i in order
        ],
        marker="o",
        capsize=3,
        color="#2ca02c",
        label="Stage A application macro-F1",
    )
    left.errorbar(
        overhead[order],
        [_number(clean_padding[i], "imlink_peer_top1_mean") for i in order],
        xerr=overhead_std[order],
        yerr=[_number(clean_padding[i], "imlink_peer_top1_std") for i in order],
        fmt="o-",
        markersize=6,
        markerfacecolor="#d62728",
        capsize=3,
        color="#d62728",
        label="IM-Link clean peer top-1",
    )
    left.errorbar(
        overhead[order],
        [_number(adaptive_padding[i], "imlink_peer_top1_mean") for i in order],
        xerr=overhead_std[order],
        yerr=[
            _number(adaptive_padding[i], "imlink_peer_top1_std")
            for i in order
        ],
        fmt="s--",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="#9467bd",
        capsize=3,
        color="#9467bd",
        label="IM-Link adaptive peer top-1",
    )
    for index, name in enumerate(sorted_names):
        if name == "combined_1024_0.5":
            left.scatter(
                overhead[order][index],
                _number(clean_padding[order[index]], "imlink_peer_top1_mean"),
                marker="*",
                s=180,
                color="#d62728",
                edgecolor="black",
                zorder=5,
                label="Combined defence",
            )
    left.set_xlabel("Bandwidth overhead (%)")
    left.set_ylabel("Score (unitless)")
    left.set_title("Padding and combined defence")
    left.set_ylim(0, 1.05)
    left.legend(fontsize=8)
    _configure_axes(left)

    right = axes[1]
    clean_jitter = [defence[(name, "clean")] for name in jitter_names]
    adaptive_jitter = [defence[(name, "adaptive")] for name in jitter_names]
    delays = np.asarray(
        [_number(row, "mean_added_delay_s_mean") for row in clean_jitter]
    )
    delay_std = np.asarray(
        [_number(row, "mean_added_delay_s_std") for row in clean_jitter]
    )
    order = np.argsort(delays, kind="stable")
    sorted_names = [jitter_names[index] for index in order]
    right.errorbar(
        delays[order],
        [
            _number(clean_jitter[i], "b0_timing_top1_clean_reference_mean")
            for i in order
        ],
        xerr=delay_std[order],
        yerr=[
            _number(clean_jitter[i], "b0_timing_top1_clean_reference_std")
            for i in order
        ],
        marker="o",
        capsize=3,
        color=METHOD_COLORS["B0-timing"],
        label="B0 timing top-1",
    )
    right.errorbar(
        delays[order],
        [_number(clean_jitter[i], "imlink_peer_top1_mean") for i in order],
        xerr=delay_std[order],
        yerr=[_number(clean_jitter[i], "imlink_peer_top1_std") for i in order],
        fmt="o-",
        markersize=6,
        markerfacecolor=METHOD_COLORS["IM-Link-full"],
        capsize=3,
        color=METHOD_COLORS["IM-Link-full"],
        label="IM-Link clean top-1",
    )
    right.errorbar(
        delays[order],
        [_number(adaptive_jitter[i], "imlink_peer_top1_mean") for i in order],
        xerr=delay_std[order],
        yerr=[
            _number(adaptive_jitter[i], "imlink_peer_top1_std")
            for i in order
        ],
        fmt="s--",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="#9467bd",
        capsize=3,
        color="#9467bd",
        label="IM-Link adaptive top-1",
    )
    for index, name in enumerate(sorted_names):
        if name == "combined_1024_0.5":
            right.scatter(
                delays[order][index],
                _number(adaptive_jitter[order[index]], "imlink_peer_top1_mean"),
                marker="*",
                s=180,
                color="#d62728",
                edgecolor="black",
                zorder=5,
                label="Combined defence",
            )
    right.set_xlabel("Mean added delay (seconds)")
    right.set_ylabel("Top-1 peer accuracy (unitless)")
    right.set_title("Timing jitter and combined defence")
    right.set_ylim(0, 1.05)
    right.legend(fontsize=8)
    _configure_axes(right)
    fig.suptitle("SIMULATED PAIRS defence evaluation (mean ± std over seeds)")
    _save_figure(fig, path)


def _fmt_metric(row: dict[str, str], metric: str) -> str:
    return _mean_std(row, metric)


def _comparison_rows(
    activity_summary: dict[tuple[str, str], dict[str, str]],
    correlation: dict[tuple[str, str], dict[str, str]],
    fusion: dict[tuple[str, str], dict[str, str]],
    defence: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    activity_xgb = activity_summary[("B3-xgboost", "test")]
    b4_50 = fusion[("50", "B4-siamese")]
    b4_100 = fusion[("100", "B4-siamese")]
    imlink_50 = fusion[("50", "IM-Link-full")]
    imlink_100 = fusion[("100", "IM-Link-full")]
    b1_50 = correlation[("50", "B1-rules")]
    pad_clean = defence[("padding_1024", "clean")]
    pad_adaptive = defence[("padding_1024", "adaptive")]
    combined_clean = defence[("combined_1024_0.5", "clean")]
    combined_adaptive = defence[("combined_1024_0.5", "adaptive")]

    return [
        {
            "Study": "P3 Mehavilla et al. 2026",
            "Task": "Activity fingerprinting on IM",
            "Dataset and setting": "Reported per-platform activity classification",
            "Reported result (as stated in the paper)": (
                "F1 0.98 [TO VERIFY against the full paper] (WhatsApp) down to "
                "0.62 [TO VERIFY against the full paper] (X)"
            ),
            "Our comparable result": (
                "Stage A XGBoost application macro-F1, test: "
                f"{_fmt_metric(activity_xgb, 'macro_f1')}"
            ),
            "Directly comparable?": "No",
            "Why not / caveats": (
                "P3 classifies user activity per platform; IM-Link classifies "
                "the application."
            ),
        },
        {
            "Study": "P4 RECTor (Wu et al. 2026)",
            "Task": "Tor flow correlation",
            "Dataset and setting": "Reported mapping ratios 0.5 and 0.1",
            "Reported result (as stated in the paper)": (
                "85% [TO VERIFY against the full paper] TPR at 20% "
                "[TO VERIFY against the full paper] FPR (ratio 0.5 "
                "[TO VERIFY against the full paper]); 70% "
                "[TO VERIFY against the full paper] at ratio 0.1 "
                "[TO VERIFY against the full paper]"
            ),
            "Our comparable result": (
                f"IM-Link TPR@FPR=0.2: N=50 {_fmt_metric(imlink_50, 'tpr_fpr_0.2')}; "
                f"N=100 {_fmt_metric(imlink_100, 'tpr_fpr_0.2')}. "
                f"B4: N=50 {_fmt_metric(b4_50, 'tpr_fpr_0.2')}; "
                f"N=100 {_fmt_metric(b4_100, 'tpr_fpr_0.2')} "
                "(mapping ratio 1.0)"
            ),
            "Directly comparable?": "No",
            "Why not / caveats": (
                "Tor web traffic versus SIMULATED PAIRS IM traffic; every flow "
                "has a partner (mapping ratio 1.0), and this operating point "
                "is saturated."
            ),
        },
        {
            "Study": "P5 FlowCoPCL 2026",
            "Task": "Flow correlation",
            "Dataset and setting": "Tor; reported FPR=1e-5",
            "Reported result (as stated in the paper)": (
                "Over 99% [TO VERIFY against the full paper] TPR at FPR=1e-5 "
                "[TO VERIFY against the full paper]"
            ),
            "Our comparable result": (
                f"IM-Link TPR@FPR=1e-3: N=50 "
                f"{_fmt_metric(imlink_50, 'tpr_fpr_0.001')}; N=100 "
                f"{_fmt_metric(imlink_100, 'tpr_fpr_0.001')}. "
                f"B4: N=50 {_fmt_metric(b4_50, 'tpr_fpr_0.001')}; "
                f"N=100 {_fmt_metric(b4_100, 'tpr_fpr_0.001')}"
            ),
            "Directly comparable?": "No",
            "Why not / caveats": (
                "Tor traffic and a different operating point. The available "
                "negative-pair count is insufficient to estimate TPR at 1e-5 reliably."
            ),
        },
        {
            "Study": "P1 Li 2026",
            "Task": "Rule-based flow correlation",
            "Dataset and setting": "Manual inspection of correlation rules",
            "Reported result (as stated in the paper)": (
                "No accuracy reported [TO VERIFY against the full paper]"
            ),
            "Our comparable result": (
                f"Automated B1-rules, N=50: top-1 "
                f"{_fmt_metric(b1_50, 'top1')}; TPR@FPR=0.2 "
                f"{_fmt_metric(b1_50, 'tpr_fpr_0.2')}"
            ),
            "Directly comparable?": "Qualitative only",
            "Why not / caveats": (
                "Manual rule inspection cannot be compared quantitatively with "
                "an automated SIMULATED PAIRS evaluation."
            ),
        },
        {
            "Study": "P10 Securitas NSDI 2026",
            "Task": "Traffic analysis defence",
            "Dataset and setting": "Web/IoT/apps; Var-CNN with packet-level obfuscation",
            "Reported result (as stated in the paper)": (
                "6.21% [TO VERIFY against the full paper] bandwidth overhead; "
                "attack accuracy 99.40% [TO VERIFY against the full paper] to "
                "3.51% [TO VERIFY against the full paper]"
            ),
            "Our comparable result": (
                "Padding 1024: overhead "
                f"{_mean_std(pad_clean, 'overhead_pct')}%, IM-Link clean top-1 "
                f"{_mean_std(pad_clean, 'imlink_peer_top1')}, adaptive "
                f"{_mean_std(pad_adaptive, 'imlink_peer_top1')}. Combined "
                "1024 + 0.5s jitter: overhead "
                f"{_mean_std(combined_clean, 'overhead_pct')}%, added delay "
                f"{_mean_std(combined_clean, 'mean_added_delay_s')}s"
            ),
            "Directly comparable?": "No",
            "Why not / caveats": (
                "Different web/IoT/app traffic; packet-level obfuscation versus "
                "IM-Link's feature-level padding/jitter approximation."
            ),
        },
        {
            "Study": "P11 FRUGAL NDSS 2026",
            "Task": "Traffic analysis defence",
            "Dataset and setting": "Reported 30% overhead",
            "Reported result (as stated in the paper)": (
                "Attack success 2.68% [TO VERIFY against the full paper] at "
                "30% [TO VERIFY against the full paper] overhead"
            ),
            "Our comparable result": (
                "Padding 1024: overhead "
                f"{_mean_std(pad_clean, 'overhead_pct')}%, IM-Link clean top-1 "
                f"{_mean_std(pad_clean, 'imlink_peer_top1')}, adaptive "
                f"{_mean_std(pad_adaptive, 'imlink_peer_top1')}. Combined "
                "1024 + 0.5s jitter: overhead "
                f"{_mean_std(combined_clean, 'overhead_pct')}%, added delay "
                f"{_mean_std(combined_clean, 'mean_added_delay_s')}s"
            ),
            "Directly comparable?": "No",
            "Why not / caveats": (
                "Different traffic and attack metric; packet-level defence "
                "versus a feature-level approximation."
            ),
        },
    ]


def _write_comparison(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _print_comparison(rows: list[dict[str, str]]) -> None:
    print("SIMULATED PAIRS: paper comparison table")
    for row in rows:
        print("-" * 100)
        for column in COMPARISON_COLUMNS:
            print(f"{column}: {row[column]}")


def run(project_dir: Path, output_dir: Path) -> list[str]:
    activity_rows = _read_csv(
        project_dir / "results" / "activity" / "activity_summary.csv"
    )
    correlation_rows = _read_csv(
        project_dir / "results" / "correlation" / "table_correlation.csv"
    )
    fusion_rows = _read_csv(
        project_dir / "results" / "fusion" / "table_fusion.csv"
    )
    ablation_rows = _read_csv(
        project_dir / "results" / "fusion" / "table_ablation.csv"
    )
    joint_rows = _read_csv(project_dir / "results" / "fusion" / "table_joint.csv")
    fusion_roc_path = project_dir / "results" / "fusion" / "roc_fusion.csv"
    correlation_roc_path = (
        project_dir / "results" / "correlation" / "roc_correlation.csv"
    )
    abstain_path = project_dir / "results" / "step7" / "table_abstain.csv"
    coverage_path = project_dir / "results" / "step7" / "coverage_curves.csv"
    defence_rows = _read_csv(
        project_dir / "results" / "step7" / "table_defence.csv"
    )

    activity = _index(activity_rows, "model", "split")
    correlation = _index(correlation_rows, "N", "method")
    fusion = _index(fusion_rows, "N", "method")
    ablation = _index(ablation_rows, "variant", "N")
    joint = _index(joint_rows, "metric")
    defence = _index(defence_rows, "defence", "attacker")
    coverage_rows = _read_csv(coverage_path)
    _read_csv(abstain_path)
    fusion_roc = _load_roc(fusion_roc_path)
    correlation_roc = _load_roc(correlation_roc_path)
    for key in fusion_roc.keys() & correlation_roc.keys():
        if not (
            np.array_equal(fusion_roc[key]["fpr"], correlation_roc[key]["fpr"])
            and np.allclose(fusion_roc[key]["tpr"], correlation_roc[key]["tpr"])
        ):
            raise ValueError(
                f"Step 5 and Step 6 saved ROC inputs disagree for {key}"
            )

    required_roc_keys = {
        (50, method)
        for method in ("B0-timing", "B1-rules", "B4-siamese", "IM-Link-full")
    }
    missing_roc = sorted(required_roc_keys - fusion_roc.keys())
    if missing_roc:
        raise ValueError(f"Saved IM-Link ROC data is missing entries: {missing_roc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _make_fig1(activity, figures_dir / FIGURE_FILES[0])
    _make_fig2(project_dir, figures_dir / FIGURE_FILES[1])
    _make_fig3(fusion_roc, figures_dir / FIGURE_FILES[2])
    _make_fig4(fusion, figures_dir / FIGURE_FILES[3])
    _make_fig5(joint, figures_dir / FIGURE_FILES[4])
    _make_fig6(coverage_rows, figures_dir / FIGURE_FILES[5])
    _make_fig7(ablation, figures_dir / FIGURE_FILES[6])
    _make_fig8(defence, figures_dir / FIGURE_FILES[7])

    comparison_path = output_dir / "table_comparison.csv"
    comparison = _read_csv(comparison_path)
    _print_comparison(comparison)
    print("SIMULATED PAIRS: figures created")
    for filename in FIGURE_FILES:
        print(f"  {figures_dir / filename}")
    print(f"  {comparison_path}")
    missing_inputs: list[str] = []
    print(
        "Figure inputs not saved as files: row-normalized confusion matrix for "
        "the saved Stage A model. Computed here with its predict_proba only; "
        "no model was retrained."
    )
    print("SIMULATED PAIRS: no other required figure inputs were missing.")
    print("SIMULATED PAIRS: no model was retrained.")
    return [str(figures_dir / filename) for filename in FIGURE_FILES]


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create result figures and a paper-comparison table from saved outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "step8",
    )
    args = parser.parse_args()
    run(project_dir, args.output_dir)


if __name__ == "__main__":
    main()
