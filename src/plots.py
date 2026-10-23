"""Matplotlib figures generated from evaluated results (never from illustrative numbers)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

MODEL_COLOR = "#2a6fdb"
RANDOM_COLOR = "#8a8a8a"
COMPARATOR_COLORS = ["#e07b39", "#3aa676", "#a05cc9"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def classification_figure(y, scores, summary, path: Path, comparators: dict | None = None):
    """ROC, precision-recall, cumulative hit recovery and EF@k vs random on the test split."""
    comparators = comparators or {}
    y = np.asarray(y)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, color=MODEL_COLOR, lw=2, label=f"Random Forest (AUC {summary['roc_auc']:.3f})")
    for (name, s), c in zip(comparators.items(), COMPARATOR_COLORS):
        f, t, _ = roc_curve(y, s)
        ax.plot(f, t, color=c, lw=1.2, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=RANDOM_COLOR, label="Random ranking")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (held-out test)")
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)

    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y, scores)
    ax.plot(rec, prec, color=MODEL_COLOR, lw=2, label=f"Random Forest (AP {summary['average_precision']:.3f})")
    for (name, s), c in zip(comparators.items(), COMPARATOR_COLORS):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=c, lw=1.2, label=name)
    ax.axhline(y.mean(), ls="--", color=RANDOM_COLOR, label=f"Prevalence ({y.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (held-out test)")
    ax.legend(fontsize=8, loc="upper right")
    _style(ax)

    ax = axes[1, 0]
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    cum_hits = np.cumsum(y[order])
    frac = np.arange(1, len(y) + 1) / len(y)
    ax.plot(frac * 100, cum_hits, color=MODEL_COLOR, lw=2, label="Random Forest ranking")
    ax.plot(frac * 100, frac * y.sum(), ls="--", color=RANDOM_COLOR, label="Expected random selection")
    ax.axvline(5, color="black", lw=0.8, ls=":")
    ax.set(xlim=(0, 25), ylim=(0, max(cum_hits[int(0.25 * len(y))], 1) * 1.1),
           xlabel="% of test pool screened", ylabel="Assay actives recovered",
           title="Hit recovery vs. screening budget")
    ax.legend(fontsize=8, loc="upper left")
    _style(ax)

    ax = axes[1, 1]
    labels, model_hits, random_hits = [], [], []
    for key in ("enrichment@1%", "enrichment@5%", "enrichment@10%"):
        e = summary[key]
        labels.append(f"Top {key.split('@')[1]}\n(k={e['n_selected']})")
        model_hits.append(e["hits"])
        random_hits.append(e["expected_random_hits"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, random_hits, 0.4, color=RANDOM_COLOR, label="Expected random hits")
    bars = ax.bar(x + 0.2, model_hits, 0.4, color=MODEL_COLOR, label="Model hits")
    for b, key in zip(bars, ("enrichment@1%", "enrichment@5%", "enrichment@10%")):
        ax.annotate(f"EF {summary[key]['enrichment_factor']:.2f}x", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Assay actives in selection", title="Hits at fixed budgets: model vs. random")
    ax.legend(fontsize=8, loc="upper left")
    _style(ax)

    fig.suptitle("Experiment A - MoleculeNet HIV, scaffold-split held-out test set", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validation_figure(trials_df, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    df = trials_df.sort_values("valid_average_precision")
    colors = [MODEL_COLOR if c else RANDOM_COLOR for c in df["candidate_for_selection"]]
    ax.barh(df["model"], df["valid_average_precision"], color=colors)
    for i, (ap, ef) in enumerate(zip(df["valid_average_precision"], df["valid_ef@5%"])):
        ax.text(ap, i, f"  AP {ap:.3f} | EF@5% {ef:.2f}", va="center", fontsize=8)
    ax.set(xlabel="Validation average precision",
           title="Validation model selection (blue = eligible RF candidates)")
    ax.set_xlim(0, df["valid_average_precision"].max() * 1.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def similarity_figure(by_bin: list, path: Path):
    rows = [r for r in by_bin if "average_precision" in r]
    if not rows:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows))
    ax1.bar(x - 0.2, [r["average_precision"] for r in rows], 0.4, color=MODEL_COLOR, label="Average precision")
    ax1.bar(x + 0.2, [r["prevalence"] for r in rows], 0.4, color=RANDOM_COLOR, label="Active prevalence (random AP)")
    ax1.set_xticks(x, [f"{r['bin']}\nn={r['n']}, act={r['actives']}" for r in rows], fontsize=8)
    ax1.set(xlabel="Max Tanimoto similarity of test molecule to training pool", ylabel="Score",
            title="Test performance by similarity to training chemistry")
    ax1.legend(fontsize=8)
    _style(ax1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regression_figure(y_true, y_pred, summary, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.7, color=MODEL_COLOR, edgecolor="none")
    lo = min(np.min(y_true), np.min(y_pred)) - 0.5
    hi = max(np.max(y_true), np.max(y_pred)) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color=RANDOM_COLOR, label="y = x")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured log10 solubility (mol/L)",
           ylabel="Predicted log10 solubility (mol/L)", title=title)
    ax.text(0.03, 0.97, f"MAE {summary['mae']:.3f}\nRMSE {summary['rmse']:.3f}\nR$^2$ {summary['r2']:.3f}\nn={summary['n']}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"))
    ax.legend(loc="lower right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regression_comparison_figure(rows: list, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["valid_mae"] for r in rows], 0.4, color=RANDOM_COLOR, label="Validation MAE")
    ax.bar(x + 0.2, [r["test_mae"] for r in rows], 0.4, color=MODEL_COLOR, label="Test MAE (refit on train+valid)")
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="MAE, log10(mol/L)", title="ESOL solubility: all candidate models")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness_figure(df, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (scheme, group) in enumerate(df.groupby("split_scheme", sort=False)):
        ax.scatter(np.full(len(group), i), group["ef@5%"], color=COMPARATOR_COLORS[i % 3], s=40, zorder=3)
        ax.hlines(group["ef@5%"].mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
    ax.axhline(1, ls="--", color=RANDOM_COLOR, label="Random expectation (EF = 1)")
    ax.set_xticks(range(df["split_scheme"].nunique()), list(df["split_scheme"].unique()), fontsize=8)
    ax.set(ylabel="Test EF@5%", title="Sensitivity of EF@5% to the data partition")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
