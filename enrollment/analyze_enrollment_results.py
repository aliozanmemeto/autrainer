# analyze_enrollment_results.py
"""Analyze and visualize enrollment personalization results from autrainer.

Computes UAR per speaker and compares across baseline, neutral, emotional, and all enrollment modes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def compute_uar(targets: np.ndarray, predictions: np.ndarray, classes: List[str]) -> float:
    """Compute Unweighted Average Recall."""
    recalls = []
    for cls in classes:
        mask = targets == cls
        if mask.sum() == 0:
            continue
        recall = (predictions[mask] == cls).sum() / mask.sum()
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def load_predictions(results_dir: Path, corpus_labels_path: Path) -> pd.DataFrame:
    """Load predictions and merge with corpus labels."""
    preds_path = results_dir / "test_results.csv"
    preds = pd.read_csv(preds_path)

    # Load corpus labels
    labels_all = pd.read_csv(
        corpus_labels_path,
        sep=r"\s+",
        header=None,
        names=["utt_id", "true_label", "agreement"],
    )

    # Get test split (Mont school)
    labels_all["school"] = labels_all["utt_id"].apply(lambda x: x.split("_")[0])
    labels_all["speaker"] = labels_all["utt_id"].apply(lambda x: x.split("_")[1])
    labels_test = labels_all[labels_all["school"] == "Mont"].reset_index(drop=True)

    # Merge
    N = len(preds)
    if len(labels_test) != N:
        print(f"Warning: predictions ({N}) != test labels ({len(labels_test)})")
        labels_test = labels_test.iloc[:N].reset_index(drop=True)

    df = pd.concat([preds.reset_index(drop=True), labels_test.reset_index(drop=True)], axis=1)
    df["correct"] = df["predictions"] == df["true_label"]

    return df


def compute_speaker_uar(df: pd.DataFrame, classes: List[str]) -> Dict[str, float]:
    """Compute UAR per speaker."""
    speaker_uar = {}
    for speaker in df["speaker"].unique():
        df_spk = df[df["speaker"] == speaker]
        uar = compute_uar(
            df_spk["true_label"].values,
            df_spk["predictions"].values,
            classes
        )
        speaker_uar[speaker] = uar
    return speaker_uar


def compute_speaker_stats(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    """Compute detailed stats per speaker."""
    records = []
    for speaker in df["speaker"].unique():
        df_spk = df[df["speaker"] == speaker]

        record = {
            "speaker": speaker,
            "n_samples": len(df_spk),
            "uar": compute_uar(df_spk["true_label"].values, df_spk["predictions"].values, classes),
            "accuracy": df_spk["correct"].mean(),
        }

        # Per-class recall
        for cls in classes:
            mask = df_spk["true_label"] == cls
            if mask.sum() > 0:
                record[f"recall_{cls}"] = (df_spk.loc[mask, "predictions"] == cls).sum() / mask.sum()
                record[f"n_{cls}"] = mask.sum()
            else:
                record[f"recall_{cls}"] = np.nan
                record[f"n_{cls}"] = 0

        records.append(record)

    return pd.DataFrame(records).sort_values("uar", ascending=True)


# ============== MAIN ANALYSIS ==============

def main():
    # Paths
    base_results = Path("/data/chi-gpu1/perser/AIBO/results/aibo-2cl")
    corpus_labels_path = Path(
        "/data/chi-gpu1/perser/AIBO/labels/IS2009EmotionChallenge/"
        "chunk_labels_2cl_corpus.txt"
    )

    # Model result directories
    models = {
        "baseline": base_results / "aibo-2cl-w2v2-personalization-baseline/training/EnrollmentAIBO-2cl_Wav2Vec2Enrollment_Adam_0.0001_4_epoch_10_None_None_1/_test",
        "neutral": base_results / "aibo-2cl-w2v2-personalization-neutral/training/EnrollmentAIBO-2cl_Wav2Vec2Enrollment_Adam_0.0001_4_epoch_10_None_None_1/_test",
        "emotional": base_results / "aibo-2cl-w2v2-personalization-emotional/training/EnrollmentAIBO-2cl_Wav2Vec2Enrollment_Adam_0.0001_4_epoch_10_None_None_1/_test",
        "all": base_results / "aibo-2cl-w2v2-personalization-all/training/EnrollmentAIBO-2cl_Wav2Vec2Enrollment_Adam_0.0001_4_epoch_10_None_None_1/_test",
    }

    classes = ["IDL", "NEG"]

    # Load all predictions
    print("Loading predictions...")
    dfs = {}
    for name, path in models.items():
        if path.exists():
            dfs[name] = load_predictions(path, corpus_labels_path)
            print(f"  {name}: {len(dfs[name])} samples")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    # Compute overall UAR per model
    print("\n=== Overall UAR ===")
    overall_uar = {}
    for name, df in dfs.items():
        uar = compute_uar(df["true_label"].values, df["predictions"].values, classes)
        overall_uar[name] = uar
        print(f"  {name}: {uar:.4f}")

    # Compute speaker UAR for each model
    print("\n=== Computing per-speaker UAR ===")
    speaker_uars = {}
    for name, df in dfs.items():
        speaker_uars[name] = compute_speaker_uar(df, classes)

    # Create comparison DataFrame
    speakers = list(speaker_uars["baseline"].keys()) if "baseline" in speaker_uars else []
    comparison_data = []
    for spk in speakers:
        row = {"speaker": spk}
        for model_name in dfs.keys():
            row[model_name] = speaker_uars[model_name].get(spk, np.nan)
        comparison_data.append(row)

    df_compare = pd.DataFrame(comparison_data)

    # Compute improvements over baseline
    if "baseline" in dfs:
        for model_name in ["neutral", "emotional", "all"]:
            if model_name in dfs:
                df_compare[f"{model_name}_delta"] = df_compare[model_name] - df_compare["baseline"]

    # Sort by baseline UAR
    df_compare = df_compare.sort_values("baseline", ascending=True)

    print("\n=== Per-Speaker UAR Comparison ===")
    print(df_compare.to_string())

    # ============== VISUALIZATIONS ==============

    fig_dir = base_results / "visualizations"
    fig_dir.mkdir(exist_ok=True)

    # 1. UAR per speaker - all models comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df_compare))
    width = 0.2

    colors = {"baseline": "#1f77b4", "neutral": "#ff7f0e", "emotional": "#2ca02c", "all": "#d62728"}

    for i, model_name in enumerate(["baseline", "neutral", "emotional", "all"]):
        if model_name in df_compare.columns:
            ax.bar(x + i*width, df_compare[model_name], width,
                   label=model_name, color=colors[model_name], alpha=0.8)

    ax.set_xlabel("Speaker (sorted by baseline UAR)")
    ax.set_ylabel("UAR")
    ax.set_title("UAR per Speaker: Enrollment Methods Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_compare["speaker"], rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")

    plt.tight_layout()
    plt.savefig(fig_dir / "uar_per_speaker_comparison.png", dpi=150)
    plt.show()

    # 2. UAR Improvement over baseline
    if "baseline" in dfs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, model_name in enumerate(["neutral", "emotional", "all"]):
            if f"{model_name}_delta" in df_compare.columns:
                deltas = df_compare[f"{model_name}_delta"].values
                colors_bar = ["green" if d > 0 else "red" for d in deltas]

                axes[i].bar(range(len(deltas)), deltas, color=colors_bar, alpha=0.7)
                axes[i].axhline(y=0, color="black", linewidth=0.5)
                axes[i].axhline(y=deltas.mean(), color="blue", linestyle="--",
                               label=f"Mean: {deltas.mean():+.3f}")
                axes[i].set_xlabel("Speaker")
                axes[i].set_ylabel("UAR Change")
                axes[i].set_title(f"{model_name.capitalize()} vs Baseline")
                axes[i].legend()

                # Count improvements
                n_improved = (deltas > 0).sum()
                n_worse = (deltas < 0).sum()
                axes[i].text(0.02, 0.98, f"↑{n_improved} ↓{n_worse}",
                            transform=axes[i].transAxes, va="top", fontsize=10)

        plt.tight_layout()
        plt.savefig(fig_dir / "uar_improvement_over_baseline.png", dpi=150)
        plt.show()

    # 3. Scatter: Baseline UAR vs Improvement
    if "baseline" in dfs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, model_name in enumerate(["neutral", "emotional", "all"]):
            if f"{model_name}_delta" in df_compare.columns:
                x_vals = df_compare["baseline"].values
                y_vals = df_compare[f"{model_name}_delta"].values

                axes[i].scatter(x_vals, y_vals, alpha=0.7, s=60)
                axes[i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

                # Fit trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                axes[i].plot(x_line, p(x_line), "r--", alpha=0.5, label=f"trend")

                # Correlation
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                axes[i].set_xlabel("Baseline UAR")
                axes[i].set_ylabel("UAR Improvement")
                axes[i].set_title(f"{model_name.capitalize()} (r={corr:.3f})")
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(fig_dir / "baseline_vs_improvement.png", dpi=150)
        plt.show()

    # 4. Distribution of UAR across speakers
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [m for m in ["baseline", "neutral", "emotional", "all"] if m in df_compare.columns]
    data_for_box = [df_compare[m].dropna().values for m in model_names]

    bp = ax.boxplot(data_for_box, labels=model_names, patch_artist=True)

    for patch, model in zip(bp["boxes"], model_names):
        patch.set_facecolor(colors.get(model, "gray"))
        patch.set_alpha(0.7)

    ax.set_ylabel("UAR")
    ax.set_title("Distribution of Per-Speaker UAR")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Add mean markers
    for i, model in enumerate(model_names):
        mean_val = df_compare[model].mean()
        ax.scatter([i+1], [mean_val], marker="D", color="black", s=50, zorder=3)
        ax.text(i+1.1, mean_val, f"{mean_val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "uar_distribution_boxplot.png", dpi=150)
    plt.show()

    # 5. Heatmap: Speaker x Model
    fig, ax = plt.subplots(figsize=(8, 12))

    heatmap_data = df_compare.set_index("speaker")[model_names]

    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0.5, ax=ax, cbar_kws={"label": "UAR"})
    ax.set_title("UAR per Speaker and Model")
    ax.set_xlabel("Enrollment Method")

    plt.tight_layout()
    plt.savefig(fig_dir / "uar_heatmap_speaker_model.png", dpi=150)
    plt.show()

    # 6. Summary statistics
    print("\n=== Summary Statistics ===")
    summary = pd.DataFrame({
        "Overall UAR": overall_uar,
        "Mean Speaker UAR": {m: df_compare[m].mean() for m in model_names},
        "Median Speaker UAR": {m: df_compare[m].median() for m in model_names},
        "Std Speaker UAR": {m: df_compare[m].std() for m in model_names},
        "Min Speaker UAR": {m: df_compare[m].min() for m in model_names},
        "Max Speaker UAR": {m: df_compare[m].max() for m in model_names},
    }).T
    print(summary.to_string())

    # Save results
    df_compare.to_csv(fig_dir / "speaker_uar_comparison.csv", index=False)
    summary.to_csv(fig_dir / "summary_statistics.csv")

    print(f"\nVisualizations saved to: {fig_dir}")

    return df_compare, summary


if __name__ == "__main__":
    df_compare, summary = main()
