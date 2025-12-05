# visualization.py - Attention visualization utilities for enrollment models
"""Visualization tools for enrollment-based personalization models."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class EnrollmentVisualizer:
    """Visualize attention patterns from enrollment models."""

    def __init__(
        self,
        model,
        class_names: List[str],
        neutral_class: str,
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            model: Wav2Vec2Enrollment model
            class_names: List of class names (e.g., ["IDL", "NEG"] for 2cl)
            neutral_class: Name of neutral class (e.g., "IDL")
            save_dir: Directory to save figures
        """
        self.model = model
        self.class_names = class_names
        self.neutral_class = neutral_class
        self.emotional_classes = [c for c in class_names if c != neutral_class]
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected attention weights
        self.attention_data = []

    def get_enrollment_labels(self) -> List[str]:
        """Get labels for enrollment positions based on enrollment mode."""
        mode = self.model.enrollment
        if mode == "neutral":
            return [self.neutral_class]
        elif mode == "emotional":
            return self.emotional_classes
        elif mode == "all":
            return [self.neutral_class] + self.emotional_classes
        return []

    def collect_attention(
        self,
        attention_weights: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        speaker_ids: Optional[List[str]] = None,
    ):
        """Collect attention weights for later visualization.

        Args:
            attention_weights: [B, C] attention weights from model
            predictions: [B] predicted class indices
            targets: [B] ground truth class indices
            speaker_ids: Optional list of speaker IDs
        """
        B = attention_weights.shape[0]
        for i in range(B):
            self.attention_data.append({
                "attention": attention_weights[i].cpu().numpy(),
                "prediction": predictions[i].item(),
                "target": targets[i].item(),
                "correct": predictions[i].item() == targets[i].item(),
                "speaker": speaker_ids[i] if speaker_ids else None,
            })

    def plot_attention_heatmap(
        self,
        title: str = "Attention Weights by Predicted Class",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot heatmap of average attention per predicted class.

        Returns:
            matplotlib Figure
        """
        if not self.attention_data:
            raise ValueError("No attention data collected. Run collect_attention first.")

        enrollment_labels = self.get_enrollment_labels()
        n_enrollment = len(enrollment_labels)
        n_classes = len(self.class_names)

        # Aggregate attention by predicted class
        attention_matrix = np.zeros((n_classes, n_enrollment))
        counts = np.zeros(n_classes)

        for item in self.attention_data:
            pred = item["prediction"]
            attention_matrix[pred] += item["attention"]
            counts[pred] += 1

        # Normalize
        for i in range(n_classes):
            if counts[i] > 0:
                attention_matrix[i] /= counts[i]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attention_matrix,
            annot=True,
            fmt=".3f",
            xticklabels=enrollment_labels,
            yticklabels=self.class_names,
            cmap="Blues",
            ax=ax,
        )
        ax.set_xlabel("Enrollment Utterance")
        ax.set_ylabel("Predicted Class")
        ax.set_title(title)

        if self.save_dir:
            fig.savefig(self.save_dir / "attention_by_predicted_class.png", dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_correct_vs_incorrect(
        self,
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """Compare attention patterns for correct vs incorrect predictions."""
        if not self.attention_data:
            raise ValueError("No attention data collected.")

        enrollment_labels = self.get_enrollment_labels()

        correct = [d["attention"] for d in self.attention_data if d["correct"]]
        incorrect = [d["attention"] for d in self.attention_data if not d["correct"]]

        if not correct or not incorrect:
            print("Warning: Need both correct and incorrect predictions for comparison")
            return None

        correct_mean = np.mean(correct, axis=0)
        incorrect_mean = np.mean(incorrect, axis=0)
        correct_std = np.std(correct, axis=0)
        incorrect_std = np.std(incorrect, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Bar plot comparison
        x = np.arange(len(enrollment_labels))
        width = 0.35

        axes[0].bar(x - width/2, correct_mean, width, yerr=correct_std,
                    label=f'Correct (n={len(correct)})', capsize=3)
        axes[0].bar(x + width/2, incorrect_mean, width, yerr=incorrect_std,
                    label=f'Incorrect (n={len(incorrect)})', capsize=3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(enrollment_labels)
        axes[0].set_ylabel("Attention Weight")
        axes[0].set_title("Attention: Correct vs Incorrect")
        axes[0].legend()

        # Difference plot
        diff = correct_mean - incorrect_mean
        colors = ['green' if d > 0 else 'red' for d in diff]
        axes[1].bar(enrollment_labels, diff, color=colors)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel("Attention Difference (Correct - Incorrect)")
        axes[1].set_title("Attention Difference")

        plt.tight_layout()

        if self.save_dir:
            fig.savefig(self.save_dir / "attention_correct_vs_incorrect.png", dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_distribution(
        self,
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """Plot distribution of attention weights per enrollment position."""
        if not self.attention_data:
            raise ValueError("No attention data collected.")

        enrollment_labels = self.get_enrollment_labels()
        n_enrollment = len(enrollment_labels)

        # Extract attention per position
        attention_per_pos = [[] for _ in range(n_enrollment)]
        for item in self.attention_data:
            for i, w in enumerate(item["attention"]):
                attention_per_pos[i].append(w)

        fig, axes = plt.subplots(1, n_enrollment, figsize=figsize)
        if n_enrollment == 1:
            axes = [axes]

        for i, (label, weights) in enumerate(zip(enrollment_labels, attention_per_pos)):
            axes[i].hist(weights, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel("Attention Weight")
            axes[i].set_ylabel("Count")
            axes[i].set_title(f"{label}\nμ={np.mean(weights):.3f}")
            axes[i].axvline(np.mean(weights), color='red', linestyle='--', label='Mean')

        plt.tight_layout()

        if self.save_dir:
            fig.savefig(self.save_dir / "attention_distribution.png", dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_by_speaker(
        self,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """Plot attention patterns per speaker."""
        if not self.attention_data or self.attention_data[0]["speaker"] is None:
            raise ValueError("No speaker data available.")

        enrollment_labels = self.get_enrollment_labels()

        # Aggregate by speaker
        speaker_attention = {}
        speaker_accuracy = {}

        for item in self.attention_data:
            spk = item["speaker"]
            if spk not in speaker_attention:
                speaker_attention[spk] = []
                speaker_accuracy[spk] = []
            speaker_attention[spk].append(item["attention"])
            speaker_accuracy[spk].append(item["correct"])

        # Average per speaker
        speakers = list(speaker_attention.keys())[:top_n]
        attention_matrix = np.array([np.mean(speaker_attention[s], axis=0) for s in speakers])
        accuracies = [np.mean(speaker_accuracy[s]) for s in speakers]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Heatmap
        sns.heatmap(
            attention_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=enrollment_labels,
            yticklabels=[f"{s} ({acc:.0%})" for s, acc in zip(speakers, accuracies)],
            cmap="Blues",
            ax=axes[0],
        )
        axes[0].set_xlabel("Enrollment Utterance")
        axes[0].set_ylabel("Speaker (Accuracy)")
        axes[0].set_title("Attention by Speaker")

        # Accuracy vs attention correlation
        for i, label in enumerate(enrollment_labels):
            att_vals = [np.mean(speaker_attention[s], axis=0)[i] for s in speakers]
            axes[1].scatter(att_vals, accuracies, label=label, alpha=0.7)

        axes[1].set_xlabel("Mean Attention Weight")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy vs Attention")
        axes[1].legend()

        plt.tight_layout()

        if self.save_dir:
            fig.savefig(self.save_dir / "attention_by_speaker.png", dpi=150, bbox_inches="tight")

        return fig

    def clear(self):
        """Clear collected attention data."""
        self.attention_data = []


def extract_attention_from_batch(model, batch) -> torch.Tensor:
    """Extract attention weights from a forward pass.

    Args:
        model: Wav2Vec2Enrollment model
        batch: EnrollmentDataBatch

    Returns:
        Attention weights tensor [B, C]
    """
    model.eval()
    with torch.no_grad():
        _ = model(
            features=batch.features,
            enroll_neutral=batch.enroll_neutral,
            enroll_emotional=batch.enroll_emotional,
        )

    if hasattr(model, '_last_attention_weights'):
        return model._last_attention_weights
    return None


# Example usage script
def create_attention_report(
    model,
    dataloader,
    class_names: List[str],
    neutral_class: str,
    save_dir: str,
    device: str = "cuda",
):
    """Generate full attention visualization report.

    Args:
        model: Trained Wav2Vec2Enrollment model
        dataloader: DataLoader with EnrollmentDataBatch items
        class_names: List of emotion class names
        neutral_class: Name of neutral class
        save_dir: Directory to save visualizations
        device: Device to run on
    """
    model = model.to(device)
    model.eval()

    visualizer = EnrollmentVisualizer(
        model=model,
        class_names=class_names,
        neutral_class=neutral_class,
        save_dir=save_dir,
    )

    print("Collecting attention weights...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            logits = model(
                features=batch.features,
                enroll_neutral=batch.enroll_neutral,
                enroll_emotional=batch.enroll_emotional,
            )

            predictions = logits.argmax(dim=-1)

            # Collect attention
            if hasattr(model, '_last_attention_weights') and model._last_attention_weights is not None:
                visualizer.collect_attention(
                    attention_weights=model._last_attention_weights,
                    predictions=predictions,
                    targets=batch.target,
                )

    print(f"Collected {len(visualizer.attention_data)} samples")

    # Generate plots
    print("Generating visualizations...")
    visualizer.plot_attention_heatmap()
    visualizer.plot_attention_correct_vs_incorrect()
    visualizer.plot_attention_distribution()

    print(f"Saved visualizations to {save_dir}")

    return visualizer
