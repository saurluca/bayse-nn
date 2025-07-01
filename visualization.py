import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(results, save_path="model_comparison.png"):
    """Plot comparison of all models across different metrics"""
    model_names = [result["name"] for result in results]
    accuracies = [result["accuracy"] for result in results]
    losses = [result["loss"] for result in results]
    causal_scores = [result["causal_score"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy comparison
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, accuracies, color="skyblue", alpha=0.7)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Add accuracy values on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    # Loss comparison
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, losses, color="lightcoral", alpha=0.7)
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Model Loss Comparison")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # Add loss values on bars
    for bar, loss in zip(bars2, losses):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{loss:.3f}",
            ha="center",
            va="bottom",
        )

    # Causal discovery score comparison
    ax3 = axes[2]
    colors = ["green" if score > 0 else "red" for score in causal_scores]
    bars3 = ax3.bar(model_names, causal_scores, color=colors, alpha=0.7)
    ax3.set_ylabel("Causal Discovery Score")
    ax3.set_title("Causal Discovery Performance")
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    # Add score values on bars
    for bar, score in zip(bars3, causal_scores):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if score > 0 else -0.03),
            f"{score:.3f}",
            ha="center",
            va="bottom" if score > 0 else "top",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance_comparison(
    results, save_path="feature_importance_comparison.png"
):
    """Plot feature importance comparison across models"""
    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(feature_names))
    width = 0.8 / len(results)  # Width of bars

    for i, result in enumerate(results):
        if result["feature_importance"]:
            offset = (i - len(results) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                result["feature_importance"],
                width,
                label=result["name"],
                alpha=0.7,
            )

            # Add value labels on bars
            for bar, importance in zip(bars, result["feature_importance"]):
                if bar.get_height() != 0:  # Only label non-zero bars
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if bar.get_height() > 0 else -0.02),
                        f"{importance:.2f}",
                        ha="center",
                        va="bottom" if bar.get_height() > 0 else "top",
                        fontsize=8,
                    )

    ax.set_xlabel("Features")
    ax.set_ylabel("Importance (Cancer Prob Drop)")
    ax.set_title("Feature Importance Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add expected behavior annotation
    ax.text(
        0.02,
        0.98,
        "Expected: Tumor & Genetic HIGH, others LOW",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_progress(training_history, save_path="training_progress.png"):
    """Plot training progress for models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy over time
    for model_name, history in training_history.items():
        if "accuracy" in history:
            epochs = range(len(history["accuracy"]))
            ax1.plot(epochs, history["accuracy"], marker="o", label=model_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training Accuracy Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Loss over time
    for model_name, history in training_history.items():
        if "loss" in history:
            epochs = range(len(history["loss"]))
            ax2.plot(epochs, history["loss"], marker="o", label=model_name)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss Progress")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_robustness_analysis(robustness_results, save_path="robustness_analysis.png"):
    """Plot intervention robustness analysis"""
    model_names = list(robustness_results.keys())
    causal_sensitivity = [
        robustness_results[name]["causal_sensitivity"] for name in model_names
    ]
    confound_robustness = [
        robustness_results[name]["confound_robustness"] for name in model_names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Causal sensitivity
    bars1 = ax1.bar(model_names, causal_sensitivity, color="lightgreen", alpha=0.7)
    ax1.set_ylabel("Causal Sensitivity")
    ax1.set_title("Sensitivity to Causal Interventions\n(Higher is Better)")
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    for bar, sens in zip(bars1, causal_sensitivity):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{sens:.3f}",
            ha="center",
            va="bottom",
        )

    # Confound robustness
    bars2 = ax2.bar(model_names, confound_robustness, color="lightblue", alpha=0.7)
    ax2.set_ylabel("Confound Robustness")
    ax2.set_title("Robustness to Confound Removal\n(Higher is Better)")
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    for bar, rob in zip(bars2, confound_robustness):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rob:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_causal_evidence_heatmap(results, save_path="causal_evidence_heatmap.png"):
    """Plot heatmap of causal evidence across models and features"""
    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]

    # Collect causal evidence data
    evidence_data = []
    model_names = []

    for result in results:
        if result["causal_evidence"] is not None:
            evidence_data.append(result["causal_evidence"])
            model_names.append(result["name"])

    if not evidence_data:
        print("No causal evidence data found for heatmap")
        return

    evidence_matrix = np.array(evidence_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(evidence_matrix, cmap="RdYlBu_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Causal Evidence")

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(feature_names)):
            text = ax.text(
                j,
                i,
                f"{evidence_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_title("Causal Evidence Across Models and Features")
    ax.set_xlabel("Features")
    ax.set_ylabel("Models")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def create_summary_report(
    results, robustness_results=None, save_path="model_summary_report.png"
):
    """Create a comprehensive summary report"""
    fig = plt.figure(figsize=(20, 12))

    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    model_names = [result["name"] for result in results]

    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [result["accuracy"] for result in results]
    ax1.bar(range(len(model_names)), accuracies, color="skyblue", alpha=0.7)
    ax1.set_title("Accuracy")
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.set_ylim(0, 1)

    # 2. Loss comparison
    ax2 = fig.add_subplot(gs[0, 1])
    losses = [result["loss"] for result in results]
    ax2.bar(range(len(model_names)), losses, color="lightcoral", alpha=0.7)
    ax2.set_title("Loss (MSE)")
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha="right")

    # 3. Causal discovery score
    ax3 = fig.add_subplot(gs[0, 2])
    causal_scores = [result["causal_score"] for result in results]
    colors = ["green" if score > 0 else "red" for score in causal_scores]
    ax3.bar(range(len(model_names)), causal_scores, color=colors, alpha=0.7)
    ax3.set_title("Causal Discovery Score")
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha="right")
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # 4. Feature importance heatmap
    ax4 = fig.add_subplot(gs[1, :])
    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
    importance_matrix = []

    for result in results:
        if result["feature_importance"]:
            importance_matrix.append(result["feature_importance"])
        else:
            importance_matrix.append([0] * 5)

    importance_matrix = np.array(importance_matrix)
    im = ax4.imshow(importance_matrix, cmap="RdYlBu_r", aspect="auto")
    ax4.set_xticks(range(len(feature_names)))
    ax4.set_xticklabels(feature_names)
    ax4.set_yticks(range(len(model_names)))
    ax4.set_yticklabels(model_names)
    ax4.set_title("Feature Importance Heatmap")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Add text annotations to heatmap
    for i in range(len(model_names)):
        for j in range(len(feature_names)):
            ax4.text(
                j,
                i,
                f"{importance_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                weight="bold",
            )

    # 5. Robustness analysis (if available)
    if robustness_results:
        ax5 = fig.add_subplot(gs[2, 0])
        causal_sens = [
            robustness_results[name]["causal_sensitivity"]
            for name in model_names
            if name in robustness_results
        ]
        model_names_rob = [name for name in model_names if name in robustness_results]
        ax5.bar(range(len(model_names_rob)), causal_sens, color="lightgreen", alpha=0.7)
        ax5.set_title("Causal Sensitivity")
        ax5.set_xticks(range(len(model_names_rob)))
        ax5.set_xticklabels(model_names_rob, rotation=45, ha="right")
        ax5.set_ylim(0, 1)

        ax6 = fig.add_subplot(gs[2, 1])
        confound_rob = [
            robustness_results[name]["confound_robustness"]
            for name in model_names
            if name in robustness_results
        ]
        ax6.bar(range(len(model_names_rob)), confound_rob, color="lightblue", alpha=0.7)
        ax6.set_title("Confound Robustness")
        ax6.set_xticks(range(len(model_names_rob)))
        ax6.set_xticklabels(model_names_rob, rotation=45, ha="right")
        ax6.set_ylim(0, 1)

    # 6. Summary statistics table
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis("off")

    # Create summary table data
    table_data = []
    for result in results:
        row = [
            result["name"][:15],  # Truncate long names
            f"{result['accuracy']:.3f}",
            f"{result['loss']:.3f}",
            f"{result['causal_score']:.3f}",
        ]
        table_data.append(row)

    # Sort by accuracy (descending)
    table_data.sort(key=lambda x: float(x[1]), reverse=True)

    table = ax7.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Loss", "Causal Score"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax7.set_title("Performance Summary (Ranked by Accuracy)", pad=20)

    plt.suptitle(
        "Causal Learning Models: Comprehensive Performance Analysis",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
