import numpy as np
import matplotlib.pyplot as plt
from main import generate_cancer_data, sigmoid


class BaselineNeuron:
    """Standard correlation-based learning (baseline)"""

    def __init__(self, n_inputs, eta=0.3):
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = np.random.randn() * 0.1
        self.eta = eta
        self.n_inputs = n_inputs

    def forward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def update_correlation(self, inputs, target):
        """Standard gradient descent - learns correlations"""
        prediction = self.forward(inputs)
        error = target - prediction

        # Standard weight updates
        self.weights += (
            self.eta * error * inputs * prediction * (1 - prediction)
        )  # sigmoid derivative
        self.bias += self.eta * error * prediction * (1 - prediction)


class HybridCausalNeuron:
    """Combines correlation learning with causal intervention testing"""

    def __init__(self, n_inputs, eta=0.2, causal_weight=0.3):
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = np.random.randn() * 0.1
        self.eta = eta
        self.n_inputs = n_inputs
        self.causal_evidence = np.zeros(n_inputs)
        self.causal_weight = (
            causal_weight  # How much to weight causal vs correlation learning
        )

    def forward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def test_interventions(self, inputs, target):
        """Test causal effects through random interventions"""
        intervention_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            # Randomly sample intervention values
            intervention_effects = []

            for _ in range(2):  # Test both 0 and 1
                intervention_val = np.random.randint(0, 2)
                if inputs[i] == intervention_val:
                    continue

                # Perform intervention
                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                original_pred = self.forward(inputs)
                intervened_pred = self.forward(modified_inputs)

                # How much does intervention improve prediction?
                original_error = abs(target - original_pred)
                intervened_error = abs(target - intervened_pred)

                improvement = original_error - intervened_error
                intervention_effects.append(improvement)

            if intervention_effects:
                intervention_scores[i] = np.mean(intervention_effects)

        return intervention_scores

    def update_hybrid(self, inputs, target):
        """Hybrid learning: correlation + causal intervention"""
        prediction = self.forward(inputs)
        error = target - prediction
        sigmoid_deriv = prediction * (1 - prediction)

        # 1. Standard correlation learning
        correlation_update = self.eta * error * inputs * sigmoid_deriv

        # 2. Causal intervention testing
        intervention_scores = self.test_interventions(inputs, target)
        self.causal_evidence = 0.9 * self.causal_evidence + 0.1 * intervention_scores

        # 3. Combine updates with causal weighting
        for i in range(self.n_inputs):
            if self.causal_evidence[i] > 0.01:  # Positive causal evidence
                causal_multiplier = 1 + self.causal_weight * self.causal_evidence[i]
            else:  # Negative or no causal evidence
                causal_multiplier = 1 + self.causal_weight * self.causal_evidence[i]

            self.weights[i] += correlation_update[i] * causal_multiplier

        self.bias += self.eta * error * sigmoid_deriv


class SelectiveInterventionNeuron:
    """Learns by selectively enabling/disabling inputs during training"""

    def __init__(self, n_inputs, eta=0.25, intervention_prob=0.3):
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = np.random.randn() * 0.1
        self.eta = eta
        self.n_inputs = n_inputs
        self.intervention_prob = intervention_prob
        self.causal_importance = np.ones(n_inputs)  # Track importance of each input

    def forward(self, inputs, mask=None):
        """Forward pass with optional input masking"""
        if mask is not None:
            masked_inputs = inputs * mask
        else:
            masked_inputs = inputs
        return sigmoid(np.dot(self.weights, masked_inputs) + self.bias)

    def update_selective(self, inputs, target):
        """Learn by randomly disabling inputs and seeing effect on performance"""

        # 1. Normal prediction
        normal_pred = self.forward(inputs)
        normal_error = (target - normal_pred) ** 2

        # 2. Test selective interventions
        for i in range(self.n_inputs):
            if np.random.random() < self.intervention_prob:
                # Create mask that disables input i
                mask = np.ones(self.n_inputs)
                mask[i] = 0

                masked_pred = self.forward(inputs, mask)
                masked_error = (target - masked_pred) ** 2

                # If disabling this input hurts performance, it's important
                importance_change = masked_error - normal_error
                self.causal_importance[i] = (
                    0.95 * self.causal_importance[i] + 0.05 * importance_change
                )

        # 3. Weight update with importance scaling
        error = target - normal_pred
        sigmoid_deriv = normal_pred * (1 - normal_pred)

        for i in range(self.n_inputs):
            # Scale learning by causal importance
            importance_scale = max(
                0.1, self.causal_importance[i]
            )  # Don't let it go to zero
            self.weights[i] += (
                self.eta * error * inputs[i] * sigmoid_deriv * importance_scale
            )

        self.bias += self.eta * error * sigmoid_deriv


def evaluate_neuron(neuron, test_data, neuron_name):
    """Comprehensive evaluation of a neuron"""
    correct = 0
    total_loss = 0
    predictions = []
    targets = []

    for inputs, target in test_data:
        if hasattr(neuron, "forward"):
            pred = neuron.forward(inputs)
        else:
            pred = neuron.predict(inputs)

        predictions.append(pred)
        targets.append(target)

        # Accuracy
        pred_class = 1 if pred > 0.5 else 0
        if pred_class == target:
            correct += 1

        # Loss
        total_loss += (target - pred) ** 2

    accuracy = correct / len(test_data)
    avg_loss = total_loss / len(test_data)

    # Calculate feature importance through intervention
    test_inputs = np.array([1, 1, 1, 1, 0])  # High-risk patient
    if hasattr(neuron, "forward"):
        original_pred = neuron.forward(test_inputs)
    else:
        original_pred = neuron.predict(test_inputs)

    feature_importance = []
    for i in range(5):
        modified = test_inputs.copy()
        modified[i] = 0
        if hasattr(neuron, "forward"):
            new_pred = neuron.forward(modified)
        else:
            new_pred = neuron.predict(modified)
        importance = original_pred - new_pred
        feature_importance.append(importance)

    return {
        "name": neuron_name,
        "accuracy": accuracy,
        "loss": avg_loss,
        "feature_importance": feature_importance,
        "weights": getattr(neuron, "weights", None),
        "causal_evidence": getattr(neuron, "causal_evidence", None),
        "causal_importance": getattr(neuron, "causal_importance", None),
    }


def run_learning_experiment():
    """Run comprehensive comparison of learning methods"""

    print("=" * 80)
    print("COMPREHENSIVE LEARNING COMPARISON EXPERIMENT")
    print("=" * 80)
    print()

    # Generate data
    train_data = generate_cancer_data(5000)
    test_data = generate_cancer_data(1000)

    print("Data generated:")
    print(f"- Training samples: {len(train_data)}")
    print(f"- Test samples: {len(test_data)}")
    print()

    # Create different neurons
    neurons = {
        "Baseline (Correlation)": BaselineNeuron(n_inputs=5, eta=0.3),
        "Hybrid (Corr + Causal)": HybridCausalNeuron(
            n_inputs=5, eta=0.2, causal_weight=0.5
        ),
        "Selective Intervention": SelectiveInterventionNeuron(
            n_inputs=5, eta=0.25, intervention_prob=0.4
        ),
    }

    # Training
    epochs = 30
    results_history = {name: {"accuracy": [], "loss": []} for name in neurons.keys()}

    print("Training all models...")
    print()

    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(train_data)

        # Train each neuron
        for name, neuron in neurons.items():
            for inputs, target in train_data:
                if name == "Baseline (Correlation)":
                    neuron.update_correlation(inputs, target)
                elif name == "Hybrid (Corr + Causal)":
                    neuron.update_hybrid(inputs, target)
                elif name == "Selective Intervention":
                    neuron.update_selective(inputs, target)

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch}:")
            for name, neuron in neurons.items():
                result = evaluate_neuron(neuron, test_data, name)
                results_history[name]["accuracy"].append(result["accuracy"])
                results_history[name]["loss"].append(result["loss"])
                print(
                    f"  {name:25}: Accuracy={result['accuracy']:.3f}, Loss={result['loss']:.3f}"
                )
            print()

    # Final evaluation
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    final_results = {}
    for name, neuron in neurons.items():
        result = evaluate_neuron(neuron, test_data, name)
        final_results[name] = result

        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Loss: {result['loss']:.3f}")
        print(f"  Weights: {result['weights']}")
        if result["causal_evidence"] is not None:
            print(f"  Causal Evidence: {result['causal_evidence']}")
        if result["causal_importance"] is not None:
            print(f"  Causal Importance: {result['causal_importance']}")

    # Feature importance analysis
    print(f"\n{'=' * 80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
    expected_causal = [True, True, False, False, False]  # True causal factors

    print(
        f"{'Model':<25} | {'Tumor':<8} | {'Genetic':<8} | {'Age':<8} | {'Screen':<8} | {'Noise':<8}"
    )
    print("-" * 80)

    for name, result in final_results.items():
        importance = result["feature_importance"]
        print(
            f"{name:<25} | {importance[0]:<8.3f} | {importance[1]:<8.3f} | {importance[2]:<8.3f} | {importance[3]:<8.3f} | {importance[4]:<8.3f}"
        )

    print(
        f"{'Expected (causal)':<25} | {'HIGH':<8} | {'HIGH':<8} | {'low':<8} | {'low':<8} | {'~0':<8}"
    )

    # Critical analysis
    print(f"\n{'=' * 80}")
    print("CRITICAL ANALYSIS")
    print("=" * 80)

    # Find best performing model
    best_accuracy = max(result["accuracy"] for result in final_results.values())
    best_models = [
        name
        for name, result in final_results.items()
        if result["accuracy"] == best_accuracy
    ]

    print(f"Best accuracy: {best_accuracy:.3f} achieved by: {', '.join(best_models)}")
    print()

    # Analyze causal discovery quality
    print("Causal Discovery Quality:")
    for name, result in final_results.items():
        importance = result["feature_importance"]

        # Score causal discovery: high importance for causal factors, low for non-causal
        causal_score = (importance[0] + importance[1]) - (
            importance[2] + importance[3] + abs(importance[4])
        )
        print(f"  {name}: Causal score = {causal_score:.3f}")
        print(
            f"    Causal factors (Tumor+Genetic): {importance[0]:.3f} + {importance[1]:.3f} = {importance[0] + importance[1]:.3f}"
        )
        print(
            f"    Non-causal factors: Age={importance[2]:.3f}, Screen={importance[3]:.3f}, Noise={importance[4]:.3f}"
        )

    # Plot results
    plot_results(results_history, final_results, feature_names)

    return final_results


def plot_results(results_history, final_results, feature_names):
    """Create visualizations of the results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Accuracy over time
    ax1 = axes[0, 0]
    for name, history in results_history.items():
        epochs = range(0, len(history["accuracy"]) * 5, 5)
        ax1.plot(epochs, history["accuracy"], marker="o", label=name)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training Progress: Accuracy")
    ax1.legend()
    ax1.grid(True)

    # 2. Loss over time
    ax2 = axes[0, 1]
    for name, history in results_history.items():
        epochs = range(0, len(history["loss"]) * 5, 5)
        ax2.plot(epochs, history["loss"], marker="o", label=name)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Progress: Loss")
    ax2.legend()
    ax2.grid(True)

    # 3. Feature importance comparison
    ax3 = axes[1, 0]
    x = np.arange(len(feature_names))
    width = 0.25

    for i, (name, result) in enumerate(final_results.items()):
        ax3.bar(
            x + i * width, result["feature_importance"], width, label=name, alpha=0.8
        )

    ax3.set_xlabel("Features")
    ax3.set_ylabel("Importance (Cancer Prob Drop)")
    ax3.set_title("Feature Importance Comparison")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(feature_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Final accuracy comparison
    ax4 = axes[1, 1]
    names = list(final_results.keys())
    accuracies = [result["accuracy"] for result in final_results.values()]
    colors = ["blue", "orange", "green"]

    bars = ax4.bar(names, accuracies, color=colors, alpha=0.7)
    ax4.set_ylabel("Final Accuracy")
    ax4.set_title("Final Model Comparison")
    ax4.set_ylim(0, 1)

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("learning_comparison_results.png", dpi=150, bbox_inches="tight")
    print("\nResults saved to 'learning_comparison_results.png'")


if __name__ == "__main__":
    results = run_learning_experiment()
