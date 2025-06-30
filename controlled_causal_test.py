import numpy as np
from learning_comparison import BaselineNeuron, HybridCausalNeuron, evaluate_neuron


def generate_controlled_causal_data(n_samples=1000, correlation_strength=0.6):
    """
    Generate data with controlled causal vs correlational relationships
    to test if models can distinguish them
    """
    data = []

    for _ in range(n_samples):
        # TRUE CAUSAL FACTORS (these directly determine cancer)
        tumor_marker = np.random.randint(0, 2)
        genetic_risk = np.random.randint(0, 2)

        # PURE CAUSAL RELATIONSHIP - no noise, no correlation confusion
        # Cancer = 1 if (tumor_marker=1) OR (genetic_risk=1), else 0
        has_cancer = max(tumor_marker, genetic_risk)

        # CORRELATIONAL FACTORS (follow cancer but don't cause it)
        # Age: correlation_strength probability of matching cancer status
        if np.random.random() < correlation_strength:
            age = has_cancer
        else:
            age = 1 - has_cancer

        # Screening: correlation_strength probability of matching cancer status
        if np.random.random() < correlation_strength:
            screening = has_cancer
        else:
            screening = 1 - has_cancer

        # Pure noise
        noise = np.random.randint(0, 2)

        inputs = np.array([tumor_marker, genetic_risk, age, screening, noise])
        data.append((inputs, has_cancer))

    return data


def analyze_data_structure(data, name="Dataset"):
    """Analyze the causal vs correlational structure in the data"""

    print(f"\n=== Data Structure Analysis: {name} ===")

    # Convert to arrays for analysis
    inputs_array = np.array([d[0] for d in data])
    targets = np.array([d[1] for d in data])

    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]

    print("Individual feature correlations with cancer:")
    for i, name in enumerate(feature_names):
        correlation = np.corrcoef(inputs_array[:, i], targets)[0, 1]
        print(f"  {name:12}: {correlation:.3f}")

    # Test true causal effect by intervention
    print("\nTrue causal effects (by intervention on data generation):")

    # For each feature, see what happens when we force it to 0 vs 1
    for i, fname in enumerate(feature_names):
        effect_scores = []

        for force_val in [0, 1]:
            # Count cancer rate when this feature is forced to force_val
            matching_samples = inputs_array[inputs_array[:, i] == force_val]
            matching_targets = targets[inputs_array[:, i] == force_val]

            if len(matching_targets) > 0:
                cancer_rate = np.mean(matching_targets)
                effect_scores.append(cancer_rate)

        if len(effect_scores) == 2:
            causal_effect = (
                effect_scores[1] - effect_scores[0]
            )  # Rate when 1 minus rate when 0
            print(
                f"  {fname:12}: {causal_effect:+.3f} (cancer rate: 0→{effect_scores[0]:.3f}, 1→{effect_scores[1]:.3f})"
            )


class AggressiveCausalNeuron:
    """More aggressive causal learning that prioritizes causal evidence over correlation"""

    def __init__(self, n_inputs, eta=0.2, causal_strength=2.0):
        self.weights = np.random.randn(n_inputs) * 0.1  # Start smaller
        self.bias = 0.0
        self.eta = eta
        self.n_inputs = n_inputs
        self.causal_evidence = np.zeros(n_inputs)
        self.causal_strength = causal_strength

    def forward(self, inputs):
        return 1 / (
            1 + np.exp(-np.clip(np.dot(self.weights, inputs) + self.bias, -500, 500))
        )

    def test_aggressive_causality(self, inputs, target):
        """Aggressively test for causal relationships"""
        causal_scores = np.zeros(self.n_inputs)

        original_pred = self.forward(inputs)

        for i in range(self.n_inputs):
            # Test BOTH intervention directions
            intervention_effects = []

            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val
                intervened_pred = self.forward(modified_inputs)

                # Strong causal signal: does intervention move prediction toward target?
                original_distance = abs(target - original_pred)
                intervened_distance = abs(target - intervened_pred)

                improvement = original_distance - intervened_distance
                intervention_effects.append(improvement)

            if intervention_effects:
                # Take the BEST intervention effect (most positive)
                causal_scores[i] = max(intervention_effects)

        return causal_scores

    def update_aggressive_causal(self, inputs, target):
        """Aggressive causal learning that strongly weights causal evidence"""

        # Test causal effects
        causal_scores = self.test_aggressive_causality(inputs, target)

        # Update causal evidence with stronger momentum for positive evidence
        for i in range(self.n_inputs):
            if causal_scores[i] > 0:
                # Strengthen positive causal evidence quickly
                self.causal_evidence[i] = (
                    0.8 * self.causal_evidence[i] + 0.2 * causal_scores[i]
                )
            else:
                # Weaken negative evidence slowly
                self.causal_evidence[i] = (
                    0.95 * self.causal_evidence[i] + 0.05 * causal_scores[i]
                )

        # Weight updates ONLY for inputs with strong causal evidence
        prediction = self.forward(inputs)
        error = target - prediction
        sigmoid_deriv = prediction * (1 - prediction)

        for i in range(self.n_inputs):
            if (
                self.causal_evidence[i] > 0.02
            ):  # Only update inputs with positive causal evidence
                # Scale update by causal evidence strength
                causal_multiplier = 1 + self.causal_strength * self.causal_evidence[i]
                self.weights[i] += (
                    self.eta * error * inputs[i] * sigmoid_deriv * causal_multiplier
                )
            else:
                # Slightly reduce weights for non-causal inputs
                self.weights[i] *= 0.999


def run_controlled_experiment():
    """Run experiment with controlled causal structure"""

    print("=" * 90)
    print("CONTROLLED CAUSAL DISCOVERY EXPERIMENT")
    print("=" * 90)
    print("Testing if models can distinguish causal from correlational factors")
    print("when the causal structure is crystal clear in the data.")
    print()

    # Test different correlation strengths
    correlation_strengths = [0.6, 0.8, 0.95]

    for corr_strength in correlation_strengths:
        print(f"\n{'=' * 60}")
        print(f"TESTING WITH CORRELATION STRENGTH = {corr_strength}")
        print(f"{'=' * 60}")

        # Generate controlled data
        train_data = generate_controlled_causal_data(3000, corr_strength)
        test_data = generate_controlled_causal_data(1000, corr_strength)

        # Analyze data structure
        analyze_data_structure(test_data, f"Correlation {corr_strength}")

        # Test models
        models = {
            "Baseline": BaselineNeuron(n_inputs=5, eta=0.3),
            "Hybrid": HybridCausalNeuron(n_inputs=5, eta=0.2, causal_weight=1.0),
            "Aggressive Causal": AggressiveCausalNeuron(
                n_inputs=5, eta=0.2, causal_strength=3.0
            ),
        }

        # Train models
        epochs = 20
        print(f"\nTraining models for {epochs} epochs...")

        for epoch in range(epochs):
            np.random.shuffle(train_data)

            for inputs, target in train_data:
                models["Baseline"].update_correlation(inputs, target)
                models["Hybrid"].update_hybrid(inputs, target)
                models["Aggressive Causal"].update_aggressive_causal(inputs, target)

        # Evaluate
        print(f"\nResults for correlation strength {corr_strength}:")
        print("-" * 60)

        for name, model in models.items():
            result = evaluate_neuron(model, test_data, name)

            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  Weights: {result['weights']}")

            # Causal discovery quality
            importance = result["feature_importance"]
            causal_score = (importance[0] + importance[1]) - (
                importance[2] + importance[3] + abs(importance[4])
            )

            print(
                f"  Feature importance: Tumor={importance[0]:.3f}, Genetic={importance[1]:.3f}, Age={importance[2]:.3f}, Screen={importance[3]:.3f}, Noise={importance[4]:.3f}"
            )
            print(
                f"  Causal discovery score: {causal_score:.3f} (higher = better causal discovery)"
            )

            if hasattr(model, "causal_evidence"):
                print(f"  Causal evidence: {model.causal_evidence}")

        print("\nExpected: Tumor and Genetic should have HIGH importance (~0.5 each)")
        print("Expected: Age and Screening should have LOWER importance")
        print("Expected: Noise should have ~0 importance")


def test_intervention_robustness():
    """Test how robust models are to correlational confounders"""

    print(f"\n{'=' * 90}")
    print("INTERVENTION ROBUSTNESS TEST")
    print("=" * 90)
    print("Testing model predictions when we artificially break correlations")

    # Generate data with high correlation (0.9)
    test_data = generate_controlled_causal_data(1000, 0.9)

    # Train models on this data
    models = {
        "Baseline": BaselineNeuron(n_inputs=5, eta=0.3),
        "Aggressive Causal": AggressiveCausalNeuron(
            n_inputs=5, eta=0.2, causal_strength=3.0
        ),
    }

    for epoch in range(15):
        np.random.shuffle(test_data)
        for inputs, target in test_data:
            models["Baseline"].update_correlation(inputs, target)
            models["Aggressive Causal"].update_aggressive_causal(inputs, target)

    # Test on artificially constructed cases
    print("\nTesting on edge cases that break correlations:")

    test_cases = [
        ([1, 0, 0, 0, 0], "Tumor=1, everything else=0 (should predict cancer=1)"),
        ([0, 1, 0, 0, 0], "Genetic=1, everything else=0 (should predict cancer=1)"),
        ([0, 0, 1, 1, 0], "Only Age+Screening=1 (should predict cancer=0)"),
        ([0, 0, 0, 0, 1], "Only Noise=1 (should predict cancer=0)"),
    ]

    for test_input, description in test_cases:
        test_input = np.array(test_input)
        print(f"\n{description}:")

        for name, model in models.items():
            pred = model.forward(test_input)
            print(f"  {name:20}: {pred:.3f}")


if __name__ == "__main__":
    run_controlled_experiment()
    test_intervention_robustness()
