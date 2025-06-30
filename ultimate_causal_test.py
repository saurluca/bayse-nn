import numpy as np
from learning_comparison import BaselineNeuron, HybridCausalNeuron, evaluate_neuron


def generate_ultimate_challenge_data(n_samples=1000):
    """
    Generate data where CORRELATIONAL signals are STRONGER than CAUSAL signals
    This is the ultimate test for causal discovery methods
    """
    data = []

    for _ in range(n_samples):
        # TRUE CAUSAL FACTORS (weak individual effects)
        tumor_marker = np.random.randint(0, 2)
        genetic_risk = np.random.randint(0, 2)

        # WEAK CAUSAL RELATIONSHIP: both factors needed for high cancer risk
        # Cancer probability depends on BOTH factors together (interaction effect)
        if tumor_marker == 1 and genetic_risk == 1:
            cancer_prob = 0.8  # High risk only when both present
        elif tumor_marker == 1 or genetic_risk == 1:
            cancer_prob = 0.3  # Medium risk with one factor
        else:
            cancer_prob = 0.1  # Low risk with no factors

        has_cancer = 1 if np.random.random() < cancer_prob else 0

        # STRONG CORRELATIONAL FACTORS (stronger individual correlations than causal factors)
        # Age: Very strong correlation (0.9) but NOT causal
        if np.random.random() < 0.9:
            age = has_cancer
        else:
            age = 1 - has_cancer

        # Lifestyle: Very strong correlation (0.85) but NOT causal
        if np.random.random() < 0.85:
            lifestyle = has_cancer
        else:
            lifestyle = 1 - has_cancer

        # Pure noise
        noise = np.random.randint(0, 2)

        inputs = np.array([tumor_marker, genetic_risk, age, lifestyle, noise])
        data.append((inputs, has_cancer))

    return data


class InteractionAwareCausalNeuron:
    """Causal learning that can detect interaction effects between inputs"""

    def __init__(self, n_inputs, eta=0.15):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.interaction_weights = (
            np.random.randn(n_inputs, n_inputs) * 0.05
        )  # Interaction terms
        self.bias = 0.0
        self.eta = eta
        self.n_inputs = n_inputs
        self.causal_evidence = np.zeros(n_inputs)
        self.interaction_evidence = np.zeros((n_inputs, n_inputs))

    def forward(self, inputs):
        # Linear terms
        linear_output = np.dot(self.weights, inputs)

        # Interaction terms
        interaction_output = 0
        for i in range(self.n_inputs):
            for j in range(i + 1, self.n_inputs):
                interaction_output += (
                    self.interaction_weights[i, j] * inputs[i] * inputs[j]
                )

        total_output = linear_output + interaction_output + self.bias
        return 1 / (1 + np.exp(-np.clip(total_output, -500, 500)))

    def test_interactions(self, inputs, target):
        """Test for causal interactions between inputs"""
        original_pred = self.forward(inputs)
        interaction_scores = np.zeros((self.n_inputs, self.n_inputs))

        # Test all pairs of inputs
        for i in range(self.n_inputs):
            for j in range(i + 1, self.n_inputs):
                # Test the four combinations of inputs i and j
                combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
                effects = []

                for val_i, val_j in combinations:
                    if inputs[i] == val_i and inputs[j] == val_j:
                        continue

                    modified = inputs.copy()
                    modified[i] = val_i
                    modified[j] = val_j

                    modified_pred = self.forward(modified)

                    # Does this combination improve prediction?
                    original_error = abs(target - original_pred)
                    modified_error = abs(target - modified_pred)
                    improvement = original_error - modified_error

                    effects.append(improvement)

                if effects:
                    interaction_scores[i, j] = max(effects)  # Best improvement

        return interaction_scores

    def test_individual_causality(self, inputs, target):
        """Test individual causal effects"""
        causal_scores = np.zeros(self.n_inputs)
        original_pred = self.forward(inputs)

        for i in range(self.n_inputs):
            effects = []
            for val in [0, 1]:
                if inputs[i] == val:
                    continue

                modified = inputs.copy()
                modified[i] = val
                modified_pred = self.forward(modified)

                original_error = abs(target - original_pred)
                modified_error = abs(target - modified_pred)
                improvement = original_error - modified_error
                effects.append(improvement)

            if effects:
                causal_scores[i] = max(effects)

        return causal_scores

    def update_interaction_causal(self, inputs, target):
        """Update with both individual and interaction causal learning"""

        # Test individual and interaction causality
        individual_scores = self.test_individual_causality(inputs, target)
        interaction_scores = self.test_interactions(inputs, target)

        # Update evidence
        self.causal_evidence = 0.9 * self.causal_evidence + 0.1 * individual_scores
        self.interaction_evidence = (
            0.9 * self.interaction_evidence + 0.1 * interaction_scores
        )

        # Update weights
        prediction = self.forward(inputs)
        error = target - prediction
        sigmoid_deriv = prediction * (1 - prediction)

        # Individual weight updates (only for inputs with causal evidence)
        for i in range(self.n_inputs):
            if self.causal_evidence[i] > 0.01:
                self.weights[i] += self.eta * error * inputs[i] * sigmoid_deriv

        # Interaction weight updates
        for i in range(self.n_inputs):
            for j in range(i + 1, self.n_inputs):
                if self.interaction_evidence[i, j] > 0.01:
                    interaction_term = inputs[i] * inputs[j]
                    self.interaction_weights[i, j] += (
                        self.eta * error * interaction_term * sigmoid_deriv
                    )

        self.bias += self.eta * error * sigmoid_deriv * 0.1


def run_ultimate_challenge():
    """The ultimate challenge: Can causal learning beat strong correlational confounders?"""

    print("=" * 100)
    print("THE ULTIMATE CAUSAL DISCOVERY CHALLENGE")
    print("=" * 100)
    print("Testing scenario where:")
    print("- CAUSAL factors have weak individual effects but strong interaction")
    print("- CORRELATIONAL factors have very strong individual correlations")
    print("- Question: Can causal learning methods discover the true causal structure?")
    print()

    # Generate the ultimate challenge data
    train_data = generate_ultimate_challenge_data(4000)
    test_data = generate_ultimate_challenge_data(1000)

    # Analyze the data structure
    print("=== Data Structure Analysis ===")
    inputs_array = np.array([d[0] for d in train_data])
    targets = np.array([d[1] for d in train_data])

    feature_names = ["Tumor", "Genetic", "Age", "Lifestyle", "Noise"]

    print("Individual correlations with cancer:")
    for i, name in enumerate(feature_names):
        correlation = np.corrcoef(inputs_array[:, i], targets)[0, 1]
        print(f"  {name:12}: {correlation:.3f}")

    # Test interaction effect
    tumor_and_genetic = (
        inputs_array[:, 0] * inputs_array[:, 1]
    )  # Both tumor and genetic
    interaction_corr = np.corrcoef(tumor_and_genetic, targets)[0, 1]
    print(f"  Tumor*Genetic : {interaction_corr:.3f} (interaction)")

    print()
    print("Cancer rates by factor combinations:")
    for tumor in [0, 1]:
        for genetic in [0, 1]:
            mask = (inputs_array[:, 0] == tumor) & (inputs_array[:, 1] == genetic)
            if np.sum(mask) > 0:
                cancer_rate = np.mean(targets[mask])
                print(
                    f"  Tumor={tumor}, Genetic={genetic}: {cancer_rate:.3f} cancer rate"
                )

    # Test different models
    models = {
        "Baseline Correlation": BaselineNeuron(n_inputs=5, eta=0.25),
        "Hybrid Causal": HybridCausalNeuron(n_inputs=5, eta=0.2, causal_weight=1.5),
        "Interaction Causal": InteractionAwareCausalNeuron(n_inputs=5, eta=0.15),
    }

    print("\n=== Training Models ===")
    epochs = 25

    for epoch in range(epochs):
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            models["Baseline Correlation"].update_correlation(inputs, target)
            models["Hybrid Causal"].update_hybrid(inputs, target)
            models["Interaction Causal"].update_interaction_causal(inputs, target)

        if epoch % 5 == 0:
            print(f"Epoch {epoch} completed...")

    # Evaluate models
    print(f"\n{'=' * 100}")
    print("FINAL RESULTS")
    print("=" * 100)

    for name, model in models.items():
        result = evaluate_neuron(model, test_data, name)

        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Loss: {result['loss']:.3f}")
        print(f"  Weights: {result['weights']}")

        # Feature importance analysis
        importance = result["feature_importance"]
        print("  Feature importance:")
        for i, fname in enumerate(feature_names):
            print(f"    {fname:12}: {importance[i]:.3f}")

        # Causal discovery score
        causal_factors_importance = importance[0] + importance[1]
        correlational_factors_importance = importance[2] + importance[3]
        causal_score = causal_factors_importance - correlational_factors_importance
        print(f"  Causal discovery score: {causal_score:.3f}")

        if hasattr(model, "causal_evidence"):
            print(f"  Individual causal evidence: {model.causal_evidence}")

        if hasattr(model, "interaction_evidence"):
            max_interaction = np.max(model.interaction_evidence)
            print(f"  Strongest interaction evidence: {max_interaction:.3f}")

    # Test critical cases
    print(f"\n{'=' * 100}")
    print("CRITICAL INTERVENTION TESTS")
    print("=" * 100)
    print("Testing on cases that distinguish causal from correlational learning:")

    test_cases = [
        (
            [1, 1, 0, 0, 0],
            "Both causal factors, no correlational (should predict HIGH cancer)",
        ),
        (
            [0, 0, 1, 1, 0],
            "No causal factors, both correlational (should predict LOW cancer)",
        ),
        ([1, 0, 0, 0, 0], "One causal factor only (should predict MEDIUM cancer)"),
        ([0, 1, 0, 0, 0], "Other causal factor only (should predict MEDIUM cancer)"),
    ]

    for test_input, description in test_cases:
        test_input = np.array(test_input)
        print(f"\n{description}:")

        for name, model in models.items():
            if hasattr(model, "forward"):
                pred = model.forward(test_input)
            else:
                pred = model.predict(test_input)
            print(f"  {name:20}: {pred:.3f}")

    print(f"\n{'=' * 100}")
    print("INTERPRETATION")
    print("=" * 100)
    print("Success criteria:")
    print("- HIGH accuracy (>80%)")
    print(
        "- Causal factors (Tumor, Genetic) should have higher importance than correlational (Age, Lifestyle)"
    )
    print(
        "- Critical test: 'Both causal factors' should give highest cancer prediction"
    )
    print(
        "- Critical test: 'Both correlational factors' should give lowest cancer prediction"
    )


if __name__ == "__main__":
    run_ultimate_challenge()
