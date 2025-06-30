import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Safe sigmoid function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class DendriticCausalNeuron:
    """
    Simplified dendritic model with three key mechanisms:
    1. Branch-specific processing
    2. Calcium-based learning
    3. Plateau potentials
    """

    def __init__(self, n_inputs, n_branches=3, eta=0.1):
        self.n_inputs = n_inputs
        self.n_branches = n_branches
        self.eta = eta

        # Branch weights: proximal (fast), distal (slow), inhibitory
        self.proximal_weights = np.random.randn(n_inputs) * 0.3
        self.distal_weights = np.random.randn(n_inputs) * 0.2
        self.inhibitory_weights = np.random.randn(n_inputs) * 0.1

        # Calcium levels (learning traces)
        self.calcium_proximal = np.zeros(n_inputs)
        self.calcium_distal = np.zeros(n_inputs)

        # Plateau potential state
        self.plateau_active = False
        self.plateau_strength = 0.0

        # Causal evidence per pathway
        self.proximal_evidence = np.zeros(n_inputs)
        self.distal_evidence = np.zeros(n_inputs)

        self.bias = np.random.randn() * 0.1

    def dendritic_integration(self, inputs):
        """Process inputs through different dendritic pathways"""
        # Proximal pathway (fast, direct)
        proximal_input = np.dot(self.proximal_weights, inputs)
        proximal_output = sigmoid(proximal_input)

        # Update proximal calcium (fast)
        self.calcium_proximal = 0.7 * self.calcium_proximal + 0.3 * inputs

        # Distal pathway (slow, contextual)
        distal_input = np.dot(self.distal_weights, inputs)

        # Check for plateau potential generation
        if distal_input > 0.5 and not self.plateau_active:
            self.plateau_active = True
            self.plateau_strength = min(1.0, distal_input)

        if self.plateau_active:
            # Plateau amplifies distal signal
            distal_output = sigmoid(distal_input + 0.5 * self.plateau_strength)
            self.plateau_strength *= 0.98  # Slow decay
            if self.plateau_strength < 0.1:
                self.plateau_active = False
        else:
            distal_output = sigmoid(distal_input)

        # Update distal calcium (slower)
        self.calcium_distal = 0.9 * self.calcium_distal + 0.1 * inputs

        # Inhibitory pathway (confound detection)
        inhibitory_input = np.dot(self.inhibitory_weights, inputs)
        inhibitory_output = sigmoid(inhibitory_input)

        return proximal_output, distal_output, inhibitory_output

    def forward(self, inputs):
        """Forward pass with dendritic integration"""
        proximal_out, distal_out, inhibitory_out = self.dendritic_integration(inputs)

        # Combine pathways with inhibition
        excitation = proximal_out + 0.5 * distal_out
        net_input = excitation / (1 + 0.3 * inhibitory_out) + self.bias

        return sigmoid(net_input)

    def test_pathway_causality(self, inputs, target):
        """Test causal contributions of different pathways"""
        original_output = self.forward(inputs)

        proximal_scores = np.zeros(self.n_inputs)
        distal_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                # Test proximal pathway alone
                temp_distal = self.distal_weights.copy()
                temp_inhib = self.inhibitory_weights.copy()
                self.distal_weights *= 0
                self.inhibitory_weights *= 0

                proximal_output = self.forward(modified_inputs)
                proximal_effect = abs(proximal_output - original_output)
                proximal_scores[i] += proximal_effect

                # Test distal pathway alone
                self.distal_weights = temp_distal
                self.inhibitory_weights *= 0
                temp_proximal = self.proximal_weights.copy()
                self.proximal_weights *= 0

                distal_output = self.forward(modified_inputs)
                distal_effect = abs(distal_output - original_output)
                distal_scores[i] += distal_effect

                # Restore weights
                self.proximal_weights = temp_proximal
                self.inhibitory_weights = temp_inhib

        return proximal_scores, distal_scores

    def update_dendritic_learning(self, inputs, target):
        """Update using dendritic-specific learning rules"""
        proximal_scores, distal_scores = self.test_pathway_causality(inputs, target)

        # Update evidence
        self.proximal_evidence = 0.9 * self.proximal_evidence + 0.1 * proximal_scores
        self.distal_evidence = 0.9 * self.distal_evidence + 0.1 * distal_scores

        prediction_error = target - self.forward(inputs)

        # Update proximal weights (calcium-dependent)
        for i in range(self.n_inputs):
            if self.proximal_evidence[i] > 0.05:
                calcium_boost = 1.0 + self.calcium_proximal[i]
                weight_update = self.eta * calcium_boost * prediction_error * inputs[i]
                self.proximal_weights[i] += weight_update

        # Update distal weights (plateau-dependent)
        for i in range(self.n_inputs):
            if self.distal_evidence[i] > 0.03:
                plateau_boost = (
                    1.0 + (2.0 if self.plateau_active else 0.0) * self.plateau_strength
                )
                weight_update = self.eta * plateau_boost * prediction_error * inputs[i]
                self.distal_weights[i] += weight_update

        # Update inhibitory weights (anti-correlation)
        for i in range(self.n_inputs):
            if self.proximal_evidence[i] < 0.02 and inputs[i] > 0.5:
                # Strengthen inhibition for non-causal but active inputs
                self.inhibitory_weights[i] += (
                    0.1 * self.eta * prediction_error * inputs[i]
                )


def generate_test_data(n_samples=1000):
    """Generate test data with clear causal structure"""
    data = []

    for _ in range(n_samples):
        # True causal factors
        cause1 = np.random.random() > 0.5
        cause2 = np.random.random() > 0.5

        # Causal effect with interaction
        if cause1 and cause2:
            effect_prob = 0.9
        elif cause1 or cause2:
            effect_prob = 0.6
        else:
            effect_prob = 0.1

        effect = np.random.random() < effect_prob

        # Confounding factors (correlated with effect but not causal)
        confound1 = np.random.random() < (0.3 + 0.5 * effect)
        confound2 = np.random.random() < (0.2 + 0.6 * effect)

        # Noise
        noise = np.random.random() > 0.5

        inputs = np.array([cause1, cause2, confound1, confound2, noise], dtype=float)
        target = float(effect)

        data.append((inputs, target))

    return data


def test_dendritic_model():
    """Test the dendritic causal neuron"""
    print("=== Testing Dendritic Causal Learning ===\n")

    # Generate data
    train_data = generate_test_data(2000)
    test_data = generate_test_data(500)

    # Create model
    model = DendriticCausalNeuron(5)

    feature_names = ["Cause1", "Cause2", "Confound1", "Confound2", "Noise"]

    # Training
    print("Training...")
    for epoch in range(30):
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            model.update_dendritic_learning(inputs, target)

        if (epoch + 1) % 10 == 0:
            # Test accuracy
            correct = 0
            for inputs, target in test_data[:100]:
                prediction = model.forward(inputs)
                if (prediction > 0.5) == target:
                    correct += 1
            print(f"Epoch {epoch + 1}: Accuracy = {correct / 100:.3f}")

    # Final testing
    correct = 0
    for inputs, target in test_data:
        prediction = model.forward(inputs)
        if (prediction > 0.5) == target:
            correct += 1

    accuracy = correct / len(test_data)
    print(f"\nFinal Test Accuracy: {accuracy:.3f}")

    # Analyze causal discovery
    print("\nCausal Evidence Analysis:")
    print("Proximal Pathway (Direct Causality):")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {model.proximal_evidence[i]:.4f}")

    print("\nDistal Pathway (Contextual Causality):")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {model.distal_evidence[i]:.4f}")

    print("\nCalcium Levels:")
    print(f"  Proximal: {model.calcium_proximal}")
    print(f"  Distal: {model.calcium_distal}")

    print(
        f"\nPlateau State: Active={model.plateau_active}, Strength={model.plateau_strength:.3f}"
    )

    # Test intervention robustness
    print("\n=== Intervention Robustness Test ===")
    intervention_failures = 0

    for inputs, target in test_data[:100]:
        if target == 1:  # Only test positive cases
            # Test if removing true causes reduces prediction
            causal_inputs = inputs.copy()
            causal_inputs[0] = 0  # Remove cause1
            causal_inputs[1] = 0  # Remove cause2

            original_pred = model.forward(inputs)
            intervened_pred = model.forward(causal_inputs)

            if intervened_pred >= original_pred:
                intervention_failures += 1

    print(f"Intervention failures: {intervention_failures}/100")
    print("(Lower is better - model should predict lower when causal inputs removed)")

    return model


def visualize_dendritic_mechanisms(model):
    """Visualize the learned dendritic mechanisms"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    feature_names = ["Cause1", "Cause2", "Confound1", "Confound2", "Noise"]

    # Proximal pathway evidence
    axes[0].bar(feature_names, model.proximal_evidence)
    axes[0].set_title("Proximal Pathway\n(Direct Causality)")
    axes[0].set_ylabel("Causal Evidence")
    axes[0].tick_params(axis="x", rotation=45)

    # Distal pathway evidence
    axes[1].bar(feature_names, model.distal_evidence)
    axes[1].set_title("Distal Pathway\n(Contextual Causality)")
    axes[1].set_ylabel("Causal Evidence")
    axes[1].tick_params(axis="x", rotation=45)

    # Calcium traces
    x = np.arange(len(feature_names))
    width = 0.35
    axes[2].bar(x - width / 2, model.calcium_proximal, width, label="Proximal Ca²⁺")
    axes[2].bar(x + width / 2, model.calcium_distal, width, label="Distal Ca²⁺")
    axes[2].set_title("Calcium Traces\n(Learning Signals)")
    axes[2].set_ylabel("Calcium Level")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(feature_names, rotation=45)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("dendritic_mechanisms.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    model = test_dendritic_model()
    visualize_dendritic_mechanisms(model)
