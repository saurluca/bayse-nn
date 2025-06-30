import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow


class CausalNeuron:
    def __init__(self, n_inputs, eta=0.1, threshold_change=0.05):
        self.weights = np.random.randn(n_inputs) * 0.3  # Random initialization
        self.n_inputs = n_inputs
        self.bias = np.random.randn() * 0.1
        self.eta = eta
        self.threshold_change = threshold_change
        self.causal_evidence = np.zeros(n_inputs)  # Track causal evidence
        self.intervention_history = []  # Track intervention results

    def forward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def test_causal_effect(self, inputs, target_cancer):
        """
        Test if each input has a TRUE causal effect by intervention
        Returns causal scores that measure how much interventions help predict the target
        """
        original_output = self.forward(inputs)
        causal_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            intervention_effects = []

            # Test both intervention values
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue  # Skip if no change

                # Perform intervention
                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val
                intervened_output = self.forward(modified_inputs)

                # Key insight: TRUE causal effect should help predict the target better
                original_error = (target_cancer - original_output) ** 2
                intervened_error = (target_cancer - intervened_output) ** 2

                # Positive score if intervention reduces prediction error
                error_reduction = original_error - intervened_error
                intervention_effects.append(error_reduction)

            # Causal score is average error reduction from interventions
            if intervention_effects:
                causal_scores[i] = np.mean(intervention_effects)

        return causal_scores

    def update_causal_supervised(self, inputs, target_cancer):
        """
        Enhanced supervised causal learning that separates causal discovery from weight updates
        """
        # Step 1: Test for causal effects
        causal_scores = self.test_causal_effect(inputs, target_cancer)

        # Step 2: Update causal evidence with momentum
        self.causal_evidence = 0.95 * self.causal_evidence + 0.05 * causal_scores

        # Step 3: Only update weights for inputs with strong positive causal evidence
        prediction_error = target_cancer - self.forward(inputs)

        for i in range(self.n_inputs):
            # Strong threshold: only update if causal evidence is substantial
            if self.causal_evidence[i] > 0.02:  # Increased threshold
                # Standard supervised update, but only for causally relevant inputs
                self.weights[i] += self.eta * prediction_error * inputs[i]
            elif self.causal_evidence[i] < -0.02:  # Negative causal evidence
                # Reduce weight for inputs that hurt prediction through interventions
                self.weights[i] -= self.eta * 0.1 * inputs[i]


def generate_cancer_data(n_samples=1000):
    """
    Generate realistic cancer detection data with clearer causal structure
    """
    data = []

    for _ in range(n_samples):
        # True causal factors
        tumor_marker = np.random.random()  # 0 to 1
        genetic_risk = np.random.random()  # 0 to 1

        # Stronger causal relationship for clearer testing
        cancer_prob = 0.05 + 0.7 * tumor_marker + 0.4 * genetic_risk
        cancer_prob = min(cancer_prob, 0.95)

        has_cancer = 1 if np.random.random() < cancer_prob else 0

        # Age: correlated but not directly causal
        if has_cancer:
            age_score = np.random.normal(0.75, 0.15)  # Higher age if cancer
        else:
            age_score = np.random.normal(0.35, 0.15)  # Lower age if no cancer
        age_score = np.clip(age_score, 0, 1)

        # Previous screening: follows cancer status (not causal)
        screening_accuracy = 0.85
        if np.random.random() < screening_accuracy:
            prev_screening = has_cancer
        else:
            prev_screening = 1 - has_cancer

        # Pure noise
        noise = np.random.random()

        # Convert to binary
        inputs = np.array(
            [
                1 if tumor_marker > 0.5 else 0,  # Tumor marker
                1 if genetic_risk > 0.5 else 0,  # Genetic risk
                1 if age_score > 0.6 else 0,  # Age
                prev_screening,  # Previous screening
                1 if noise > 0.5 else 0,  # Noise
            ]
        )

        data.append((inputs, has_cancer))

    return data


def test_cancer_detection():
    """Enhanced test with better evaluation metrics"""

    print("=== Enhanced Causal Learning for Cancer Detection ===")
    print("Goal: Neuron should SPIKE when patient has cancer")
    print("Challenge: Distinguish CAUSAL factors from CORRELATED factors")
    print()
    print("Input features:")
    print("- Input 0: Tumor marker (CAUSAL)")
    print("- Input 1: Genetic risk (CAUSAL)")
    print("- Input 2: Age (CORRELATED only)")
    print("- Input 3: Previous screening (CORRELATED only)")
    print("- Input 4: Random noise (NO RELATIONSHIP)")
    print()

    # Generate data
    training_data = generate_cancer_data(3000)
    test_data = generate_cancer_data(1000)

    # Create neuron
    neuron = CausalNeuron(n_inputs=5, eta=0.2, threshold_change=0.05)

    print(f"Initial weights: {neuron.weights}")
    print()

    # Training with progress tracking
    print("Training progress:")
    for epoch in range(25):
        np.random.shuffle(training_data)

        for inputs, target in training_data:
            neuron.update_causal_supervised(inputs, target)

        if epoch % 5 == 0:
            # Test accuracy
            correct = 0
            total_cancer_prob = 0

            for test_inputs, test_target in test_data:
                prediction = neuron.forward(test_inputs)
                total_cancer_prob += prediction
                predicted_cancer = 1 if prediction > 0.5 else 0
                if predicted_cancer == test_target:
                    correct += 1

            accuracy = correct / len(test_data)
            avg_output = total_cancer_prob / len(test_data)

            print(
                f"Epoch {epoch:2d}: Accuracy={accuracy:.3f}, Avg_output={avg_output:.3f}"
            )
            print(f"           Weights: {neuron.weights}")
            print(f"           Causal evidence: {neuron.causal_evidence}")
            print()

    # Final evaluation
    print("=== FINAL RESULTS ===")
    print(f"Final weights: {neuron.weights}")
    print(f"Causal evidence: {neuron.causal_evidence}")
    print()

    # Expected vs actual pattern
    feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
    expected_causal = [0.8, 0.6, 0.1, 0.1, 0.0]  # Expected causal evidence

    print("Feature Analysis:")
    print("Feature      | Weight  | Causal Evidence | Expected | Status")
    print("-" * 65)
    for i, name in enumerate(feature_names):
        status = (
            "✓"
            if (expected_causal[i] > 0.3 and neuron.causal_evidence[i] > 0.05)
            or (expected_causal[i] < 0.3 and neuron.causal_evidence[i] < 0.05)
            else "✗"
        )
        print(
            f"{name:12} | {neuron.weights[i]:6.3f} | {neuron.causal_evidence[i]:14.3f} | {expected_causal[i]:8.1f} | {status}"
        )

    # Intervention test
    print("\n=== Intervention Test on High-Risk Patient ===")
    high_risk_patient = np.array([1, 1, 1, 1, 0])  # All positive except noise
    original_prob = neuron.forward(high_risk_patient)
    print(f"High-risk patient: {high_risk_patient}")
    print(f"Original cancer probability: {original_prob:.3f}")
    print()

    print("What happens if we remove each factor?")
    for i, name in enumerate(feature_names):
        modified = high_risk_patient.copy()
        modified[i] = 0  # Remove this factor
        new_prob = neuron.forward(modified)
        change = original_prob - new_prob  # How much probability drops
        print(
            f"Remove {name:12}: prob drops by {change:+.3f} (new prob: {new_prob:.3f})"
        )

    print("\nInterpretation:")
    print("- Large drops for Tumor/Genetic indicate causal importance")
    print("- Small drops for Age/Screening indicate correlational-only")
    print("- Near-zero drops for Noise indicate no relationship")

    return neuron


# Run the cancer detection experiment
if __name__ == "__main__":
    neuron = test_cancer_detection()
