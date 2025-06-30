import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class PredictiveCausalNeuron:
    """
    Model based on predictive coding theory and dendritic error computation.
    The idea: true causal factors should help predict outcomes better than
    correlational factors when tested through interventions.
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Prediction weights (what we think will happen)
        self.prediction_weights = np.random.randn(n_inputs) * 0.2

        # Error correction weights (how to fix prediction errors)
        self.error_weights = np.random.randn(n_inputs) * 0.2

        # Causal strength estimates (updated through interventions)
        self.causal_strengths = np.zeros(n_inputs)

        # Prediction error history for each input
        self.error_history = [[] for _ in range(n_inputs)]

        # Intervention tracking
        self.intervention_effects = np.zeros(n_inputs)
        self.intervention_counts = np.zeros(n_inputs)

        # Predictive accuracy per input
        self.prediction_accuracy = np.zeros(n_inputs)

        self.bias = np.random.randn() * 0.1

    def forward(self, inputs, use_predictions=True):
        """Forward pass with predictive processing"""
        if use_predictions:
            # Combine current inputs with predictions
            predicted_inputs = sigmoid(np.dot(self.prediction_weights, inputs))
            error_correction = np.dot(self.error_weights, inputs)

            total_activation = predicted_inputs + error_correction + self.bias
        else:
            # Direct processing without predictions
            total_activation = np.dot(self.prediction_weights, inputs) + self.bias

        return sigmoid(total_activation)

    def predict_intervention_outcome(
        self, inputs, intervention_input, intervention_value
    ):
        """Predict what would happen if we intervened on a specific input"""
        # Create modified input vector
        modified_inputs = inputs.copy()
        modified_inputs[intervention_input] = intervention_value

        # Predict outcome using current causal model
        predicted_effect = 0.0

        for i in range(self.n_inputs):
            if i == intervention_input:
                # Direct causal effect
                predicted_effect += self.causal_strengths[i] * intervention_value
            else:
                # Indirect effects through other variables
                predicted_effect += self.causal_strengths[i] * modified_inputs[i] * 0.1

        return sigmoid(predicted_effect + self.bias)

    def test_predictive_interventions(self, inputs, target):
        """Test causal effects by comparing predictions to actual intervention results"""
        intervention_scores = np.zeros(self.n_inputs)

        for input_idx in range(self.n_inputs):
            prediction_errors = []

            for intervention_val in [0, 1]:
                if inputs[input_idx] == intervention_val:
                    continue

                # Predict what would happen
                predicted_outcome = self.predict_intervention_outcome(
                    inputs, input_idx, intervention_val
                )

                # Actually do the intervention
                modified_inputs = inputs.copy()
                modified_inputs[input_idx] = intervention_val
                actual_outcome = self.forward(modified_inputs)

                # Measure prediction accuracy
                prediction_error = abs(predicted_outcome - actual_outcome)
                prediction_errors.append(prediction_error)

                # Also measure causal effect size
                causal_effect = abs(actual_outcome - self.forward(inputs))

                # Good causal factors: large effect, small prediction error
                if prediction_error < 0.1:  # Prediction was accurate
                    intervention_scores[input_idx] += causal_effect
                else:
                    intervention_scores[input_idx] += (
                        causal_effect * 0.1
                    )  # Penalty for bad prediction

            # Update prediction accuracy tracking
            if prediction_errors:
                avg_error = np.mean(prediction_errors)
                self.prediction_accuracy[input_idx] = (
                    0.9 * self.prediction_accuracy[input_idx]
                    + 0.1 * (1.0 - avg_error)  # Higher is better
                )

        return intervention_scores

    def update_causal_model(self, inputs, target):
        """Update causal model based on prediction errors and interventions"""
        # Test interventions and get scores
        intervention_scores = self.test_predictive_interventions(inputs, target)

        # Update causal strength estimates
        self.causal_strengths = 0.9 * self.causal_strengths + 0.1 * intervention_scores

        # Compute prediction error for this trial
        prediction = self.forward(inputs)
        prediction_error = target - prediction

        # Update prediction weights based on causal strengths
        for i in range(self.n_inputs):
            # Strong causal factors should contribute more to predictions
            causal_contribution = self.causal_strengths[i] / (
                np.sum(self.causal_strengths) + 1e-6
            )

            weight_update = (
                self.eta
                * prediction_error
                * inputs[i]
                * (1.0 + 2.0 * causal_contribution)  # Boost for causal factors
            )

            self.prediction_weights[i] += weight_update

        # Update error correction weights
        for i in range(self.n_inputs):
            # Error weights should help when predictions fail
            if self.prediction_accuracy[i] < 0.5:  # Poor prediction accuracy
                error_update = self.eta * prediction_error * inputs[i]
                self.error_weights[i] += error_update

        # Store error history
        for i in range(self.n_inputs):
            if inputs[i] > 0.5:  # Input was active
                self.error_history[i].append(abs(prediction_error))
                # Keep history manageable
                if len(self.error_history[i]) > 50:
                    self.error_history[i].pop(0)

    def get_causal_ranking(self):
        """Get inputs ranked by causal strength"""
        rankings = []

        for i in range(self.n_inputs):
            # Combine multiple evidence sources
            causal_score = self.causal_strengths[i]
            prediction_score = self.prediction_accuracy[i]

            # Error-based evidence: causal factors should lead to consistent predictions
            if len(self.error_history[i]) > 5:
                error_consistency = 1.0 / (1.0 + np.std(self.error_history[i]))
            else:
                error_consistency = 0.5

            # Final causal evidence
            total_evidence = (
                0.5 * causal_score + 0.3 * prediction_score + 0.2 * error_consistency
            )

            rankings.append((i, total_evidence))

        # Sort by evidence strength
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class ContrastiveCausalNeuron:
    """
    Model based on contrastive learning for causal discovery.
    Learns by comparing outcomes in similar situations with and without specific factors.
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Standard processing weights
        self.weights = np.random.randn(n_inputs) * 0.2

        # Contrastive evidence: how much each factor matters for discrimination
        self.contrastive_evidence = np.zeros(n_inputs)

        # Store recent examples for contrastive comparison
        self.positive_examples = []  # Cases where target = 1
        self.negative_examples = []  # Cases where target = 0
        self.max_examples = 100

        # Discrimination power per input
        self.discrimination_power = np.zeros(n_inputs)

        self.bias = np.random.randn() * 0.1

    def forward(self, inputs):
        """Standard forward pass"""
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def find_contrastive_pairs(self, current_inputs, current_target):
        """Find examples that differ minimally but have different outcomes"""
        contrastive_scores = np.zeros(self.n_inputs)

        # Compare with opposite class examples
        if current_target == 1:
            comparison_pool = self.negative_examples
        else:
            comparison_pool = self.positive_examples

        for comparison_inputs, comparison_target in comparison_pool:
            # Find inputs that differ between current and comparison
            differences = np.abs(current_inputs - comparison_inputs)

            # For each differing input, check if it explains the outcome difference
            for i in range(self.n_inputs):
                if differences[i] > 0.1:  # Inputs differ on this dimension
                    # This input differs and outcomes differ - potential causal factor
                    similarity = 1.0 - np.mean(differences)  # How similar overall

                    if similarity > 0.7:  # Very similar except for this factor
                        contrastive_scores[i] += 1.0

        return contrastive_scores

    def compute_discrimination_power(self):
        """Compute how well each input discriminates between positive and negative cases"""
        if len(self.positive_examples) < 5 or len(self.negative_examples) < 5:
            return

        for i in range(self.n_inputs):
            # Get input values for positive and negative cases
            pos_values = [ex[0][i] for ex in self.positive_examples]
            neg_values = [ex[0][i] for ex in self.negative_examples]

            # Measure separation between distributions
            pos_mean = np.mean(pos_values)
            neg_mean = np.mean(neg_values)

            pos_std = np.std(pos_values) + 1e-6
            neg_std = np.std(neg_values) + 1e-6

            # Cohen's d (effect size)
            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            effect_size = abs(pos_mean - neg_mean) / pooled_std

            self.discrimination_power[i] = effect_size

    def update_contrastive_learning(self, inputs, target):
        """Update using contrastive learning principles"""
        # Store this example
        if target == 1:
            self.positive_examples.append((inputs.copy(), target))
            if len(self.positive_examples) > self.max_examples:
                self.positive_examples.pop(0)
        else:
            self.negative_examples.append((inputs.copy(), target))
            if len(self.negative_examples) > self.max_examples:
                self.negative_examples.pop(0)

        # Find contrastive evidence
        contrastive_scores = self.find_contrastive_pairs(inputs, target)

        # Update contrastive evidence
        self.contrastive_evidence = (
            0.9 * self.contrastive_evidence + 0.1 * contrastive_scores
        )

        # Update discrimination power
        self.compute_discrimination_power()

        # Standard weight update with contrastive modulation
        prediction_error = target - self.forward(inputs)

        for i in range(self.n_inputs):
            # Modulate learning by contrastive evidence and discrimination power
            contrastive_boost = 1.0 + self.contrastive_evidence[i] * 0.5
            discrimination_boost = 1.0 + self.discrimination_power[i] * 0.3

            total_boost = contrastive_boost * discrimination_boost

            weight_update = self.eta * total_boost * prediction_error * inputs[i]
            self.weights[i] += weight_update


def generate_clear_causal_data(n_samples=1000):
    """Generate data with very clear causal structure for testing"""
    data = []

    for _ in range(n_samples):
        # True causal factors (necessary and sufficient)
        key_cause = np.random.random() > 0.5
        amplifier = np.random.random() > 0.5

        # Effect occurs when key cause is present
        base_effect_prob = 0.1
        if key_cause:
            base_effect_prob = 0.7
            if amplifier:  # Amplifier makes it even more likely
                base_effect_prob = 0.9

        effect = np.random.random() < base_effect_prob

        # Confounding factors - highly correlated with effect but not causal
        strong_confound = np.random.random() < (0.2 + 0.7 * effect)
        weak_confound = np.random.random() < (0.4 + 0.3 * effect)

        # Random noise
        noise1 = np.random.random() > 0.5
        noise2 = np.random.random() > 0.5

        inputs = np.array(
            [key_cause, amplifier, strong_confound, weak_confound, noise1, noise2],
            dtype=float,
        )

        data.append((inputs, effect))

    return data


def test_predictive_models():
    """Test predictive coding approaches to causal learning"""
    print("=== Testing Predictive Causal Learning Models ===\n")

    # Generate clear causal data
    train_data = generate_clear_causal_data(2000)
    test_data = generate_clear_causal_data(500)

    models = {
        "Predictive Coding": PredictiveCausalNeuron(6),
        "Contrastive Learning": ContrastiveCausalNeuron(6),
    }

    feature_names = [
        "KeyCause",
        "Amplifier",
        "StrongConfound",
        "WeakConfound",
        "Noise1",
        "Noise2",
    ]
    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} Model...")

        # Training
        for epoch in range(30):
            np.random.shuffle(train_data)

            for inputs, target in train_data:
                if model_name == "Predictive Coding":
                    model.update_causal_model(inputs, target)
                else:  # Contrastive
                    model.update_contrastive_learning(inputs, target)

        # Testing
        correct = 0
        for inputs, target in test_data:
            prediction = model.forward(inputs)
            if (prediction > 0.5) == target:
                correct += 1

        accuracy = correct / len(test_data)
        results[model_name] = {"accuracy": accuracy}

        print(f"\nFinal Test Accuracy: {accuracy:.3f}")

        # Model-specific analysis
        if model_name == "Predictive Coding":
            print("Causal Strength Rankings:")
            rankings = model.get_causal_ranking()
            for rank, (input_idx, evidence) in enumerate(rankings):
                print(f"  {rank + 1}. {feature_names[input_idx]}: {evidence:.4f}")

            print("\nPrediction Accuracy per Input:")
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {model.prediction_accuracy[i]:.4f}")

        else:  # Contrastive
            print("Contrastive Evidence:")
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {model.contrastive_evidence[i]:.4f}")

            print("Discrimination Power:")
            for i, feature in enumerate(feature_names):
                print(f"  {feature}: {model.discrimination_power[i]:.4f}")

        print()

    # Test causal intervention robustness
    print("=== Intervention Robustness Test ===")

    for model_name, model in models.items():
        print(f"\n{model_name}:")

        key_cause_failures = 0
        confound_robustness = 0

        for inputs, target in test_data[:100]:
            if target == 1:  # Test positive cases
                original_pred = model.forward(inputs)

                # Test removing key cause (should reduce prediction)
                if inputs[0] == 1:  # KeyCause was active
                    no_key_cause = inputs.copy()
                    no_key_cause[0] = 0
                    no_key_pred = model.forward(no_key_cause)

                    if no_key_pred >= original_pred:
                        key_cause_failures += 1

                # Test removing confounds (should NOT significantly reduce prediction)
                if inputs[2] == 1:  # Strong confound was active
                    no_confound = inputs.copy()
                    no_confound[2] = 0
                    no_confound_pred = model.forward(no_confound)

                    # Good model: removing confound shouldn't hurt much
                    if no_confound_pred > original_pred * 0.8:
                        confound_robustness += 1

        print(f"  Key cause intervention failures: {key_cause_failures}/100")
        print(f"  Confound robustness: {confound_robustness}/100")

    return results, models


if __name__ == "__main__":
    results, models = test_predictive_models()
