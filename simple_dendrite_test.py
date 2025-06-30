import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class BranchSpecificNeuron:
    """
    Simple model with branch-specific learning inspired by active dendrites research.
    Each branch specializes in detecting different types of causal relationships.
    """

    def __init__(self, n_inputs, n_branches=4, eta=0.1):
        self.n_inputs = n_inputs
        self.n_branches = n_branches
        self.eta = eta

        # Each branch has its own weights and specialization
        self.branch_weights = [
            np.random.randn(n_inputs) * 0.3 for _ in range(n_branches)
        ]

        # Branch specializations:
        # 0: Direct causality, 1: Interaction effects, 2: Context, 3: Inhibition
        self.branch_roles = ["direct", "interaction", "context", "inhibition"]

        # Branch activation thresholds
        self.thresholds = [0.3, 0.5, 0.4, 0.6]

        # Integration weights (learned)
        self.integration_weights = np.array([1.0, 0.8, 0.6, -0.5])

        # Causal evidence per branch
        self.causal_evidence = [np.zeros(n_inputs) for _ in range(n_branches)]

        # Branch states
        self.branch_activations = np.zeros(n_branches)

        self.bias = 0.0

    def compute_branch_outputs(self, inputs):
        """Compute outputs for each specialized branch"""
        outputs = []

        for branch_id in range(self.n_branches):
            weighted_sum = np.dot(self.branch_weights[branch_id], inputs)

            if self.branch_roles[branch_id] == "direct":
                # Direct pathway - linear integration
                output = sigmoid(weighted_sum)

            elif self.branch_roles[branch_id] == "interaction":
                # Interaction detection - supralinear
                if weighted_sum > self.thresholds[branch_id]:
                    output = sigmoid(weighted_sum + 0.5)  # Boost for interactions
                else:
                    output = sigmoid(weighted_sum * 0.5)  # Suppress weak signals

            elif self.branch_roles[branch_id] == "context":
                # Context pathway - slower, sustained
                context_strength = np.mean(inputs)  # Global context
                modulated_sum = weighted_sum * (1 + context_strength)
                output = sigmoid(modulated_sum)

            else:  # inhibition
                # Inhibitory pathway - detects spurious correlations
                correlation_strength = np.std(inputs)  # Detect uniform activation
                if correlation_strength < 0.3:  # Many inputs active together
                    output = sigmoid(weighted_sum + 0.3)  # Strong inhibition
                else:
                    output = sigmoid(weighted_sum * 0.2)  # Weak inhibition

            self.branch_activations[branch_id] = output
            outputs.append(output)

        return np.array(outputs)

    def forward(self, inputs):
        """Forward pass with branch specialization"""
        branch_outputs = self.compute_branch_outputs(inputs)

        # Weighted integration of branches
        net_input = np.dot(self.integration_weights, branch_outputs) + self.bias

        return sigmoid(net_input)

    def test_branch_interventions(self, inputs, target):
        """Test causal effects by branch"""
        original_output = self.forward(inputs)
        branch_scores = [np.zeros(self.n_inputs) for _ in range(self.n_branches)]

        for branch_id in range(self.n_branches):
            for input_idx in range(self.n_inputs):
                effects = []

                for intervention_val in [0, 1]:
                    if inputs[input_idx] == intervention_val:
                        continue

                    # Temporarily isolate this branch
                    temp_weights = [w.copy() for w in self.branch_weights]
                    for other_branch in range(self.n_branches):
                        if other_branch != branch_id:
                            self.branch_weights[other_branch] *= 0

                    # Test intervention
                    modified_inputs = inputs.copy()
                    modified_inputs[input_idx] = intervention_val

                    intervened_output = self.forward(modified_inputs)
                    effect = abs(intervened_output - original_output)
                    effects.append(effect)

                    # Restore weights
                    self.branch_weights = temp_weights

                if effects:
                    branch_scores[branch_id][input_idx] = np.mean(effects)

        return branch_scores

    def update_specialized_learning(self, inputs, target):
        """Update with branch-specific learning rules"""
        branch_scores = self.test_branch_interventions(inputs, target)
        prediction_error = target - self.forward(inputs)

        # Update causal evidence per branch
        for branch_id in range(self.n_branches):
            self.causal_evidence[branch_id] = (
                0.9 * self.causal_evidence[branch_id] + 0.1 * branch_scores[branch_id]
            )

            # Branch-specific learning rates
            if self.branch_roles[branch_id] == "direct":
                learning_rate = self.eta
            elif self.branch_roles[branch_id] == "interaction":
                learning_rate = self.eta * 0.5  # Slower for complex patterns
            elif self.branch_roles[branch_id] == "context":
                learning_rate = self.eta * 0.3  # Even slower for context
            else:  # inhibition
                learning_rate = self.eta * 0.8

            # Update weights based on causal evidence and branch activation
            for i in range(self.n_inputs):
                if self.causal_evidence[branch_id][i] > 0.05:
                    # Activity-dependent learning
                    activity_factor = 1.0 + self.branch_activations[branch_id]

                    weight_update = (
                        learning_rate * activity_factor * prediction_error * inputs[i]
                    )

                    self.branch_weights[branch_id][i] += weight_update

        # Update integration weights based on branch effectiveness
        for branch_id in range(self.n_branches):
            branch_effectiveness = np.mean(self.causal_evidence[branch_id])

            if branch_effectiveness > 0.03:
                integration_update = (
                    0.1
                    * self.eta
                    * prediction_error
                    * self.branch_activations[branch_id]
                )

                # Don't let inhibitory branch become positive
                if self.branch_roles[branch_id] == "inhibition":
                    self.integration_weights[branch_id] = min(
                        self.integration_weights[branch_id] + integration_update, -0.1
                    )
                else:
                    self.integration_weights[branch_id] += integration_update


def generate_specialized_data(n_samples=1000):
    """Generate data that benefits from branch specialization"""
    data = []

    for _ in range(n_samples):
        # Direct causal factor
        direct_cause = np.random.random() > 0.5

        # Interaction factors (only causal when both present)
        factor1 = np.random.random() > 0.5
        factor2 = np.random.random() > 0.5
        interaction_effect = factor1 and factor2

        # Context factor (modulates other effects)
        context = np.random.random() > 0.5

        # Confound (appears causal but isn't)
        confound = np.random.random() > 0.5

        # Compute effect
        base_prob = 0.1

        if direct_cause:
            base_prob += 0.4 * (1.5 if context else 1.0)  # Context amplifies

        if interaction_effect:
            base_prob += 0.3 * (1.3 if context else 1.0)  # Context amplifies

        # Confound correlation (not causal)
        if base_prob > 0.5:  # If effect likely
            confound = np.random.random() < 0.8  # Make confound highly correlated

        effect = np.random.random() < np.clip(base_prob, 0.05, 0.95)

        # Add noise
        noise = np.random.random() > 0.5

        inputs = np.array(
            [direct_cause, factor1, factor2, context, confound, noise], dtype=float
        )

        data.append((inputs, effect))

    return data


def test_branch_specialization():
    """Test the branch-specialized neuron"""
    print("=== Testing Branch-Specialized Causal Learning ===\n")

    # Generate specialized data
    train_data = generate_specialized_data(2000)
    test_data = generate_specialized_data(500)

    model = BranchSpecificNeuron(6, n_branches=4)

    feature_names = [
        "DirectCause",
        "Factor1",
        "Factor2",
        "Context",
        "Confound",
        "Noise",
    ]

    print("Training...")
    for epoch in range(30):
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            model.update_specialized_learning(inputs, target)

        if (epoch + 1) % 10 == 0:
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

    # Analyze branch specializations
    print("\n=== Branch Specialization Analysis ===")
    for branch_id, role in enumerate(model.branch_roles):
        print(f"\n{role.upper()} Branch (ID: {branch_id}):")
        print(f"  Integration weight: {model.integration_weights[branch_id]:.3f}")
        print("  Causal evidence:")
        for i, feature in enumerate(feature_names):
            evidence = model.causal_evidence[branch_id][i]
            print(f"    {feature}: {evidence:.4f}")

    # Test intervention robustness
    print("\n=== Intervention Robustness Test ===")

    # Test direct causality
    direct_failures = 0
    interaction_failures = 0

    for inputs, target in test_data[:100]:
        if target == 1:  # Positive cases only
            original_pred = model.forward(inputs)

            # Test removing direct cause
            if inputs[0] == 1:  # DirectCause was active
                no_direct = inputs.copy()
                no_direct[0] = 0
                no_direct_pred = model.forward(no_direct)

                if no_direct_pred >= original_pred:
                    direct_failures += 1

            # Test removing interaction
            if inputs[1] == 1 and inputs[2] == 1:  # Both factors active
                no_interaction = inputs.copy()
                no_interaction[1] = 0
                no_interaction[2] = 0
                no_interaction_pred = model.forward(no_interaction)

                if no_interaction_pred >= original_pred:
                    interaction_failures += 1

    print(f"Direct cause intervention failures: {direct_failures}/100")
    print(f"Interaction intervention failures: {interaction_failures}/100")
    print("(Lower is better)")

    return model


if __name__ == "__main__":
    model = test_branch_specialization()
