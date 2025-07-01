import numpy as np
from utils import sigmoid


class HybridCausalNeuron:
    """
    Combines correlation learning with causal intervention testing.
    This model learns both correlations and tests for causal relationships
    through random interventions.
    """

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
        """Standard forward pass"""
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def test_interventions(self, inputs, target):
        """Test causal effects through random interventions"""
        intervention_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            # Randomly sample intervention values
            intervention_effects = []

            for intervention_val in [0, 1]:  # Test both 0 and 1
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

    def update(self, inputs, target):
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

    def predict(self, inputs):
        """Alias for forward for consistency"""
        return self.forward(inputs)
