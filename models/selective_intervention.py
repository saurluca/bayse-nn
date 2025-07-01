import numpy as np
from utils import sigmoid


class SelectiveInterventionNeuron:
    """
    Learns by selectively enabling/disabling inputs during training.
    This model tests causal importance by randomly masking inputs
    and observing the effect on performance.
    """

    def __init__(self, n_inputs, eta=0.25, intervention_prob=0.3):
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = 0.0
        self.eta = eta
        self.n_inputs = n_inputs
        self.intervention_prob = intervention_prob
        self.causal_importance = np.ones(n_inputs)  # Track importance of each input
        self.causal_evidence = self.causal_importance  # Alias for compatibility

    def forward(self, inputs, mask=None):
        """Forward pass with optional input masking"""
        if mask is not None:
            masked_inputs = inputs * mask
        else:
            masked_inputs = inputs
        return sigmoid(np.dot(self.weights, masked_inputs) + self.bias)

    def update(self, inputs, target):
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

    def predict(self, inputs):
        """Alias for forward for consistency"""
        return self.forward(inputs)
