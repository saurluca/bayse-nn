import numpy as np
from utils import sigmoid


class BaselineNeuron:
    """
    Standard correlation-based learning (baseline model)
    This model only learns correlations without any causal reasoning
    """

    def __init__(self, n_inputs, eta=0.3):
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = np.random.randn() * 0.1
        self.eta = eta
        self.n_inputs = n_inputs

    def forward(self, inputs):
        """Standard forward pass"""
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def update(self, inputs, target):
        """Standard gradient descent - learns correlations only"""
        prediction = self.forward(inputs)
        error = target - prediction
        sigmoid_deriv = prediction * (1 - prediction)

        # Standard weight updates
        self.weights += self.eta * error * inputs * sigmoid_deriv
        self.bias += self.eta * error * sigmoid_deriv

    def predict(self, inputs):
        """Alias for forward for consistency"""
        return self.forward(inputs)
