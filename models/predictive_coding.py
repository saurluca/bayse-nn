import numpy as np
from utils import sigmoid


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
        self.causal_evidence = self.causal_strengths  # Alias for compatibility

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

    def update(self, inputs, target):
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

    def predict(self, inputs):
        """Alias for forward for consistency"""
        return self.forward(inputs)

    @property
    def weights(self):
        """Return the primary weights for compatibility"""
        return self.prediction_weights
