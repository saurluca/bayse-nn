import numpy as np
from utils import sigmoid


class CausalDendriteNeuron:
    """
    Direct Causal Learning Neuron

    Core principle: Learn to distinguish causation from correlation through:
    1. Direct intervention testing during training
    2. Track feature importance under normal vs intervened conditions
    3. Bias learning toward genuinely causal features

    Simple but effective approach inspired by Pearl's causal framework.
    """

    def __init__(
        self,
        n_inputs,
        n_dendrites=3,
        eta=0.2,
        alpha_init=0.8,
        tau_init=0.5,
        intervention_strength=0.7,
    ):
        self.n_inputs = n_inputs
        self.n_dendrites = n_dendrites
        self.eta = eta
        self.intervention_strength = intervention_strength

        # Main prediction weights
        self.weights = np.random.randn(n_inputs) * 0.3
        self.bias = np.random.randn() * 0.1

        # Causal learning tracking
        self.causal_evidence = np.zeros(n_inputs)
        self.normal_correlations = np.zeros(
            n_inputs
        )  # Feature-target correlations under normal conditions
        self.intervention_correlations = np.zeros(
            n_inputs
        )  # Feature-target correlations under intervention

        # Dendritic temporal processing (simplified)
        self.dendritic_states = np.zeros((n_dendrites, n_inputs))
        self.temporal_decay = np.linspace(
            0.3, 0.8, n_dendrites
        )  # Different time constants

        # Learning statistics
        self.updates = 0
        self.intervention_tests = 0
        self.feature_activation_count = np.zeros(n_inputs)
        self.feature_success_count = np.zeros(n_inputs)

        # Intervention scheduling
        self.last_intervention_feature = -1
        self.intervention_cooldown = 0

    def _update_dendritic_states(self, inputs):
        """Update temporal dendritic integration with different time constants"""
        for d in range(self.n_dendrites):
            decay = self.temporal_decay[d]
            self.dendritic_states[d] = (
                decay * self.dendritic_states[d] + (1 - decay) * inputs
            )
        return np.mean(self.dendritic_states, axis=0)  # Average across dendrites

    def _should_intervene(self):
        """Decide whether to perform causal intervention"""
        if self.updates < 50:  # Build up some baseline first
            return False
        if self.intervention_cooldown > 0:
            self.intervention_cooldown -= 1
            return False
        return np.random.random() < 0.15  # 15% intervention rate

    def _select_intervention_target(self, inputs):
        """Select which feature to intervene on"""
        # Focus on features that are:
        # 1. Currently active
        # 2. Have strong normal correlations but uncertain causal status
        active_features = np.where(inputs > 0.5)[0]
        if len(active_features) == 0:
            return -1

        # Calculate intervention priority: high correlation but low causal certainty
        priorities = np.zeros(self.n_inputs)
        for i in active_features:
            correlation_strength = abs(self.normal_correlations[i])
            causal_uncertainty = 1.0 - abs(self.causal_evidence[i])
            priorities[i] = correlation_strength * causal_uncertainty

        if np.max(priorities) > 0.01:
            return np.argmax(priorities)
        return np.random.choice(active_features)  # Random if no clear choice

    def _perform_intervention(self, inputs, target_feature):
        """Intervene on a specific feature"""
        modified_inputs = inputs.copy()
        # Strong intervention: set feature to opposite of its current tendency
        if inputs[target_feature] > 0.5:
            modified_inputs[target_feature] = (
                self.intervention_strength * inputs[target_feature]
            )
        else:
            modified_inputs[target_feature] = 1.0 - self.intervention_strength
        return modified_inputs

    def forward(self, inputs):
        """Forward pass with dendritic temporal processing"""
        # Update dendritic temporal integration
        dendritic_input = self._update_dendritic_states(inputs)

        # Combine current inputs with temporal context
        combined_input = 0.7 * inputs + 0.3 * dendritic_input

        # Weight by causal evidence (bias toward causal features)
        causal_bias = 1.0 + 0.4 * self.causal_evidence
        weighted_input = combined_input * causal_bias

        # Standard prediction
        activation = np.dot(self.weights, weighted_input) + self.bias
        return sigmoid(activation)

    def update(self, inputs, target):
        """Update with direct causal intervention testing"""
        self.updates += 1

        # Track feature activations and successes under normal conditions
        for i in range(self.n_inputs):
            if inputs[i] > 0.5:
                self.feature_activation_count[i] += 1
                if target > 0.5:
                    self.feature_success_count[i] += 1

        # Update normal correlations (feature importance under normal conditions)
        prediction = self.forward(inputs)
        error = target - prediction

        # Update normal correlations based on prediction accuracy
        for i in range(self.n_inputs):
            if inputs[i] > 0.5:
                # If feature is active and prediction is good, strengthen correlation
                correlation_update = inputs[i] * (1.0 - abs(error))
                self.normal_correlations[i] = (
                    0.9 * self.normal_correlations[i] + 0.1 * correlation_update
                )

        # Decide whether to perform intervention
        perform_intervention = self._should_intervene()
        intervention_target = -1

        if perform_intervention:
            intervention_target = self._select_intervention_target(inputs)

            if intervention_target >= 0:
                self.intervention_tests += 1
                self.last_intervention_feature = intervention_target
                self.intervention_cooldown = 5  # Wait before next intervention

                # Perform intervention and measure effect
                modified_inputs = self._perform_intervention(
                    inputs, intervention_target
                )
                intervention_prediction = self.forward(modified_inputs)
                intervention_error = target - intervention_prediction

                # Compare normal vs intervention performance
                normal_accuracy = 1.0 - abs(error)
                intervention_accuracy = 1.0 - abs(intervention_error)

                # If intervention significantly hurts performance, feature is likely causal
                performance_drop = normal_accuracy - intervention_accuracy

                # Update causal evidence
                if performance_drop > 0.05:  # Significant performance drop
                    causal_boost = min(0.2, performance_drop * 3)
                    self.causal_evidence[intervention_target] = min(
                        1.0, self.causal_evidence[intervention_target] + causal_boost
                    )
                elif performance_drop < -0.02:  # Performance actually improved
                    # Feature might be spurious/correlational
                    self.causal_evidence[intervention_target] *= 0.95

                # Update intervention correlations
                if modified_inputs[intervention_target] > 0.5:
                    correlation_update = (
                        modified_inputs[intervention_target] * intervention_accuracy
                    )
                    self.intervention_correlations[intervention_target] = (
                        0.8 * self.intervention_correlations[intervention_target]
                        + 0.2 * correlation_update
                    )

                # Use intervened inputs for learning this step
                inputs = modified_inputs
                prediction = intervention_prediction
                error = intervention_error

        # Standard weight updates with causal bias
        sigmoid_deriv = prediction * (1 - prediction)

        # Boost learning rate for causal features
        causal_learning_boost = 1.0 + 0.3 * self.causal_evidence
        learning_rates = self.eta * causal_learning_boost

        # Update weights
        for i in range(self.n_inputs):
            weight_update = learning_rates[i] * error * sigmoid_deriv * inputs[i]
            self.weights[i] += weight_update

        self.bias += self.eta * error * sigmoid_deriv

        # Decay causal evidence slowly to forget outdated beliefs
        if self.updates % 100 == 0:
            self.causal_evidence *= 0.98

    def predict(self, inputs):
        """Alias for forward"""
        return self.forward(inputs)

    def get_causal_interpretation(self):
        """Return causal analysis"""
        feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]

        analysis = {
            "causal_evidence": self.causal_evidence.copy(),
            "normal_correlations": self.normal_correlations.copy(),
            "intervention_correlations": self.intervention_correlations.copy(),
            "intervention_tests": self.intervention_tests,
            "total_updates": self.updates,
        }

        # Feature importance under normal conditions
        if np.sum(self.feature_activation_count) > 0:
            normal_importance = self.feature_success_count / np.maximum(
                self.feature_activation_count, 1
            )
            analysis["normal_importance"] = normal_importance

        # Causal vs correlational classification
        analysis["feature_classification"] = []
        for i, name in enumerate(feature_names):
            if self.causal_evidence[i] > 0.3:
                classification = "CAUSAL"
            elif self.normal_correlations[i] > 0.3:
                classification = "CORRELATIONAL"
            else:
                classification = "WEAK"
            analysis["feature_classification"].append(f"{name}: {classification}")

        return analysis
