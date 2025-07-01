import numpy as np
from utils import sigmoid


class ActiveDendriteNeuron:
    """
    Based on Hawkins et al. research on active dendrites.
    Each dendritic segment acts as a coincidence detector and can generate
    independent spikes that modulate somatic integration.
    """

    def __init__(self, n_inputs, n_segments=8, segment_size=3, eta=0.1):
        self.n_inputs = n_inputs
        self.n_segments = n_segments
        self.segment_size = segment_size
        self.eta = eta

        # Each segment connects to a subset of inputs
        self.segment_connections = []
        self.segment_weights = []

        for _ in range(n_segments):
            # Random subset of inputs for each segment
            connections = np.random.choice(n_inputs, segment_size, replace=False)
            weights = np.random.randn(segment_size) * 0.3

            self.segment_connections.append(connections)
            self.segment_weights.append(weights)

        # Segment activation thresholds (different for each segment)
        self.segment_thresholds = np.random.uniform(0.4, 0.8, n_segments)

        # Somatic weights (how much each segment contributes)
        self.somatic_weights = np.random.randn(n_segments) * 0.2

        # Segment-specific causal evidence
        self.segment_causal_evidence = [
            np.zeros(segment_size) for _ in range(n_segments)
        ]

        # Active segment tracking
        self.active_segments = np.zeros(n_segments)

        self.bias = np.random.randn() * 0.1

        # For compatibility with evaluation
        self.causal_evidence = np.zeros(n_inputs)

    def compute_segment_activation(self, inputs, segment_id):
        """Compute activation for a specific dendritic segment"""
        connections = self.segment_connections[segment_id]
        weights = self.segment_weights[segment_id]

        # Get input values for this segment
        segment_inputs = inputs[connections]

        # Weighted sum
        activation = np.dot(weights, segment_inputs)

        # Check if segment reaches threshold (dendritic spike)
        if activation > self.segment_thresholds[segment_id]:
            self.active_segments[segment_id] = 1.0
            # Non-linear amplification when threshold crossed
            return activation + 0.5  # Dendritic spike boost
        else:
            self.active_segments[segment_id] *= 0.9  # Decay
            return max(0, activation)  # Subthreshold

    def forward(self, inputs):
        """Forward pass with active dendritic segments"""
        segment_outputs = []

        for segment_id in range(self.n_segments):
            segment_output = self.compute_segment_activation(inputs, segment_id)
            segment_outputs.append(segment_output)

        # Somatic integration with segment-specific weights
        somatic_input = np.dot(self.somatic_weights, segment_outputs) + self.bias

        return sigmoid(somatic_input)

    def test_segment_causality(self, inputs, target):
        """Test causal contributions of each segment"""
        original_output = self.forward(inputs)

        segment_causal_scores = [
            np.zeros(self.segment_size) for _ in range(self.n_segments)
        ]

        for segment_id in range(self.n_segments):
            connections = self.segment_connections[segment_id]

            for local_idx in range(self.segment_size):
                global_idx = connections[local_idx]
                intervention_effects = []

                for intervention_val in [0, 1]:
                    if inputs[global_idx] == intervention_val:
                        continue

                    # Intervention on this specific input
                    modified_inputs = inputs.copy()
                    modified_inputs[global_idx] = intervention_val

                    intervened_output = self.forward(modified_inputs)
                    effect = abs(intervened_output - original_output)
                    intervention_effects.append(effect)

                if intervention_effects:
                    segment_causal_scores[segment_id][local_idx] = np.mean(
                        intervention_effects
                    )

        return segment_causal_scores

    def update(self, inputs, target):
        """Update using active dendrite learning rules"""
        segment_causal_scores = self.test_segment_causality(inputs, target)
        prediction_error = target - self.forward(inputs)

        for segment_id in range(self.n_segments):
            # Update segment causal evidence
            self.segment_causal_evidence[segment_id] = (
                0.9 * self.segment_causal_evidence[segment_id]
                + 0.1 * segment_causal_scores[segment_id]
            )

            # Activity-dependent learning
            if self.active_segments[segment_id] > 0.5:
                # Segment was active - strengthen based on causal evidence
                connections = self.segment_connections[segment_id]

                for local_idx in range(self.segment_size):
                    global_idx = connections[local_idx]

                    if self.segment_causal_evidence[segment_id][local_idx] > 0.05:
                        # Activity and causality dependent learning
                        activity_boost = 1.0 + self.active_segments[segment_id]
                        weight_update = (
                            self.eta
                            * activity_boost
                            * prediction_error
                            * inputs[global_idx]
                        )
                        self.segment_weights[segment_id][local_idx] += weight_update

        # Update somatic weights based on segment effectiveness
        for segment_id in range(self.n_segments):
            segment_effectiveness = np.mean(self.segment_causal_evidence[segment_id])
            if segment_effectiveness > 0.03:
                self.somatic_weights[segment_id] += (
                    0.1 * self.eta * prediction_error * segment_effectiveness
                )

        # Update causal evidence for compatibility
        self.causal_evidence.fill(0)
        for segment_id in range(self.n_segments):
            connections = self.segment_connections[segment_id]
            for local_idx, global_idx in enumerate(connections):
                self.causal_evidence[global_idx] += self.segment_causal_evidence[
                    segment_id
                ][local_idx]

    def predict(self, inputs):
        """Alias for forward for consistency"""
        return self.forward(inputs)

    @property
    def weights(self):
        """Return aggregate weights for compatibility"""
        # Aggregate weights across segments
        aggregate_weights = np.zeros(self.n_inputs)
        for segment_id in range(self.n_segments):
            connections = self.segment_connections[segment_id]
            for local_idx, global_idx in enumerate(connections):
                aggregate_weights[global_idx] += (
                    self.segment_weights[segment_id][local_idx]
                    * self.somatic_weights[segment_id]
                )
        return aggregate_weights
