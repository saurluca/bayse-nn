import numpy as np


def sigmoid(x):
    """Safe sigmoid function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


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

    def update_active_dendrite_learning(self, inputs, target):
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


class ContextGatedNeuron:
    """
    Model based on apical dendrite context gating research.
    Uses separate pathways for bottom-up sensory input and top-down context,
    with context gating the integration of sensory information.
    """

    def __init__(self, n_inputs, n_context=2, eta=0.1):
        self.n_inputs = n_inputs
        self.n_context = n_context
        self.eta = eta

        # Bottom-up pathway (basal dendrites)
        self.bottomup_weights = np.random.randn(n_inputs) * 0.3

        # Top-down context pathway (apical dendrites)
        self.context_weights = np.random.randn(n_context) * 0.2

        # Context-input interaction weights
        self.interaction_weights = np.random.randn(n_inputs, n_context) * 0.1

        # Context gate parameters
        self.context_threshold = 0.3
        self.gate_strength = 0.0

        # Causal evidence tracking
        self.bottomup_evidence = np.zeros(n_inputs)
        self.context_evidence = np.zeros(n_context)
        self.interaction_evidence = np.zeros((n_inputs, n_context))

        self.bias = np.random.randn() * 0.1

    def compute_context_gate(self, context_inputs):
        """Compute context-dependent gating signal"""
        context_activation = np.dot(self.context_weights, context_inputs)

        # Sigmoid gating with threshold
        if context_activation > self.context_threshold:
            self.gate_strength = sigmoid(context_activation - self.context_threshold)
        else:
            self.gate_strength = 0.1  # Minimal gating when context is weak

        return self.gate_strength

    def forward(self, inputs, context_inputs=None):
        """Forward pass with context gating"""
        if context_inputs is None:
            # Use subset of inputs as context
            context_inputs = inputs[: self.n_context]

        # Bottom-up processing
        bottomup_activation = np.dot(self.bottomup_weights, inputs)

        # Context gating
        gate_strength = self.compute_context_gate(context_inputs)

        # Context-input interactions
        interaction_effects = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            for j in range(self.n_context):
                interaction_effects[i] += (
                    self.interaction_weights[i, j] * inputs[i] * context_inputs[j]
                )

        # Gated integration
        gated_bottomup = gate_strength * bottomup_activation
        gated_interactions = gate_strength * np.sum(interaction_effects)

        total_activation = gated_bottomup + gated_interactions + self.bias

        return sigmoid(total_activation)

    def test_context_gated_causality(self, inputs, target, context_inputs=None):
        """Test causality with and without context gating"""
        if context_inputs is None:
            context_inputs = inputs[: self.n_context]

        # Test bottom-up causality (context disabled)
        bottomup_scores = np.zeros(self.n_inputs)

        # Temporarily disable context
        original_gate = self.gate_strength
        self.gate_strength = 0.1

        baseline_output = self.forward(inputs, context_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                intervened_output = self.forward(modified_inputs, context_inputs)
                bottomup_scores[i] += abs(intervened_output - baseline_output)

        # Test context causality
        self.gate_strength = original_gate
        context_scores = np.zeros(self.n_context)

        gated_baseline = self.forward(inputs, context_inputs)

        for i in range(self.n_context):
            for intervention_val in [0, 1]:
                if context_inputs[i] == intervention_val:
                    continue

                modified_context = context_inputs.copy()
                modified_context[i] = intervention_val

                intervened_output = self.forward(inputs, modified_context)
                context_scores[i] += abs(intervened_output - gated_baseline)

        # Test interactions
        interaction_scores = np.zeros((self.n_inputs, self.n_context))

        for i in range(self.n_inputs):
            for j in range(self.n_context):
                # Test interaction by changing both simultaneously
                for input_val, context_val in [(0, 0), (1, 1)]:
                    if inputs[i] == input_val or context_inputs[j] == context_val:
                        continue

                    modified_inputs = inputs.copy()
                    modified_context = context_inputs.copy()
                    modified_inputs[i] = input_val
                    modified_context[j] = context_val

                    # Compare to sum of individual effects
                    individual_input = inputs.copy()
                    individual_input[i] = input_val
                    input_effect = abs(
                        self.forward(individual_input, context_inputs) - gated_baseline
                    )

                    individual_context = context_inputs.copy()
                    individual_context[j] = context_val
                    context_effect = abs(
                        self.forward(inputs, individual_context) - gated_baseline
                    )

                    joint_effect = abs(
                        self.forward(modified_inputs, modified_context) - gated_baseline
                    )

                    # Interaction is effect beyond sum of individual effects
                    interaction_scores[i, j] += max(
                        0, joint_effect - input_effect - context_effect
                    )

        return bottomup_scores, context_scores, interaction_scores

    def update_context_gated_learning(self, inputs, target, context_inputs=None):
        """Update with context-gated learning rules"""
        if context_inputs is None:
            context_inputs = inputs[: self.n_context]

        bottomup_scores, context_scores, interaction_scores = (
            self.test_context_gated_causality(inputs, target, context_inputs)
        )

        # Update evidence
        self.bottomup_evidence = 0.9 * self.bottomup_evidence + 0.1 * bottomup_scores
        self.context_evidence = 0.9 * self.context_evidence + 0.1 * context_scores
        self.interaction_evidence = (
            0.9 * self.interaction_evidence + 0.1 * interaction_scores
        )

        prediction_error = target - self.forward(inputs, context_inputs)

        # Update bottom-up weights (gating-dependent)
        for i in range(self.n_inputs):
            if self.bottomup_evidence[i] > 0.05:
                gating_boost = 1.0 + self.gate_strength
                weight_update = self.eta * gating_boost * prediction_error * inputs[i]
                self.bottomup_weights[i] += weight_update

        # Update context weights
        for i in range(self.n_context):
            if self.context_evidence[i] > 0.03:
                weight_update = self.eta * prediction_error * context_inputs[i]
                self.context_weights[i] += weight_update

        # Update interaction weights
        for i in range(self.n_inputs):
            for j in range(self.n_context):
                if self.interaction_evidence[i, j] > 0.02:
                    interaction_update = (
                        0.5
                        * self.eta
                        * prediction_error
                        * inputs[i]
                        * context_inputs[j]
                    )
                    self.interaction_weights[i, j] += interaction_update


class TemporalDendriteNeuron:
    """
    Model based on temporal dynamics in dendritic integration.
    Uses different time constants for different dendritic compartments
    to capture temporal causal relationships.
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Fast pathway (proximal dendrites, ~10ms)
        self.fast_weights = np.random.randn(n_inputs) * 0.3
        self.fast_trace = np.zeros(n_inputs)
        self.fast_decay = 0.5

        # Medium pathway (mid dendrites, ~100ms)
        self.medium_weights = np.random.randn(n_inputs) * 0.2
        self.medium_trace = np.zeros(n_inputs)
        self.medium_decay = 0.8

        # Slow pathway (distal dendrites, ~1000ms)
        self.slow_weights = np.random.randn(n_inputs) * 0.2
        self.slow_trace = np.zeros(n_inputs)
        self.slow_decay = 0.95

        # Temporal causal evidence
        self.fast_evidence = np.zeros(n_inputs)
        self.medium_evidence = np.zeros(n_inputs)
        self.slow_evidence = np.zeros(n_inputs)

        # History for temporal credit assignment
        self.history_inputs = []
        self.history_targets = []
        self.max_history = 20

        self.bias = np.random.randn() * 0.1

    def update_temporal_traces(self, inputs):
        """Update temporal traces with different decay rates"""
        self.fast_trace = (
            self.fast_decay * self.fast_trace + (1 - self.fast_decay) * inputs
        )
        self.medium_trace = (
            self.medium_decay * self.medium_trace + (1 - self.medium_decay) * inputs
        )
        self.slow_trace = (
            self.slow_decay * self.slow_trace + (1 - self.slow_decay) * inputs
        )

    def forward(self, inputs):
        """Forward pass with temporal integration"""
        self.update_temporal_traces(inputs)

        # Compute pathway outputs
        fast_output = sigmoid(np.dot(self.fast_weights, self.fast_trace))
        medium_output = sigmoid(np.dot(self.medium_weights, self.medium_trace))
        slow_output = sigmoid(np.dot(self.slow_weights, self.slow_trace))

        # Temporal hierarchy: fast modulates medium, medium modulates slow
        medium_modulated = medium_output * (1 + 0.5 * fast_output)
        slow_modulated = slow_output * (1 + 0.3 * medium_modulated)

        # Integration
        total_activation = fast_output + medium_modulated + slow_modulated + self.bias

        return sigmoid(total_activation)

    def test_temporal_causality(self, inputs, target):
        """Test causality across different temporal scales"""
        # Test each pathway in isolation
        fast_scores = np.zeros(self.n_inputs)
        medium_scores = np.zeros(self.n_inputs)
        slow_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                # Test fast pathway (disable others)
                temp_medium = self.medium_weights.copy()
                temp_slow = self.slow_weights.copy()
                self.medium_weights *= 0
                self.slow_weights *= 0

                baseline_fast = self.forward(inputs)
                intervened_fast = self.forward(modified_inputs)
                fast_scores[i] += abs(intervened_fast - baseline_fast)

                # Test medium pathway (disable slow)
                self.medium_weights = temp_medium
                self.slow_weights *= 0

                baseline_medium = self.forward(inputs)
                intervened_medium = self.forward(modified_inputs)
                medium_scores[i] += abs(intervened_medium - baseline_medium)

                # Test slow pathway (all enabled)
                self.slow_weights = temp_slow

                baseline_slow = self.forward(inputs)
                intervened_slow = self.forward(modified_inputs)
                slow_scores[i] += abs(intervened_slow - baseline_slow)

        return fast_scores, medium_scores, slow_scores

    def temporal_credit_assignment(self):
        """Assign credit based on temporal relationships in history"""
        if len(self.history_inputs) < 3:
            return (
                np.zeros(self.n_inputs),
                np.zeros(self.n_inputs),
                np.zeros(self.n_inputs),
            )

        fast_credit = np.zeros(self.n_inputs)
        medium_credit = np.zeros(self.n_inputs)
        slow_credit = np.zeros(self.n_inputs)

        # Look at different temporal delays
        for delay in range(1, min(len(self.history_inputs), self.max_history)):
            if delay >= len(self.history_targets):
                continue

            past_inputs = self.history_inputs[-(delay + 1)]
            current_target = self.history_targets[-1]

            # Weight by temporal distance
            if delay <= 2:  # Fast timescale
                weight = 1.0 / (1 + delay * 0.5)
                fast_credit += weight * past_inputs * current_target
            elif delay <= 8:  # Medium timescale
                weight = 1.0 / (1 + delay * 0.2)
                medium_credit += weight * past_inputs * current_target
            else:  # Slow timescale
                weight = 1.0 / (1 + delay * 0.1)
                slow_credit += weight * past_inputs * current_target

        return fast_credit, medium_credit, slow_credit

    def update_temporal_learning(self, inputs, target):
        """Update with temporal credit assignment"""
        # Store history
        self.history_inputs.append(inputs.copy())
        self.history_targets.append(target)

        if len(self.history_inputs) > self.max_history:
            self.history_inputs.pop(0)
            self.history_targets.pop(0)

        # Test immediate causality
        fast_scores, medium_scores, slow_scores = self.test_temporal_causality(
            inputs, target
        )

        # Temporal credit assignment
        fast_credit, medium_credit, slow_credit = self.temporal_credit_assignment()

        # Update evidence
        self.fast_evidence = 0.8 * self.fast_evidence + 0.2 * fast_scores
        self.medium_evidence = 0.9 * self.medium_evidence + 0.1 * medium_scores
        self.slow_evidence = 0.95 * self.slow_evidence + 0.05 * slow_scores

        prediction_error = target - self.forward(inputs)

        # Update weights with temporal credit
        for i in range(self.n_inputs):
            # Fast pathway
            if self.fast_evidence[i] > 0.05:
                fast_update = (
                    self.eta
                    * prediction_error
                    * (self.fast_trace[i] + 0.3 * fast_credit[i])
                )
                self.fast_weights[i] += fast_update

            # Medium pathway
            if self.medium_evidence[i] > 0.03:
                medium_update = (
                    self.eta
                    * prediction_error
                    * (self.medium_trace[i] + 0.3 * medium_credit[i])
                )
                self.medium_weights[i] += medium_update

            # Slow pathway
            if self.slow_evidence[i] > 0.02:
                slow_update = (
                    self.eta
                    * prediction_error
                    * (self.slow_trace[i] + 0.3 * slow_credit[i])
                )
                self.slow_weights[i] += slow_update


def generate_complex_temporal_data(n_samples=1000):
    """Generate data with temporal causal relationships"""
    data = []

    for _ in range(n_samples):
        # Immediate causes (fast)
        fast_cause1 = np.random.random() > 0.5
        fast_cause2 = np.random.random() > 0.5

        # Delayed causes (medium)
        medium_cause = np.random.random() > 0.5

        # Context (slow)
        slow_context = np.random.random() > 0.5

        # Effect depends on all timescales
        fast_contribution = 0.4 * (fast_cause1 or fast_cause2)
        medium_contribution = 0.3 * medium_cause
        slow_contribution = 0.2 * slow_context
        interaction = 0.1 * (fast_cause1 and medium_cause and slow_context)

        effect_prob = (
            0.1
            + fast_contribution
            + medium_contribution
            + slow_contribution
            + interaction
        )
        effect_prob = np.clip(effect_prob, 0.05, 0.95)

        effect = np.random.random() < effect_prob

        # Confounding factors
        confound = np.random.random() < (0.4 + 0.4 * effect)
        noise = np.random.random() > 0.5

        inputs = np.array(
            [fast_cause1, fast_cause2, medium_cause, slow_context, confound, noise],
            dtype=float,
        )

        context = np.array([slow_context, confound], dtype=float)

        data.append((inputs, effect, context))

    return data


def test_advanced_models():
    """Test all advanced dendritic models"""
    print("=== Testing Advanced Dendritic Models ===\n")

    # Generate complex data
    train_data = generate_complex_temporal_data(2000)
    test_data = generate_complex_temporal_data(500)

    models = {
        "Active Dendrites": ActiveDendriteNeuron(6, n_segments=6, segment_size=3),
        "Context Gated": ContextGatedNeuron(6, n_context=2),
        "Temporal Dendrites": TemporalDendriteNeuron(6),
    }

    feature_names = [
        "FastCause1",
        "FastCause2",
        "MediumCause",
        "SlowContext",
        "Confound",
        "Noise",
    ]
    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} Model...")

        # Training
        for epoch in range(25):
            np.random.shuffle(train_data)

            for inputs, target, context in train_data:
                if model_name == "Active Dendrites":
                    model.update_active_dendrite_learning(inputs, target)
                elif model_name == "Context Gated":
                    model.update_context_gated_learning(inputs, target, context)
                else:  # Temporal
                    model.update_temporal_learning(inputs, target)

        # Testing
        correct = 0
        for inputs, target, context in test_data:
            if model_name == "Active Dendrites":
                prediction = model.forward(inputs)
            elif model_name == "Context Gated":
                prediction = model.forward(inputs, context)
            else:  # Temporal
                prediction = model.forward(inputs)

            if (prediction > 0.5) == target:
                correct += 1

        accuracy = correct / len(test_data)
        results[model_name] = {"accuracy": accuracy}

        print(f"\n{model_name} Model Results:")
        print(f"Accuracy: {accuracy:.3f}")

        # Model-specific analysis
        if model_name == "Active Dendrites":
            print("Segment Activity:")
            print(
                f"  Active segments: {np.sum(model.active_segments > 0.1)}/{model.n_segments}"
            )
            print(f"  Somatic weights: {model.somatic_weights}")

        elif model_name == "Context Gated":
            print("Context Gating:")
            print(f"  Gate strength: {model.gate_strength:.3f}")
            print(f"  Bottom-up evidence: {model.bottomup_evidence}")
            print(f"  Context evidence: {model.context_evidence}")

        else:  # Temporal
            print("Temporal Evidence:")
            print(f"  Fast: {model.fast_evidence}")
            print(f"  Medium: {model.medium_evidence}")
            print(f"  Slow: {model.slow_evidence}")

        print()

    print("=== Model Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.3f}")

    return results, models


if __name__ == "__main__":
    results, models = test_advanced_models()
