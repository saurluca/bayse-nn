import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, steepness=1.0):
    """Sigmoid with controllable steepness"""
    return 1 / (1 + np.exp(-steepness * np.clip(x, -500, 500)))


def plateau_potential(x, threshold=0.5, duration=50):
    """Generate plateau potential when threshold is exceeded"""
    if x > threshold:
        return np.ones(duration) * 0.8
    else:
        return np.zeros(duration)


class CompartmentalizedCausalNeuron:
    """
    Model based on active dendrites research - different dendritic compartments
    can implement different computational functions
    """

    def __init__(self, n_inputs, n_compartments=3, eta=0.1):
        self.n_inputs = n_inputs
        self.n_compartments = n_compartments
        self.eta = eta

        # Each compartment has its own weights and properties
        self.compartment_weights = [
            np.random.randn(n_inputs) * 0.2 for _ in range(n_compartments)
        ]

        # Compartment-specific thresholds for dendritic spikes
        self.spike_thresholds = [0.3, 0.5, 0.7]

        # Inter-compartment coupling weights
        self.coupling_weights = np.random.randn(n_compartments, n_compartments) * 0.1

        # Causal evidence per compartment
        self.causal_evidence = [np.zeros(n_inputs) for _ in range(n_compartments)]

        # Plateau potential states
        self.plateau_states = np.zeros(n_compartments)

        self.bias = np.random.randn() * 0.1

    def dendritic_integration(self, inputs, compartment_id):
        """Each compartment can have different integration properties"""
        weighted_sum = np.dot(self.compartment_weights[compartment_id], inputs)

        # Different compartments use different nonlinearities
        if compartment_id == 0:  # Perisomatic - linear integration
            return weighted_sum
        elif compartment_id == 1:  # Basal dendrites - supralinear (NMDA-like)
            return (
                weighted_sum
                + 0.3 * sigmoid(weighted_sum - 0.3, steepness=5) * weighted_sum
            )
        else:  # Apical dendrites - plateau potentials
            if weighted_sum > self.spike_thresholds[compartment_id]:
                self.plateau_states[compartment_id] = 0.8  # Plateau potential
                return weighted_sum + 1.0  # Strong amplification
            else:
                self.plateau_states[compartment_id] *= 0.95  # Decay
                return weighted_sum

    def forward(self, inputs):
        """Forward pass with compartmentalized processing"""
        compartment_outputs = []

        for comp_id in range(self.n_compartments):
            comp_output = self.dendritic_integration(inputs, comp_id)
            compartment_outputs.append(comp_output)

        # Inter-compartment interactions
        coupled_outputs = []
        for i, output in enumerate(compartment_outputs):
            coupling_effect = np.sum(
                [
                    self.coupling_weights[i, j] * compartment_outputs[j]
                    for j in range(self.n_compartments)
                    if i != j
                ]
            )
            coupled_outputs.append(output + coupling_effect)

        # Final somatic integration
        somatic_input = np.sum(coupled_outputs) + self.bias
        return sigmoid(somatic_input)

    def test_causal_interventions(self, inputs, target):
        """Test causal effects using compartment-specific interventions"""
        original_output = self.forward(inputs)
        causal_scores = [np.zeros(self.n_inputs) for _ in range(self.n_compartments)]

        for comp_id in range(self.n_compartments):
            for input_idx in range(self.n_inputs):
                intervention_effects = []

                for intervention_val in [0, 1]:
                    if inputs[input_idx] == intervention_val:
                        continue

                    # Compartment-specific intervention
                    modified_inputs = inputs.copy()
                    modified_inputs[input_idx] = intervention_val

                    # Only affect specific compartment
                    temp_weights = self.compartment_weights[comp_id].copy()
                    self.compartment_weights[comp_id][input_idx] *= (
                        0.1  # Simulate intervention
                    )

                    intervened_output = self.forward(modified_inputs)

                    # Restore weights
                    self.compartment_weights[comp_id] = temp_weights

                    # Calculate prediction improvement
                    original_error = (target - original_output) ** 2
                    intervened_error = (target - intervened_output) ** 2
                    intervention_effects.append(original_error - intervened_error)

                if intervention_effects:
                    causal_scores[comp_id][input_idx] = np.mean(intervention_effects)

        return causal_scores

    def update_compartmentalized_learning(self, inputs, target):
        """Update each compartment based on its causal evidence"""
        causal_scores = self.test_causal_interventions(inputs, target)
        prediction_error = target - self.forward(inputs)

        for comp_id in range(self.n_compartments):
            # Update causal evidence with momentum
            alpha = 0.05 if comp_id == 2 else 0.1  # Apical dendrites learn slower
            self.causal_evidence[comp_id] = (
                0.9 * self.causal_evidence[comp_id] + alpha * causal_scores[comp_id]
            )

            # Update weights based on compartment-specific rules
            for i in range(self.n_inputs):
                if self.causal_evidence[comp_id][i] > 0.02:
                    # Standard learning for causal inputs
                    weight_update = self.eta * prediction_error * inputs[i]

                    # Compartment-specific modulation
                    if comp_id == 2 and self.plateau_states[comp_id] > 0.5:
                        weight_update *= 2.0  # Plateau-enhanced learning

                    self.compartment_weights[comp_id][i] += weight_update


class HierarchicalCausalNeuron:
    """
    Inspired by cortical layer structure - different layers for different
    causal relationships (direct vs indirect, local vs global)
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Layer 1: Direct causal relationships (basal dendrites)
        self.direct_weights = np.random.randn(n_inputs) * 0.2

        # Layer 2: Indirect/contextual relationships (apical dendrites)
        self.context_weights = np.random.randn(n_inputs) * 0.2

        # Layer 3: Confound detection (dendritic inhibition)
        self.confound_weights = np.random.randn(n_inputs) * 0.2

        # Causal relationship types
        self.direct_evidence = np.zeros(n_inputs)
        self.context_evidence = np.zeros(n_inputs)
        self.confound_evidence = np.zeros(n_inputs)

        # Gating mechanisms
        self.context_gate = 0.0
        self.confound_gate = 0.0

        self.bias = np.random.randn() * 0.1

    def forward(self, inputs, context_signal=None):
        """Hierarchical processing with context-dependent gating"""
        # Direct pathway (always active)
        direct_output = sigmoid(np.dot(self.direct_weights, inputs))

        # Context pathway (gated by attention/context)
        context_input = np.dot(self.context_weights, inputs)
        if context_signal is not None:
            self.context_gate = sigmoid(context_signal)
        context_output = self.context_gate * sigmoid(context_input)

        # Confound detection (inhibitory)
        confound_input = np.dot(self.confound_weights, inputs)
        confound_output = sigmoid(confound_input)
        self.confound_gate = confound_output

        # Hierarchical integration with inhibition
        total_excitation = direct_output + context_output
        inhibited_output = total_excitation / (1 + 0.5 * confound_output)

        return sigmoid(inhibited_output + self.bias)

    def test_hierarchical_causality(self, inputs, target, context_signal=None):
        """Test causality at different hierarchical levels"""
        original_output = self.forward(inputs, context_signal)

        direct_scores = np.zeros(self.n_inputs)
        context_scores = np.zeros(self.n_inputs)
        confound_scores = np.zeros(self.n_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                # Test direct causality
                temp_context = self.context_weights.copy()
                temp_confound = self.confound_weights.copy()
                self.context_weights *= 0  # Isolate direct pathway
                self.confound_weights *= 0

                direct_intervened = self.forward(modified_inputs, context_signal)
                direct_effect = abs(direct_intervened - original_output)
                direct_scores[i] += direct_effect

                # Restore and test context causality
                self.context_weights = temp_context
                self.confound_weights *= 0  # Isolate context pathway

                context_intervened = self.forward(modified_inputs, context_signal)
                context_effect = abs(context_intervened - direct_intervened)
                context_scores[i] += context_effect

                # Restore and test confound detection
                self.confound_weights = temp_confound
                full_intervened = self.forward(modified_inputs, context_signal)
                confound_effect = abs(full_intervened - context_intervened)
                confound_scores[i] += confound_effect

        return direct_scores, context_scores, confound_scores

    def update_hierarchical_learning(self, inputs, target, context_signal=None):
        """Update different hierarchical levels based on their causal contributions"""
        direct_scores, context_scores, confound_scores = (
            self.test_hierarchical_causality(inputs, target, context_signal)
        )

        # Update evidence with different time constants
        self.direct_evidence = 0.9 * self.direct_evidence + 0.1 * direct_scores
        self.context_evidence = 0.95 * self.context_evidence + 0.05 * context_scores
        self.confound_evidence = 0.85 * self.confound_evidence + 0.15 * confound_scores

        prediction_error = target - self.forward(inputs, context_signal)

        # Update direct weights (fast learning)
        for i in range(self.n_inputs):
            if self.direct_evidence[i] > 0.05:
                self.direct_weights[i] += self.eta * prediction_error * inputs[i]

        # Update context weights (slower, more selective)
        for i in range(self.n_inputs):
            if self.context_evidence[i] > 0.03 and self.context_gate > 0.3:
                self.context_weights[i] += 0.5 * self.eta * prediction_error * inputs[i]

        # Update confound weights (inhibitory learning)
        for i in range(self.n_inputs):
            if self.confound_evidence[i] > 0.02:
                # Strengthen weights that help detect confounds
                self.confound_weights[i] += (
                    0.3 * self.eta * prediction_error * inputs[i]
                )


class TemporalCausalNeuron:
    """
    Based on dendritic bistability and temporal integration research.
    Uses calcium dynamics and temporal credit assignment.
    """

    def __init__(self, n_inputs, eta=0.1, calcium_decay=0.95):
        self.n_inputs = n_inputs
        self.eta = eta
        self.calcium_decay = calcium_decay

        self.weights = np.random.randn(n_inputs) * 0.2
        self.bias = np.random.randn() * 0.1

        # Calcium-based eligibility traces (inspired by dendritic calcium)
        self.calcium_traces = np.zeros(n_inputs)
        self.eligibility_traces = np.zeros(n_inputs)

        # Temporal causal evidence
        self.temporal_evidence = np.zeros(n_inputs)

        # Plateau potential indicator
        self.plateau_amplitude = 0.0
        self.plateau_duration = 0

        # History for temporal analysis
        self.input_history = []
        self.output_history = []
        self.target_history = []

    def forward(self, inputs):
        """Forward pass with calcium dynamics"""
        # Update calcium traces (slow integration)
        self.calcium_traces = (
            self.calcium_decay * self.calcium_traces + (1 - self.calcium_decay) * inputs
        )

        # Standard weighted sum
        weighted_sum = np.dot(self.weights, inputs) + self.bias

        # Plateau potential generation (calcium-dependent)
        calcium_threshold = 0.4
        if np.mean(self.calcium_traces) > calcium_threshold:
            self.plateau_amplitude = 1.0
            self.plateau_duration = 20  # Time steps
            weighted_sum += 0.5  # Plateau boost
        else:
            if self.plateau_duration > 0:
                self.plateau_duration -= 1
                weighted_sum += 0.5 * (self.plateau_duration / 20)
            self.plateau_amplitude *= 0.9

        output = sigmoid(weighted_sum)

        # Update eligibility traces (faster than calcium)
        self.eligibility_traces = 0.8 * self.eligibility_traces + 0.2 * inputs

        return output

    def temporal_causal_analysis(self, window_size=10):
        """Analyze temporal relationships between inputs and delayed outcomes"""
        if len(self.input_history) < window_size:
            return np.zeros(self.n_inputs)

        temporal_scores = np.zeros(self.n_inputs)

        for delay in range(1, window_size):
            if delay >= len(self.target_history):
                continue

            # Look at inputs at time t and outcomes at t+delay
            for t in range(len(self.input_history) - delay):
                if t + delay >= len(self.target_history):
                    continue

                past_inputs = self.input_history[t]
                future_target = self.target_history[t + delay]

                # Measure how well past inputs predict future outcomes
                for i in range(self.n_inputs):
                    if past_inputs[i] > 0.5:  # Input was active
                        # Weight by temporal distance (closer = more causal)
                        temporal_weight = 1.0 / (1.0 + delay * 0.2)
                        temporal_scores[i] += temporal_weight * future_target

        return temporal_scores / window_size

    def update_temporal_learning(self, inputs, target):
        """Learning with temporal credit assignment"""
        output = self.forward(inputs)

        # Store history
        self.input_history.append(inputs.copy())
        self.output_history.append(output)
        self.target_history.append(target)

        # Keep history manageable
        max_history = 50
        if len(self.input_history) > max_history:
            self.input_history.pop(0)
            self.output_history.pop(0)
            self.target_history.pop(0)

        # Temporal causal analysis
        temporal_scores = self.temporal_causal_analysis()

        # Update temporal evidence
        self.temporal_evidence = 0.9 * self.temporal_evidence + 0.1 * temporal_scores

        # Standard prediction error
        prediction_error = target - output

        # Update weights using eligibility traces and temporal evidence
        for i in range(self.n_inputs):
            # Combine immediate eligibility with temporal evidence
            learning_signal = (
                0.7 * self.eligibility_traces[i] + 0.3 * self.temporal_evidence[i]
            )

            # Plateau-enhanced learning
            if self.plateau_amplitude > 0.3:
                learning_signal *= 1.5

            weight_update = self.eta * prediction_error * learning_signal
            self.weights[i] += weight_update


class MultiModalCausalNeuron:
    """
    Inspired by multi-compartment models with different dendritic domains
    processing different types of information (feedforward vs feedback)
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Feedforward pathway (basal dendrites)
        self.ff_weights = np.random.randn(n_inputs) * 0.2

        # Feedback pathway (apical dendrites)
        self.fb_weights = np.random.randn(n_inputs) * 0.2

        # Cross-modal integration weights
        self.integration_weights = np.random.randn(2) * 0.3  # [ff, fb]

        # Modal-specific causal evidence
        self.ff_evidence = np.zeros(n_inputs)
        self.fb_evidence = np.zeros(n_inputs)
        self.integration_evidence = 0.0

        # Attention/gating mechanisms
        self.ff_attention = 1.0
        self.fb_attention = 1.0

        self.bias = np.random.randn() * 0.1

    def forward(self, inputs, feedback_inputs=None):
        """Multi-modal processing with attention gating"""
        # Feedforward processing
        ff_output = self.ff_attention * sigmoid(np.dot(self.ff_weights, inputs))

        # Feedback processing (can be same inputs or different context)
        if feedback_inputs is None:
            feedback_inputs = inputs  # Self-referential feedback

        fb_output = self.fb_attention * sigmoid(
            np.dot(self.fb_weights, feedback_inputs)
        )

        # Multi-modal integration
        modal_outputs = np.array([ff_output, fb_output])
        integrated = np.dot(self.integration_weights, modal_outputs)

        return sigmoid(integrated + self.bias)

    def test_multimodal_causality(self, inputs, target, feedback_inputs=None):
        """Test causality across different modalities"""
        if feedback_inputs is None:
            feedback_inputs = inputs

        original_output = self.forward(inputs, feedback_inputs)

        ff_scores = np.zeros(self.n_inputs)
        fb_scores = np.zeros(self.n_inputs)

        # Test feedforward causality
        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                # Test with feedback disabled
                temp_fb_attention = self.fb_attention
                self.fb_attention = 0.0

                ff_intervened = self.forward(modified_inputs, feedback_inputs)
                ff_effect = abs(ff_intervened - original_output)
                ff_scores[i] += ff_effect

                self.fb_attention = temp_fb_attention

        # Test feedback causality
        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if feedback_inputs[i] == intervention_val:
                    continue

                modified_feedback = feedback_inputs.copy()
                modified_feedback[i] = intervention_val

                # Test with feedforward disabled
                temp_ff_attention = self.ff_attention
                self.ff_attention = 0.0

                fb_intervened = self.forward(inputs, modified_feedback)
                fb_effect = abs(fb_intervened - original_output)
                fb_scores[i] += fb_effect

                self.ff_attention = temp_ff_attention

        return ff_scores, fb_scores

    def update_multimodal_learning(self, inputs, target, feedback_inputs=None):
        """Update based on multi-modal causal analysis"""
        if feedback_inputs is None:
            feedback_inputs = inputs

        ff_scores, fb_scores = self.test_multimodal_causality(
            inputs, target, feedback_inputs
        )

        # Update modal evidence
        self.ff_evidence = 0.9 * self.ff_evidence + 0.1 * ff_scores
        self.fb_evidence = 0.9 * self.fb_evidence + 0.1 * fb_scores

        prediction_error = target - self.forward(inputs, feedback_inputs)

        # Update feedforward weights
        for i in range(self.n_inputs):
            if self.ff_evidence[i] > 0.05:
                self.ff_weights[i] += self.eta * prediction_error * inputs[i]

        # Update feedback weights
        for i in range(self.n_inputs):
            if self.fb_evidence[i] > 0.05:
                self.fb_weights[i] += self.eta * prediction_error * feedback_inputs[i]

        # Update integration weights based on modal effectiveness
        ff_effectiveness = np.mean(self.ff_evidence)
        fb_effectiveness = np.mean(self.fb_evidence)

        self.integration_weights[0] += (
            0.1 * self.eta * prediction_error * ff_effectiveness
        )
        self.integration_weights[1] += (
            0.1 * self.eta * prediction_error * fb_effectiveness
        )

        # Update attention weights
        if ff_effectiveness > fb_effectiveness:
            self.ff_attention = min(1.0, self.ff_attention + 0.01)
            self.fb_attention = max(0.1, self.fb_attention - 0.01)
        else:
            self.fb_attention = min(1.0, self.fb_attention + 0.01)
            self.ff_attention = max(0.1, self.ff_attention - 0.01)


def generate_enhanced_cancer_data(n_samples=1000, complexity_level="medium"):
    """Generate cancer data with different complexity levels for testing"""
    data = []

    for _ in range(n_samples):
        # Core causal factors
        tumor_marker = np.random.random()
        genetic_risk = np.random.random()

        if complexity_level == "simple":
            # Simple additive causality
            cancer_prob = 0.1 + 0.6 * tumor_marker + 0.4 * genetic_risk

        elif complexity_level == "medium":
            # Interaction effects
            interaction = tumor_marker * genetic_risk
            cancer_prob = (
                0.1 + 0.4 * tumor_marker + 0.3 * genetic_risk + 0.3 * interaction
            )

        elif complexity_level == "complex":
            # Non-linear threshold effects
            if tumor_marker > 0.7 and genetic_risk > 0.6:
                cancer_prob = 0.9  # High risk threshold
            elif tumor_marker > 0.5 or genetic_risk > 0.7:
                cancer_prob = 0.4 + 0.3 * tumor_marker + 0.2 * genetic_risk
            else:
                cancer_prob = 0.1 + 0.2 * tumor_marker + 0.1 * genetic_risk

        cancer_prob = np.clip(cancer_prob, 0.05, 0.95)
        has_cancer = 1 if np.random.random() < cancer_prob else 0

        # Confounding factors
        age_factor = 0.3 + 0.4 * has_cancer + np.random.normal(0, 0.1)
        age_factor = np.clip(age_factor, 0, 1)

        lifestyle_factor = 0.2 + 0.5 * has_cancer + np.random.normal(0, 0.15)
        lifestyle_factor = np.clip(lifestyle_factor, 0, 1)

        # Context signals (for hierarchical model)
        stress_context = np.random.random()
        screening_context = np.random.random()

        # Convert to binary
        inputs = np.array(
            [
                1 if tumor_marker > 0.5 else 0,
                1 if genetic_risk > 0.5 else 0,
                1 if age_factor > 0.6 else 0,
                1 if lifestyle_factor > 0.6 else 0,
                1 if np.random.random() > 0.5 else 0,  # Noise
            ]
        )

        context = np.array([stress_context, screening_context])

        data.append((inputs, has_cancer, context))

    return data


def test_dendritic_models():
    """Comprehensive test of all dendritic causal models"""
    print("=== Testing Advanced Dendritic Causal Learning Models ===\n")

    # Generate test data
    train_data = generate_enhanced_cancer_data(2000, "complex")
    test_data = generate_enhanced_cancer_data(500, "complex")

    models = {
        "Compartmentalized": CompartmentalizedCausalNeuron(5, n_compartments=3),
        "Hierarchical": HierarchicalCausalNeuron(5),
        "Temporal": TemporalCausalNeuron(5),
        "MultiModal": MultiModalCausalNeuron(5),
    }

    feature_names = ["Tumor", "Genetic", "Age", "Lifestyle", "Noise"]
    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} Model...")

        # Training
        for epoch in range(20):
            np.random.shuffle(train_data)

            for inputs, target, context in train_data:
                if model_name == "Hierarchical":
                    context_signal = np.mean(context)
                    model.update_hierarchical_learning(inputs, target, context_signal)
                elif model_name == "Temporal":
                    model.update_temporal_learning(inputs, target)
                elif model_name == "MultiModal":
                    # Use context as feedback inputs
                    feedback = np.concatenate(
                        [inputs[:2], context, [0]]
                    )  # Focus on causal + context
                    model.update_multimodal_learning(inputs, target, feedback)
                else:  # Compartmentalized
                    model.update_compartmentalized_learning(inputs, target)

        # Testing
        correct = 0
        for inputs, target, context in test_data:
            if model_name == "Hierarchical":
                context_signal = np.mean(context)
                prediction = model.forward(inputs, context_signal)
            elif model_name == "MultiModal":
                feedback = np.concatenate([inputs[:2], context, [0]])
                prediction = model.forward(inputs, feedback)
            else:
                prediction = model.forward(inputs)

            if (prediction > 0.5) == target:
                correct += 1

        accuracy = correct / len(test_data)
        results[model_name] = {"accuracy": accuracy}

        # Extract causal evidence based on model type
        print(f"\n{model_name} Model Results:")
        print(f"Accuracy: {accuracy:.3f}")

        if model_name == "Compartmentalized":
            print("Causal Evidence by Compartment:")
            for comp_id in range(model.n_compartments):
                comp_name = ["Perisomatic", "Basal", "Apical"][comp_id]
                print(f"  {comp_name}: {model.causal_evidence[comp_id]}")

        elif model_name == "Hierarchical":
            print("Hierarchical Causal Evidence:")
            print(f"  Direct: {model.direct_evidence}")
            print(f"  Context: {model.context_evidence}")
            print(f"  Confound: {model.confound_evidence}")

        elif model_name == "Temporal":
            print("Temporal Causal Evidence:")
            print(f"  Evidence: {model.temporal_evidence}")
            print(f"  Calcium traces: {model.calcium_traces}")

        elif model_name == "MultiModal":
            print("Multi-Modal Causal Evidence:")
            print(f"  Feedforward: {model.ff_evidence}")
            print(f"  Feedback: {model.fb_evidence}")
            print(
                f"  Attention weights: FF={model.ff_attention:.3f}, FB={model.fb_attention:.3f}"
            )

        print()

    # Comparison analysis
    print("=== Model Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.3f}")

    return results, models


def visualize_dendritic_mechanisms(models):
    """Visualize the different dendritic mechanisms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    feature_names = ["Tumor", "Genetic", "Age", "Lifestyle", "Noise"]

    # Compartmentalized model
    ax = axes[0]
    comp_model = models["Compartmentalized"]
    evidence_matrix = np.array(comp_model.causal_evidence)
    im = ax.imshow(evidence_matrix, cmap="RdYlBu_r", aspect="auto")
    ax.set_title("Compartmentalized Model\nCausal Evidence by Compartment")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Dendritic Compartments")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Perisomatic", "Basal", "Apical"])
    plt.colorbar(im, ax=ax)

    # Hierarchical model
    ax = axes[1]
    hier_model = models["Hierarchical"]
    hier_data = np.array(
        [
            hier_model.direct_evidence,
            hier_model.context_evidence,
            hier_model.confound_evidence,
        ]
    )
    im = ax.imshow(hier_data, cmap="RdYlBu_r", aspect="auto")
    ax.set_title("Hierarchical Model\nCausal Evidence by Level")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Hierarchical Levels")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Direct", "Context", "Confound"])
    plt.colorbar(im, ax=ax)

    # Temporal model
    ax = axes[2]
    temp_model = models["Temporal"]
    temporal_data = np.array(
        [
            temp_model.temporal_evidence,
            temp_model.calcium_traces,
            temp_model.eligibility_traces,
        ]
    )
    im = ax.imshow(temporal_data, cmap="RdYlBu_r", aspect="auto")
    ax.set_title("Temporal Model\nTemporal Traces")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Trace Types")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Temporal Evidence", "Calcium Traces", "Eligibility Traces"])
    plt.colorbar(im, ax=ax)

    # MultiModal model
    ax = axes[3]
    mm_model = models["MultiModal"]
    modal_data = np.array([mm_model.ff_evidence, mm_model.fb_evidence])
    im = ax.imshow(modal_data, cmap="RdYlBu_r", aspect="auto")
    ax.set_title("Multi-Modal Model\nEvidence by Pathway")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Processing Pathways")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["Feedforward", "Feedback"])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("dendritic_mechanisms_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results, models = test_dendritic_models()
    visualize_dendritic_mechanisms(models)
