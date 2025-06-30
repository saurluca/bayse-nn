import numpy as np


def sigmoid(x, steepness=1.0):
    """Safe sigmoid function"""
    return 1 / (1 + np.exp(-steepness * np.clip(x, -500, 500)))


class DendriticSpikingNeuron:
    """
    Model inspired by dendritic spike generation and calcium dynamics.
    Different dendritic branches can generate local spikes that influence learning.
    """

    def __init__(self, n_inputs, n_branches=4, eta=0.1):
        self.n_inputs = n_inputs
        self.n_branches = n_branches
        self.eta = eta

        # Weights for each dendritic branch
        self.branch_weights = [
            np.random.randn(n_inputs) * 0.3 for _ in range(n_branches)
        ]

        # Branch-specific spike thresholds
        self.spike_thresholds = np.random.uniform(0.3, 0.8, n_branches)

        # Calcium concentration in each branch (memory trace)
        self.calcium_levels = np.zeros(n_branches)

        # Branch importance weights (learned)
        self.branch_importance = np.ones(n_branches) / n_branches

        # Causal strength per input per branch
        self.causal_strength = [np.zeros(n_inputs) for _ in range(n_branches)]

        self.somatic_bias = np.random.randn() * 0.1

    def dendritic_processing(self, inputs, branch_id):
        """Process inputs in a specific dendritic branch"""
        branch_input = np.dot(self.branch_weights[branch_id], inputs)

        # Generate dendritic spike if threshold exceeded
        if branch_input > self.spike_thresholds[branch_id]:
            spike_amplitude = 1.5 * sigmoid(
                branch_input - self.spike_thresholds[branch_id]
            )

            # Increase calcium level (learning signal)
            self.calcium_levels[branch_id] = min(
                1.0, self.calcium_levels[branch_id] + 0.3
            )

            return branch_input + spike_amplitude
        else:
            # Calcium decay
            self.calcium_levels[branch_id] *= 0.95
            return branch_input

    def forward(self, inputs):
        """Forward pass with dendritic spike integration"""
        branch_outputs = []

        for branch_id in range(self.n_branches):
            branch_output = self.dendritic_processing(inputs, branch_id)
            branch_outputs.append(branch_output)

        # Weighted integration at soma
        somatic_input = (
            np.sum(
                [
                    self.branch_importance[i] * output
                    for i, output in enumerate(branch_outputs)
                ]
            )
            + self.somatic_bias
        )

        return sigmoid(somatic_input)

    def test_branch_causality(self, inputs, target):
        """Test causal contributions of each branch"""
        original_output = self.forward(inputs)
        branch_causal_scores = [np.zeros(self.n_inputs) for _ in range(self.n_branches)]

        for branch_id in range(self.n_branches):
            for input_idx in range(self.n_inputs):
                intervention_effects = []

                for intervention_val in [0, 1]:
                    if inputs[input_idx] == intervention_val:
                        continue

                    # Temporarily modify this branch's response to the input
                    modified_inputs = inputs.copy()
                    modified_inputs[input_idx] = intervention_val

                    # Selective branch intervention
                    original_weight = self.branch_weights[branch_id][input_idx]
                    self.branch_weights[branch_id][input_idx] = 0  # Silence this input

                    intervened_output = self.forward(modified_inputs)

                    # Restore weight
                    self.branch_weights[branch_id][input_idx] = original_weight

                    effect_strength = abs(intervened_output - original_output)
                    intervention_effects.append(effect_strength)

                if intervention_effects:
                    branch_causal_scores[branch_id][input_idx] = np.mean(
                        intervention_effects
                    )

        return branch_causal_scores

    def update_dendritic_learning(self, inputs, target):
        """Update based on dendritic spike-dependent plasticity"""
        branch_causal_scores = self.test_branch_causality(inputs, target)
        prediction_error = target - self.forward(inputs)

        for branch_id in range(self.n_branches):
            # Update causal strength estimates
            self.causal_strength[branch_id] = (
                0.9 * self.causal_strength[branch_id]
                + 0.1 * branch_causal_scores[branch_id]
            )

            # Calcium-dependent learning rate
            calcium_modulation = 1.0 + 2.0 * self.calcium_levels[branch_id]

            # Update branch weights
            for i in range(self.n_inputs):
                if self.causal_strength[branch_id][i] > 0.03:
                    weight_update = (
                        self.eta * calcium_modulation * prediction_error * inputs[i]
                    )
                    self.branch_weights[branch_id][i] += weight_update

            # Update branch importance based on overall effectiveness
            branch_effectiveness = np.mean(self.causal_strength[branch_id])
            self.branch_importance[branch_id] = max(
                0.1, self.branch_importance[branch_id] + 0.01 * branch_effectiveness
            )

        # Normalize branch importance
        self.branch_importance /= np.sum(self.branch_importance)


class PlateauPotentialNeuron:
    """
    Model based on plateau potentials in apical dendrites.
    Plateau potentials provide a mechanism for sustained learning signals.
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Basal dendritic weights (proximal)
        self.basal_weights = np.random.randn(n_inputs) * 0.2

        # Apical dendritic weights (distal)
        self.apical_weights = np.random.randn(n_inputs) * 0.2

        # Plateau potential state
        self.plateau_active = False
        self.plateau_duration = 0
        self.plateau_strength = 0.0

        # Context integration weights
        self.context_weights = np.random.randn(n_inputs) * 0.1

        # Causal evidence tracking
        self.basal_evidence = np.zeros(n_inputs)
        self.apical_evidence = np.zeros(n_inputs)

        self.bias = np.random.randn() * 0.1

    def generate_plateau_potential(self, apical_input, context_strength):
        """Generate plateau potential based on apical input and context"""
        # Plateau threshold depends on context
        plateau_threshold = 0.5 - 0.2 * context_strength

        if apical_input > plateau_threshold and not self.plateau_active:
            self.plateau_active = True
            self.plateau_duration = 25  # Time steps
            self.plateau_strength = min(1.0, apical_input)
            return True

        return False

    def forward(self, inputs, context_signal=None):
        """Forward pass with plateau potential dynamics"""
        # Basal dendritic integration (proximal to soma)
        basal_input = np.dot(self.basal_weights, inputs)

        # Apical dendritic integration (distal)
        apical_input = np.dot(self.apical_weights, inputs)

        # Context integration
        if context_signal is not None:
            context_input = np.dot(self.context_weights, context_signal)
            context_strength = sigmoid(context_input)
        else:
            context_strength = 0.5

        # Check for plateau potential generation
        self.generate_plateau_potential(apical_input, context_strength)

        # Plateau potential contribution
        plateau_contribution = 0.0
        if self.plateau_active:
            plateau_contribution = self.plateau_strength * (self.plateau_duration / 25)
            self.plateau_duration -= 1

            if self.plateau_duration <= 0:
                self.plateau_active = False
                self.plateau_strength *= 0.8  # Gradual decay

        # Somatic integration
        somatic_input = (
            basal_input + 0.3 * apical_input + plateau_contribution + self.bias
        )

        return sigmoid(somatic_input)

    def test_plateau_causality(self, inputs, target, context_signal=None):
        """Test causality with and without plateau potentials"""
        # Test basal pathway alone
        temp_apical = self.apical_weights.copy()
        self.apical_weights *= 0

        basal_scores = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                baseline_output = self.forward(inputs, context_signal)
                intervened_output = self.forward(modified_inputs, context_signal)

                basal_scores[i] += abs(intervened_output - baseline_output)

        # Restore apical weights and test combined pathway
        self.apical_weights = temp_apical

        combined_scores = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                baseline_output = self.forward(inputs, context_signal)
                intervened_output = self.forward(modified_inputs, context_signal)

                combined_scores[i] += abs(intervened_output - baseline_output)

        # Apical contribution is the difference
        apical_scores = combined_scores - basal_scores

        return basal_scores, apical_scores

    def update_plateau_learning(self, inputs, target, context_signal=None):
        """Update with plateau-dependent learning rules"""
        basal_scores, apical_scores = self.test_plateau_causality(
            inputs, target, context_signal
        )

        # Update evidence
        self.basal_evidence = 0.9 * self.basal_evidence + 0.1 * basal_scores
        self.apical_evidence = 0.9 * self.apical_evidence + 0.1 * apical_scores

        prediction_error = target - self.forward(inputs, context_signal)

        # Standard learning for basal weights
        for i in range(self.n_inputs):
            if self.basal_evidence[i] > 0.05:
                self.basal_weights[i] += self.eta * prediction_error * inputs[i]

        # Plateau-enhanced learning for apical weights
        plateau_multiplier = 1.0
        if self.plateau_active:
            plateau_multiplier = 1.0 + 2.0 * self.plateau_strength

        for i in range(self.n_inputs):
            if self.apical_evidence[i] > 0.03:
                weight_update = (
                    self.eta * plateau_multiplier * prediction_error * inputs[i]
                )
                self.apical_weights[i] += weight_update


class BiDirectionalCausalNeuron:
    """
    Model with separate processing for feedforward and feedback signals,
    inspired by layer 5 pyramidal neurons with distinct apical and basal processing.
    """

    def __init__(self, n_inputs, eta=0.1):
        self.n_inputs = n_inputs
        self.eta = eta

        # Feedforward weights (bottom-up signals)
        self.feedforward_weights = np.random.randn(n_inputs) * 0.2

        # Feedback weights (top-down signals)
        self.feedback_weights = np.random.randn(n_inputs) * 0.2

        # Cross-directional modulation
        self.ff_to_fb_modulation = np.random.randn(n_inputs) * 0.1
        self.fb_to_ff_modulation = np.random.randn(n_inputs) * 0.1

        # Directional gating
        self.ff_gate = 1.0
        self.fb_gate = 1.0

        # Causal evidence per direction
        self.ff_causal_evidence = np.zeros(n_inputs)
        self.fb_causal_evidence = np.zeros(n_inputs)
        self.interaction_evidence = np.zeros(n_inputs)

        self.bias = np.random.randn() * 0.1

    def forward(self, inputs, feedback_inputs=None):
        """Bidirectional processing"""
        if feedback_inputs is None:
            feedback_inputs = inputs  # Self-feedback

        # Feedforward processing
        ff_raw = np.dot(self.feedforward_weights, inputs)

        # Feedback processing
        fb_raw = np.dot(self.feedback_weights, feedback_inputs)

        # Cross-directional modulation
        ff_modulated = ff_raw + np.dot(self.fb_to_ff_modulation, feedback_inputs)
        fb_modulated = fb_raw + np.dot(self.ff_to_fb_modulation, inputs)

        # Gated outputs
        ff_output = self.ff_gate * sigmoid(ff_modulated)
        fb_output = self.fb_gate * sigmoid(fb_modulated)

        # Integration
        integrated = ff_output + fb_output + self.bias

        return sigmoid(integrated)

    def test_bidirectional_causality(self, inputs, target, feedback_inputs=None):
        """Test causality in both directions"""
        if feedback_inputs is None:
            feedback_inputs = inputs

        ff_scores = np.zeros(self.n_inputs)
        fb_scores = np.zeros(self.n_inputs)
        interaction_scores = np.zeros(self.n_inputs)

        # Test feedforward causality (feedback disabled)
        original_fb_gate = self.fb_gate
        self.fb_gate = 0.0

        baseline_ff = self.forward(inputs, feedback_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_inputs[i] = intervention_val

                intervened_ff = self.forward(modified_inputs, feedback_inputs)
                ff_scores[i] += abs(intervened_ff - baseline_ff)

        # Test feedback causality (feedforward disabled)
        self.fb_gate = original_fb_gate
        self.ff_gate = 0.0

        baseline_fb = self.forward(inputs, feedback_inputs)

        for i in range(self.n_inputs):
            for intervention_val in [0, 1]:
                if feedback_inputs[i] == intervention_val:
                    continue

                modified_feedback = feedback_inputs.copy()
                modified_feedback[i] = intervention_val

                intervened_fb = self.forward(inputs, modified_feedback)
                fb_scores[i] += abs(intervened_fb - baseline_fb)

        # Test interactions (both enabled)
        self.ff_gate = 1.0

        baseline_both = self.forward(inputs, feedback_inputs)

        for i in range(self.n_inputs):
            # Test interaction by modifying both simultaneously
            for intervention_val in [0, 1]:
                if inputs[i] == intervention_val:
                    continue

                modified_inputs = inputs.copy()
                modified_feedback = feedback_inputs.copy()
                modified_inputs[i] = intervention_val
                modified_feedback[i] = intervention_val

                intervened_both = self.forward(modified_inputs, modified_feedback)
                interaction_effect = abs(intervened_both - baseline_both)

                # Interaction is effect beyond sum of individual effects
                expected_effect = ff_scores[i] + fb_scores[i]
                interaction_scores[i] += max(0, interaction_effect - expected_effect)

        return ff_scores, fb_scores, interaction_scores

    def update_bidirectional_learning(self, inputs, target, feedback_inputs=None):
        """Update based on bidirectional causal analysis"""
        if feedback_inputs is None:
            feedback_inputs = inputs

        ff_scores, fb_scores, interaction_scores = self.test_bidirectional_causality(
            inputs, target, feedback_inputs
        )

        # Update evidence
        self.ff_causal_evidence = 0.9 * self.ff_causal_evidence + 0.1 * ff_scores
        self.fb_causal_evidence = 0.9 * self.fb_causal_evidence + 0.1 * fb_scores
        self.interaction_evidence = (
            0.9 * self.interaction_evidence + 0.1 * interaction_scores
        )

        prediction_error = target - self.forward(inputs, feedback_inputs)

        # Update feedforward weights
        for i in range(self.n_inputs):
            if self.ff_causal_evidence[i] > 0.05:
                self.feedforward_weights[i] += self.eta * prediction_error * inputs[i]

        # Update feedback weights
        for i in range(self.n_inputs):
            if self.fb_causal_evidence[i] > 0.05:
                self.feedback_weights[i] += (
                    self.eta * prediction_error * feedback_inputs[i]
                )

        # Update modulation weights based on interactions
        for i in range(self.n_inputs):
            if self.interaction_evidence[i] > 0.03:
                # Strengthen cross-directional modulation
                self.ff_to_fb_modulation[i] += (
                    0.5 * self.eta * prediction_error * inputs[i]
                )
                self.fb_to_ff_modulation[i] += (
                    0.5 * self.eta * prediction_error * feedback_inputs[i]
                )

        # Adaptive gating based on directional effectiveness
        ff_effectiveness = np.mean(self.ff_causal_evidence)
        fb_effectiveness = np.mean(self.fb_causal_evidence)

        if ff_effectiveness > fb_effectiveness * 1.2:
            self.ff_gate = min(1.0, self.ff_gate + 0.01)
            self.fb_gate = max(0.2, self.fb_gate - 0.005)
        elif fb_effectiveness > ff_effectiveness * 1.2:
            self.fb_gate = min(1.0, self.fb_gate + 0.01)
            self.ff_gate = max(0.2, self.ff_gate - 0.005)


def generate_complex_cancer_data(n_samples=1000):
    """Generate complex cancer dataset for testing advanced models"""
    data = []

    for _ in range(n_samples):
        # Primary causal factors
        tumor_marker = np.random.random()
        genetic_risk = np.random.random()

        # Non-linear causal relationship with threshold effects
        if tumor_marker > 0.7 and genetic_risk > 0.7:
            # High risk interaction
            cancer_prob = 0.85 + 0.1 * np.random.random()
        elif tumor_marker > 0.6 or genetic_risk > 0.8:
            # Medium risk
            cancer_prob = 0.4 + 0.3 * (tumor_marker + genetic_risk) / 2
        else:
            # Low risk with weak causal signal
            cancer_prob = 0.1 + 0.2 * tumor_marker + 0.15 * genetic_risk

        cancer_prob = np.clip(cancer_prob, 0.05, 0.95)
        has_cancer = 1 if np.random.random() < cancer_prob else 0

        # Strong confounding factors (designed to be misleading)
        age_corr = 0.4 + 0.5 * has_cancer + np.random.normal(0, 0.1)
        lifestyle_corr = 0.3 + 0.6 * has_cancer + np.random.normal(0, 0.1)

        # Convert to binary inputs
        inputs = np.array(
            [
                1 if tumor_marker > 0.5 else 0,
                1 if genetic_risk > 0.5 else 0,
                1 if age_corr > 0.6 else 0,
                1 if lifestyle_corr > 0.6 else 0,
                1 if np.random.random() > 0.5 else 0,  # Random noise
            ]
        )

        # Feedback/context signals (environmental factors)
        stress_level = np.random.random()
        screening_history = np.random.random()

        feedback = np.array(
            [
                1 if stress_level > 0.5 else 0,
                1 if screening_history > 0.5 else 0,
                1 if (tumor_marker + genetic_risk) > 1.0 else 0,  # Compound risk
                1 if age_corr > 0.7 else 0,  # Age feedback
                0,  # Placeholder
            ]
        )

        data.append((inputs, has_cancer, feedback))

    return data


def test_experimental_models():
    """Test all experimental dendritic models"""
    print("=== Testing Experimental Dendritic Models ===\n")

    # Generate complex test data
    train_data = generate_complex_cancer_data(2000)
    test_data = generate_complex_cancer_data(500)

    models = {
        "Dendritic Spiking": DendriticSpikingNeuron(5, n_branches=4),
        "Plateau Potential": PlateauPotentialNeuron(5),
        "Bidirectional": BiDirectionalCausalNeuron(5),
    }

    feature_names = ["Tumor", "Genetic", "Age", "Lifestyle", "Noise"]
    results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} Model...")

        # Training
        for epoch in range(25):
            np.random.shuffle(train_data)

            for inputs, target, feedback in train_data:
                if model_name == "Dendritic Spiking":
                    model.update_dendritic_learning(inputs, target)
                elif model_name == "Plateau Potential":
                    # Use first two elements of feedback as context
                    context = feedback[:2]
                    model.update_plateau_learning(inputs, target, context)
                else:  # Bidirectional
                    model.update_bidirectional_learning(inputs, target, feedback)

        # Testing
        correct = 0
        for inputs, target, feedback in test_data:
            if model_name == "Dendritic Spiking":
                prediction = model.forward(inputs)
            elif model_name == "Plateau Potential":
                context = feedback[:2]
                prediction = model.forward(inputs, context)
            else:  # Bidirectional
                prediction = model.forward(inputs, feedback)

            if (prediction > 0.5) == target:
                correct += 1

        accuracy = correct / len(test_data)
        results[model_name] = {"accuracy": accuracy}

        print(f"\n{model_name} Model Results:")
        print(f"Accuracy: {accuracy:.3f}")

        # Model-specific analysis
        if model_name == "Dendritic Spiking":
            print("Branch Causal Strengths:")
            for branch_id in range(model.n_branches):
                print(f"  Branch {branch_id}: {model.causal_strength[branch_id]}")
            print(f"Branch Importance: {model.branch_importance}")
            print(f"Calcium Levels: {model.calcium_levels}")

        elif model_name == "Plateau Potential":
            print("Pathway Evidence:")
            print(f"  Basal: {model.basal_evidence}")
            print(f"  Apical: {model.apical_evidence}")
            print(
                f"Plateau State: Active={model.plateau_active}, Strength={model.plateau_strength:.3f}"
            )

        else:  # Bidirectional
            print("Directional Evidence:")
            print(f"  Feedforward: {model.ff_causal_evidence}")
            print(f"  Feedback: {model.fb_causal_evidence}")
            print(f"  Interactions: {model.interaction_evidence}")
            print(f"Gate States: FF={model.ff_gate:.3f}, FB={model.fb_gate:.3f}")

        print()

    print("=== Model Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.3f}")

    return results, models


if __name__ == "__main__":
    results, models = test_experimental_models()
