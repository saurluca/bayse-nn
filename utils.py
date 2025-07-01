import numpy as np


def sigmoid(x):
    """Safe sigmoid function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


def generate_cancer_data(n_samples=1000):
    """
    Generate synthetic cancer detection data with known causal structure:
    - Tumor marker (causal): directly indicates cancer presence
    - Genetic risk (causal): increases cancer probability
    - Age (correlational): tends to correlate with cancer but isn't causal in this scenario
    - Screening (correlational): people with cancer more likely to be screened
    - Noise: random factor, should have no effect
    """
    data = []

    for _ in range(n_samples):
        # TRUE CAUSAL FACTORS
        tumor_marker = 1 if np.random.random() > 0.6 else 0
        genetic_risk = 1 if np.random.random() > 0.7 else 0

        # Cancer probability based on causal factors
        cancer_prob = 0.1  # Base rate
        if tumor_marker:
            cancer_prob += 0.6  # Strong causal effect
        if genetic_risk:
            cancer_prob += 0.4  # Moderate causal effect
        if tumor_marker and genetic_risk:
            cancer_prob += 0.2  # Interaction effect

        cancer_prob = min(cancer_prob, 0.95)  # Cap at 95%
        has_cancer = 1 if np.random.random() < cancer_prob else 0

        # CORRELATIONAL FACTORS (not causal)
        # Age: older people more likely to have cancer (correlation, not causation in this model)
        age_prob = 0.3 + 0.5 * has_cancer + np.random.normal(0, 0.1)
        age = 1 if np.random.random() < np.clip(age_prob, 0, 1) else 0

        # Screening: people with cancer more likely to be screened
        screening_prob = 0.2 + 0.6 * has_cancer + np.random.normal(0, 0.1)
        screening = 1 if np.random.random() < np.clip(screening_prob, 0, 1) else 0

        # Pure noise
        noise = 1 if np.random.random() > 0.5 else 0

        inputs = np.array(
            [tumor_marker, genetic_risk, age, screening, noise], dtype=float
        )
        data.append((inputs, has_cancer))

    return data


def generate_controlled_causal_data(n_samples=1000, correlation_strength=0.6):
    """
    Generate data with controlled causal vs correlational relationships
    """
    data = []

    for _ in range(n_samples):
        # TRUE CAUSAL FACTORS
        tumor_marker = np.random.randint(0, 2)
        genetic_risk = np.random.randint(0, 2)

        # Pure causal relationship: Cancer = 1 if (tumor=1) OR (genetic=1)
        has_cancer = max(tumor_marker, genetic_risk)

        # CORRELATIONAL FACTORS
        # Age: correlation_strength probability of matching cancer status
        age = (
            has_cancer if np.random.random() < correlation_strength else 1 - has_cancer
        )

        # Screening: correlation_strength probability of matching cancer status
        screening = (
            has_cancer if np.random.random() < correlation_strength else 1 - has_cancer
        )

        # Pure noise
        noise = np.random.randint(0, 2)

        inputs = np.array([tumor_marker, genetic_risk, age, screening, noise])
        data.append((inputs, has_cancer))

    return data


def evaluate_model(model, test_data, model_name):
    """
    Comprehensive evaluation of a model
    Returns accuracy, loss, and feature importance analysis
    """
    correct = 0
    total_loss = 0
    predictions = []
    targets = []

    for inputs, target in test_data:
        # Get prediction
        if hasattr(model, "forward"):
            pred = model.forward(inputs)
        elif hasattr(model, "predict"):
            pred = model.predict(inputs)
        else:
            raise AttributeError(f"Model {model_name} has no forward or predict method")

        predictions.append(pred)
        targets.append(target)

        # Accuracy
        pred_class = 1 if pred > 0.5 else 0
        if pred_class == target:
            correct += 1

        # Loss (MSE)
        total_loss += (target - pred) ** 2

    accuracy = correct / len(test_data)
    avg_loss = total_loss / len(test_data)

    # Feature importance through intervention
    feature_importance = calculate_feature_importance(model, test_data)

    # Causal discovery score
    causal_score = calculate_causal_discovery_score(feature_importance)

    return {
        "name": model_name,
        "accuracy": accuracy,
        "loss": avg_loss,
        "feature_importance": feature_importance,
        "causal_score": causal_score,
        "weights": getattr(model, "weights", None),
        "causal_evidence": getattr(model, "causal_evidence", None),
    }


def calculate_feature_importance(model, test_data, test_inputs=None):
    """Calculate feature importance through intervention testing"""
    if test_inputs is None:
        test_inputs = np.array([1, 1, 1, 1, 0])  # High-risk patient

    # Get original prediction
    if hasattr(model, "forward"):
        original_pred = model.forward(test_inputs)
    else:
        original_pred = model.predict(test_inputs)

    feature_importance = []
    n_features = len(test_inputs)

    for i in range(n_features):
        # Test intervention by setting feature to 0
        modified = test_inputs.copy()
        modified[i] = 0

        if hasattr(model, "forward"):
            new_pred = model.forward(modified)
        else:
            new_pred = model.predict(modified)

        importance = original_pred - new_pred
        feature_importance.append(importance)

    return feature_importance


def calculate_causal_discovery_score(feature_importance):
    """
    Calculate how well the model discovered causal structure
    Expected: Tumor (0) and Genetic (1) should have high importance
    Age (2), Screening (3), Noise (4) should have low importance
    """
    if len(feature_importance) < 5:
        return 0.0

    # Score based on expected causal structure
    causal_importance = feature_importance[0] + feature_importance[1]  # Tumor + Genetic
    non_causal_importance = (
        feature_importance[2] + feature_importance[3] + abs(feature_importance[4])
    )  # Age + Screening + |Noise|

    causal_score = causal_importance - non_causal_importance
    return causal_score


def test_intervention_robustness(model, test_data, n_tests=100):
    """
    Test how robust the model is to interventions
    Good causal models should be sensitive to causal interventions
    but robust to correlational interventions
    """
    causal_sensitivity = 0  # Should be high
    confound_robustness = 0  # Should be high

    test_count = min(n_tests, len(test_data))

    for i in range(test_count):
        inputs, target = test_data[i]

        if hasattr(model, "forward"):
            original_pred = model.forward(inputs)
        else:
            original_pred = model.predict(inputs)

        # Test causal intervention (remove tumor marker)
        if inputs[0] == 1:  # Tumor marker was present
            no_tumor = inputs.copy()
            no_tumor[0] = 0

            if hasattr(model, "forward"):
                no_tumor_pred = model.forward(no_tumor)
            else:
                no_tumor_pred = model.predict(no_tumor)

            # Good model should show significant change
            if abs(original_pred - no_tumor_pred) > 0.1:
                causal_sensitivity += 1

        # Test confound intervention (remove age factor)
        if inputs[2] == 1:  # Age factor was present
            no_age = inputs.copy()
            no_age[2] = 0

            if hasattr(model, "forward"):
                no_age_pred = model.forward(no_age)
            else:
                no_age_pred = model.predict(no_age)

            # Good model should be relatively robust to confound removal
            if abs(original_pred - no_age_pred) < 0.2:
                confound_robustness += 1

    return {
        "causal_sensitivity": causal_sensitivity / test_count,
        "confound_robustness": confound_robustness / test_count,
    }


def print_model_summary(result, robustness=None):
    """Print a formatted summary of model performance"""
    print(f"\n{result['name']}:")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Loss: {result['loss']:.3f}")
    print(f"  Causal Discovery Score: {result['causal_score']:.3f}")

    if result["feature_importance"]:
        feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
        print("  Feature Importance:")
        for i, (name, importance) in enumerate(
            zip(feature_names, result["feature_importance"])
        ):
            print(f"    {name}: {importance:.3f}")

    if robustness:
        print(f"  Causal Sensitivity: {robustness['causal_sensitivity']:.3f}")
        print(f"  Confound Robustness: {robustness['confound_robustness']:.3f}")

    if result["weights"] is not None:
        print(f"  Weights: {result['weights']}")

    if result["causal_evidence"] is not None:
        print(f"  Causal Evidence: {result['causal_evidence']}")
