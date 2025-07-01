#!/usr/bin/env python3
"""
Individual Model Testing Script
==============================

Demonstrates how to use individual models from the organized structure.
This is useful for testing specific models or integrating them into other projects.
"""

import numpy as np
from utils import generate_cancer_data, evaluate_model, print_model_summary

# Example: Test the breakthrough Contrastive Learning model
from models.contrastive_learning import ContrastiveCausalNeuron

# Example: Test the baseline for comparison
from models.baseline import BaselineNeuron


def test_individual_model():
    """Test a single model to demonstrate the interface"""

    print("=" * 60)
    print("INDIVIDUAL MODEL TESTING EXAMPLE")
    print("=" * 60)

    # Generate data
    print("Generating test data...")
    train_data = generate_cancer_data(n_samples=1000)
    test_data = generate_cancer_data(n_samples=200)

    # Create model
    print("Creating Contrastive Learning model...")
    model = ContrastiveCausalNeuron(n_inputs=5, eta=0.1)

    # Train model
    print("Training model...")
    epochs = 20
    for epoch in range(epochs):
        # Shuffle data each epoch
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            model.update(inputs, target)

        # Print progress every 5 epochs
        if epoch % 5 == 0:
            # Quick evaluation
            correct = 0
            for inputs, target in test_data[:50]:  # Small subset
                pred = model.forward(inputs)
                if (pred > 0.5) == target:
                    correct += 1
            accuracy = correct / 50
            print(f"  Epoch {epoch}: Accuracy = {accuracy:.3f}")

    # Final evaluation
    print("\nFinal evaluation:")
    result = evaluate_model(model, test_data, "Contrastive Learning")
    print_model_summary(result)

    # Demonstrate prediction on new data
    print("\nDemonstrating predictions:")
    test_cases = [
        ([1, 1, 0, 0, 0], "High risk: Tumor + Genetic markers present"),
        ([1, 0, 1, 1, 0], "Medium risk: Tumor present, age/screening factors"),
        ([0, 0, 1, 1, 0], "Low risk: Only correlational factors"),
        ([0, 0, 0, 0, 1], "Noise only: Should predict low"),
    ]

    for inputs, description in test_cases:
        inputs = np.array(inputs, dtype=float)
        prediction = model.forward(inputs)
        print(f"  {description}: {prediction:.3f}")

    # Demonstrate causal evidence (if available)
    if hasattr(model, "causal_evidence") and model.causal_evidence is not None:
        print("\nCausal evidence learned:")
        feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
        for i, (name, evidence) in enumerate(zip(feature_names, model.causal_evidence)):
            print(f"  {name}: {evidence:.4f}")


def compare_two_models():
    """Compare baseline vs causal model"""

    print("\n" + "=" * 60)
    print("COMPARING BASELINE VS CAUSAL MODEL")
    print("=" * 60)

    # Generate data
    train_data = generate_cancer_data(n_samples=1000)
    test_data = generate_cancer_data(n_samples=200)

    # Create models
    baseline = BaselineNeuron(n_inputs=5, eta=0.3)
    causal = ContrastiveCausalNeuron(n_inputs=5, eta=0.1)

    models = {"Baseline": baseline, "Contrastive": causal}

    # Train both models
    print("Training both models...")
    for epoch in range(15):
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            baseline.update(inputs, target)
            causal.update(inputs, target)

    # Evaluate both
    print("\nComparison results:")
    for name, model in models.items():
        result = evaluate_model(model, test_data, name)
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Causal Score: {result['causal_score']:.3f}")

        # Feature importance
        if result["feature_importance"]:
            feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
            print("  Feature Importance:")
            for feat, imp in zip(feature_names, result["feature_importance"]):
                print(f"    {feat}: {imp:.3f}")


def demonstrate_custom_usage():
    """Show how to use models in custom scenarios"""

    print("\n" + "=" * 60)
    print("CUSTOM USAGE DEMONSTRATION")
    print("=" * 60)

    # Create model
    model = ContrastiveCausalNeuron(n_inputs=5, eta=0.1)

    # Custom training data (manual)
    print("Training on custom manual data...")
    custom_training = [
        # (features, target) - manually created examples
        ([1, 1, 0, 0, 0], 1),  # Strong causal factors -> cancer
        ([1, 0, 1, 1, 0], 1),  # Tumor + correlations -> cancer
        ([0, 1, 0, 1, 0], 1),  # Genetic + screening -> cancer
        ([0, 0, 1, 1, 0], 0),  # Only correlations -> no cancer
        ([0, 0, 0, 0, 1], 0),  # Only noise -> no cancer
        ([0, 0, 0, 0, 0], 0),  # Nothing -> no cancer
    ]

    # Train on custom data multiple times
    for _ in range(100):
        for inputs, target in custom_training:
            inputs = np.array(inputs, dtype=float)
            model.update(inputs, target)

    # Test on new cases
    print("\nTesting on new cases:")
    test_cases = [
        ([1, 0, 0, 0, 0], "Only tumor marker"),
        ([0, 1, 0, 0, 0], "Only genetic risk"),
        ([0, 0, 1, 0, 0], "Only age factor"),
        ([1, 1, 1, 1, 1], "All factors present"),
    ]

    for inputs, description in test_cases:
        inputs = np.array(inputs, dtype=float)
        prediction = model.forward(inputs)
        print(f"  {description}: {prediction:.3f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run demonstrations
    test_individual_model()
    compare_two_models()
    demonstrate_custom_usage()

    print(f"\n{'=' * 60}")
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("This shows how to:")
    print("✅ Use individual models")
    print("✅ Train on your own data")
    print("✅ Compare different approaches")
    print("✅ Make predictions on new cases")
    print("✅ Access learned causal evidence")
    print("\nSee README_MODELS.md for more details!")
