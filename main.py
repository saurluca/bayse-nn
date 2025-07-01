#!/usr/bin/env python3
"""
Individual Model Testing Script
==============================

Demonstrates how to use individual models from the organized structure.
This is useful for testing specific models or integrating them into other projects.

Usage:
    uv run test_individual_model.py --models contrastive baseline
    uv run test_individual_model.py --models all
    uv run test_individual_model.py --list-models
"""

import argparse
import sys
import numpy as np
from utils import generate_cancer_data, evaluate_model, print_model_summary

# Import all available models
from models.contrastive_learning import ContrastiveCausalNeuron
from models.baseline import BaselineNeuron
from models.active_dendrite import ActiveDendriteNeuron
from models.hybrid_causal import HybridCausalNeuron
from models.predictive_coding import PredictiveCausalNeuron
from models.selective_intervention import SelectiveInterventionNeuron

# Model registry - maps model names to their classes and default parameters
MODEL_REGISTRY = {
    "contrastive": {
        "class": ContrastiveCausalNeuron,
        "params": {"n_inputs": 5, "eta": 0.1},
        "description": "Contrastive Learning model (breakthrough approach)",
    },
    "baseline": {
        "class": BaselineNeuron,
        "params": {"n_inputs": 5, "eta": 0.3},
        "description": "Standard correlation-based learning",
    },
    "active_dendrite": {
        "class": ActiveDendriteNeuron,
        "params": {"n_inputs": 5, "n_segments": 8, "segment_size": 3, "eta": 0.1},
        "description": "Active dendrite segments with coincidence detection",
    },
    "hybrid": {
        "class": HybridCausalNeuron,
        "params": {"n_inputs": 5, "eta": 0.2, "causal_weight": 0.3},
        "description": "Hybrid correlation + causal intervention learning",
    },
    "predictive": {
        "class": PredictiveCausalNeuron,
        "params": {"n_inputs": 5, "eta": 0.1},
        "description": "Predictive coding with error-based causal discovery",
    },
    "selective": {
        "class": SelectiveInterventionNeuron,
        "params": {"n_inputs": 5, "eta": 0.25, "intervention_prob": 0.3},
        "description": "Selective input masking during training",
    },
}


def list_available_models():
    """List all available models with descriptions"""
    print("Available Models:")
    print("=" * 50)
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name:<15} - {info['description']}")
    print("\nSpecial options:")
    print(f"  {'all':<15} - Test all available models")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test individual neural network models for causal learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run test_individual_model.py --models contrastive baseline
  uv run test_individual_model.py --models all
  uv run test_individual_model.py --list-models
  uv run test_individual_model.py --models contrastive --epochs 30 --train-samples 2000
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["contrastive"],
        help='Models to test (space-separated). Use "all" to test all models',
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List all available models and exit"
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)",
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=200,
        help="Number of test samples (default: 200)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed training progress"
    )

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run comparison, skip individual model testing",
    )

    return parser.parse_args()


def validate_models(model_names):
    """Validate that all specified models exist"""
    if "all" in model_names:
        return list(MODEL_REGISTRY.keys())

    invalid_models = [name for name in model_names if name not in MODEL_REGISTRY]
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print("Available models:")
        list_available_models()
        sys.exit(1)

    return model_names


def test_individual_model(model_name, args):
    """Test a single model to demonstrate the interface"""

    print("=" * 60)
    print(f"TESTING MODEL: {model_name.upper()}")
    print("=" * 60)
    print(f"Description: {MODEL_REGISTRY[model_name]['description']}")

    # Generate data
    print("Generating test data...")
    train_data = generate_cancer_data(n_samples=args.train_samples)
    test_data = generate_cancer_data(n_samples=args.test_samples)

    # Create model
    model_info = MODEL_REGISTRY[model_name]
    print(f"Creating {model_name} model...")
    model = model_info["class"](**model_info["params"])

    # Train model
    print("Training model...")
    for epoch in range(args.epochs):
        # Shuffle data each epoch
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            model.update(inputs, target)

        # Print progress every 5 epochs or if verbose
        if args.verbose or epoch % 5 == 0:
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
    result = evaluate_model(model, test_data, model_name)
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

    return result


def compare_models(model_names, args):
    """Compare multiple models"""

    print("\n" + "=" * 60)
    print(f"COMPARING MODELS: {', '.join(model_names).upper()}")
    print("=" * 60)

    # Generate data
    train_data = generate_cancer_data(n_samples=args.train_samples)
    test_data = generate_cancer_data(n_samples=args.test_samples)

    # Create models
    models = {}
    for name in model_names:
        model_info = MODEL_REGISTRY[name]
        models[name] = model_info["class"](**model_info["params"])

    # Train all models
    print("Training all models...")
    for epoch in range(args.epochs):
        np.random.shuffle(train_data)

        for inputs, target in train_data:
            for model in models.values():
                model.update(inputs, target)

        if args.verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch} completed")

    # Evaluate all models
    print("\nComparison results:")
    results = {}
    for name, model in models.items():
        result = evaluate_model(model, test_data, name)
        results[name] = result
        print(f"\n{name.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Causal Score: {result['causal_score']:.3f}")

        # Feature importance
        if result["feature_importance"]:
            feature_names = ["Tumor", "Genetic", "Age", "Screening", "Noise"]
            print("  Feature Importance:")
            for feat, imp in zip(feature_names, result["feature_importance"]):
                print(f"    {feat}: {imp:.3f}")

    return results


def demonstrate_custom_usage(model_name, args):
    """Show how to use models in custom scenarios"""

    print("\n" + "=" * 60)
    print(f"CUSTOM USAGE DEMONSTRATION - {model_name.upper()}")
    print("=" * 60)

    # Create model
    model_info = MODEL_REGISTRY[model_name]
    model = model_info["class"](**model_info["params"])

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


def main():
    """Main function"""
    args = parse_arguments()

    # Handle list models request
    if args.list_models:
        list_available_models()
        return

    # Validate and get models to test
    model_names = validate_models(args.models)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    print(f"Testing models: {', '.join(model_names)}")
    print(f"Training epochs: {args.epochs}")
    print(f"Training samples: {args.train_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Random seed: {args.seed}")

    # Test individual models unless comparison-only is requested
    if not args.compare_only:
        for model_name in model_names:
            test_individual_model(model_name, args)
            if len(model_names) == 1:  # Only show custom usage for single model
                demonstrate_custom_usage(model_name, args)

    # Compare models if more than one specified
    if len(model_names) > 1:
        compare_models(model_names, args)

    print(f"\n{'=' * 60}")
    print("TESTING COMPLETE")
    print("=" * 60)
    print("This demonstrates:")
    print("✅ Individual model testing and training")
    print("✅ Model comparison and evaluation")
    print("✅ Custom data scenarios")
    print("✅ Causal evidence analysis")
    print("\nUse --help for more options or --list-models to see available models")


if __name__ == "__main__":
    main()
