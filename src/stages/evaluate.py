import pandas as pd
import yaml
import argparse
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(config_path):
    """Evaluates the trained model and saves metrics, including confusion matrix visualization."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    test_path = config.get('data_split', {}).get('testset_path', 'data/processed/test_iris.csv')
    model_path = config.get('train', {}).get('model_path', 'models/model.joblib')
    metrics_path = config.get('evaluate', {}).get('metrics_path', 'reports/metrics.json')
    cm_path = config.get('evaluate', {}).get('cm_path', 'reports/confusion_matrix.png')
    target_column = config.get('featurize', {}).get('target_column', 'species')

    # Load test dataset
    df = pd.read_csv(test_path)

    # Split features & target
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Load trained model
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)

    # Save metrics as JSON
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save confusion matrix as PNG
    plt.savefig(cm_path)
    plt.close()

    print(f"✅ Evaluation complete. Metrics saved at {metrics_path}")
    print(f"✅ Confusion matrix saved at {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    evaluate_model(args.config)
