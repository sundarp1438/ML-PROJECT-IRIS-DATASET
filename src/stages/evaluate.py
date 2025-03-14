import pandas as pd
import yaml
import argparse
import joblib
import json
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(config_path):
    """Evaluates the trained model and saves metrics."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    test_path = config.get('data_split', {}).get('testset_path', 'data/processed/test_iris.csv')
    model_path = config.get('train', {}).get('model_path', 'models/model.joblib')
    metrics_path = config.get('evaluate', {}).get('metrics_path', 'reports/metrics.json')
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
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save metrics
    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    print(f"✅ Evaluation complete. Metrics saved at {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    evaluate_model(args.config)
