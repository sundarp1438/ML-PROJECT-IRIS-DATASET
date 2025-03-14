import pandas as pd
import yaml
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(config_path):
    """Trains a model and saves it."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_path = config.get('data_split', {}).get('trainset_path', 'data/processed/train_iris.csv')
    model_path = config.get('train', {}).get('model_path', 'models/model.joblib')
    target_column = config.get('featurize', {}).get('target_column', 'species')

    # Load train dataset
    df = pd.read_csv(train_path)
    
    # Split features & target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    train_model(args.config)
