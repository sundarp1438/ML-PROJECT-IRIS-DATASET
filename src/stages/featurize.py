import pandas as pd
import yaml
import argparse

def featurize_data(config_path):
    """Applies feature engineering and saves processed dataset."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    raw_data_path = config.get('data_load', {}).get('dataset_path', 'data/raw/iris.csv')
    processed_data_path = config.get('featurize', {}).get('processed_path', 'data/processed/featured_iris.csv')

    # Load data
    df = pd.read_csv(raw_data_path)

    # Feature engineering (Example: Convert categorical species to numerical)
    df['species'] = df['species'].astype('category').cat.codes

    # Save processed data
    df.to_csv(processed_data_path, index=False)
    print(f"✅ Processed data saved at {processed_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    featurize_data(args.config)
