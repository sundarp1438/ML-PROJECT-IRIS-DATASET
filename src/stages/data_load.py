import pandas as pd
import yaml
import argparse

def load_data(config_path):
    """Loads dataset from a raw source and saves it as CSV."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_path = config.get('data_load', {}).get('dataset_path', 'data/raw/iris.csv')

    # Load dataset (Modify this if you need to fetch from URL or DB)
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

    # Save raw data
    df.to_csv(data_path, index=False)
    print(f"✅ Data saved at {data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    load_data(args.config)
