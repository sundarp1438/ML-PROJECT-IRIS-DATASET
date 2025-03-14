import pandas as pd
import yaml
import joblib
import json
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Configuration
with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_config = config.get('data_load', {})
featurize_config = config.get('featurize', {})
data_split_config = config.get('data_split', {})
train_config = config.get('train', {})
evaluate_config = config.get('evaluate', {})

# Set up MLflow experiment
mlflow.set_experiment("Iris Classification Pipeline")

with mlflow.start_run():
    # 📥 Load Data
    data_path = data_config.get('dataset_path', 'data/raw/iris.csv')
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    mlflow.log_artifact(data_path)
    print(f"✅ Data saved at {data_path}")

    # 🔬 Feature Engineering
    df['species'] = df['species'].astype('category').cat.codes
    processed_data_path = featurize_config.get('processed_path', 'data/processed/featured_iris.csv')
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    mlflow.log_artifact(processed_data_path)
    print(f"✅ Processed data saved at {processed_data_path}")

    # 📊 Train-Test Split
    test_size = data_split_config.get('test_size', 0.2)
    random_state = config.get('base', {}).get('random_state', 42)
    train_path = data_split_config.get('trainset_path', 'data/processed/train_iris.csv')
    test_path = data_split_config.get('testset_path', 'data/processed/test_iris.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    print(f"✅ Train and test sets saved at {train_path} and {test_path}")

    # 🏋️ Model Training
    target_column = featurize_config.get('target_column', 'species')
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    model_path = train_config.get('model_path', 'models/model.joblib')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, "model")
    print(f"✅ Model saved at {model_path}")

    # 📈 Model Evaluation
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Save metrics as JSON
    metrics = {"accuracy": acc, "confusion_matrix": cm.tolist()}
    metrics_path = evaluate_config.get('metrics_path', 'reports/metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    mlflow.log_artifact(metrics_path)
    print(f"✅ Evaluation complete. Metrics saved at {metrics_path}")

    # 📊 Confusion Matrix Visualization
    cm_path = evaluate_config.get('cm_path', 'reports/confusion_matrix.png')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    print(f"✅ Confusion matrix saved at {cm_path}")

    print("\n🎯 MLflow Tracking Completed. Check MLflow UI for logs.")

