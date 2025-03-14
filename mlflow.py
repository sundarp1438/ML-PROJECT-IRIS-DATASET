import logging
import yaml
import mlflow
import mlflow.sklearn
from data_load import load_data
from data_split import split_data
from featurize import featurize_data
from train import train_model
from evaluate import evaluate_model
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        logging.info("📥 Starting MLflow model training pipeline...")

        # Load Data
        data = load_data(config['data']['data_path'])
        logging.info("✅ Data loading completed successfully")

        # Split Data
        train_data, test_data = split_data(data, config['data_split'])
        logging.info("✅ Data split into train and test sets")

        # Feature Engineering
        X_train, y_train, X_test, y_test = featurize_data(train_data, test_data, config['featurize'])
        logging.info("✅ Feature engineering completed")

        # Train Model
        model = train_model(X_train, y_train, config['train'])
        logging.info("✅ Model training completed")

        # Evaluate Model
        accuracy, roc_auc_score, cm_path, report = evaluate_model(model, X_test, y_test, config['evaluate'])
        logging.info("✅ Model evaluation completed successfully")

        # Log model parameters
        mlflow.log_params(config['train']['params'])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc_score", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])

        # Log artifacts (confusion matrix image)
        mlflow.log_artifact(cm_path)

        # Log Model
        mlflow.sklearn.log_model(model, "model")

        # Register Model
        model_name = "insurance_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        logging.info("✅ MLflow tracking and model registration completed")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{classification_report(y_test, model.predict(X_test))}")
        print("=====================================================\n")

if __name__ == "__main__":
    train_with_mlflow()
