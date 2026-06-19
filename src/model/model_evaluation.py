# src/model/model_evaluation.py

import traceback

import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


# ──  project root early so all paths (including log file) are anchored ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

# ── Logging configuration ──────────────────────────────────────────────────
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Anchor log file to ROOT_DIR instead of bare filename (which uses CWD)
file_handler = logging.FileHandler(os.path.join(ROOT_DIR, 'model_evaluation_errors.log'))
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

MLFLOW_TRACKING_URI = "http://ec2-13-218-87-51.compute-1.amazonaws.com:5000/"

# Artifact store on the EC2 server — change this to any path the ec2-user
# (or whatever user runs the MLflow server) has write access to.
# Common options: /tmp/mlruns  OR  s3://your-bucket/mlruns
MLFLOW_ARTIFACT_ROOT = "/tmp/mlruns"


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Return the experiment id, creating the experiment with an explicit
    artifact_location if it does not already exist.
    This avoids the MLflow server defaulting to /home/ubuntu which may
    not be writable by the runner.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.debug(
            "Experiment '%s' not found — creating with artifact root %s",
            experiment_name, MLFLOW_ARTIFACT_ROOT,
        )
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACT_ROOT,
        )
    else:
        experiment_id = experiment.experiment_id
        logger.debug(
            "Using existing experiment '%s' (id=%s, artifact_location=%s)",
            experiment_name, experiment_id, experiment.artifact_location,
        )
    return experiment_id


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model and return classification report + confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception:
        logger.exception("Failed to complete model evaluation")
        traceback.print_exc()
        raise


def log_confusion_matrix(cm, dataset_name: str, output_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_file_path = os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()
    logger.debug('Confusion matrix saved to %s', cm_file_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Use get_or_create_experiment so we control the artifact_location
    experiment_id = get_or_create_experiment('dvc-pipeline-runs')

    with mlflow.start_run(experiment_id=experiment_id) as run:
        try:
            root_dir = ROOT_DIR

            params = load_params(os.path.join(root_dir, 'params.yaml'))
            for key, value in params.items():
                mlflow.log_param(key, value)

            model      = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            X_test_tfidf  = vectorizer.transform(test_data['clean_comment'].values)
            y_test        = test_data['category'].values
            feature_names = vectorizer.get_feature_names_out()

            input_example = pd.DataFrame(
                X_test_tfidf.toarray()[:5],
                columns=feature_names
            )
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example.head(5)
            )

            model_path = "lgbm_model"

            # Write experiment_info.json to root_dir, not bare CWD
            save_model_info(
                run.info.run_id,
                model_path,
                os.path.join(root_dir, 'experiment_info.json')
            )

            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate on full test set
            X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)
            report, cm = evaluate_model(model, X_test_df, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall":    metrics['recall'],
                        f"test_{label}_f1-score":  metrics['f1-score']
                    })

            # Pass root_dir so PNG is written to the project root, not CWD
            log_confusion_matrix(cm, "Test Data", output_dir=root_dir)

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task",       "Sentiment Analysis")
            mlflow.set_tag("dataset",    "YouTube Comments")

        except Exception as e:
            logger.error('Failed to complete model evaluation: %s', e)
            traceback.print_exc()
            print(f"Error: {e}")


if __name__ == '__main__':
    main()