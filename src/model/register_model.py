# model_registration.py

import json
import mlflow
import logging

MLFLOW_TRACKING_URI = "http://ec2-13-218-87-51.compute-1.amazonaws.com:5000/"

# ── Logging configuration ──────────────────────────────────────────────────
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model_info(file_path: str) -> dict:
    """Load model run_id and model_path from the JSON written by model_evaluation."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error loading model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """
    Register the model in the MLflow Model Registry and promote it to Staging.

    FIX #2 – The registered version is transitioned to 'Staging' here, and
    app.py now loads by stage (models:/<name>/Staging) instead of a hardcoded
    version number, so deployments stay in sync automatically.
    """
    try:
        model_uri     = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()

        # Transition the newly registered version to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True   # demote older Staging versions automatically
        )

        logger.debug(
            'Model %s version %s registered and transitioned to Staging.',
            model_name, model_version.version
        )
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info = load_model_info('experiment_info.json')
        register_model("my_model", model_info)
    except Exception as e:
        logger.error('Failed to complete model registration: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()