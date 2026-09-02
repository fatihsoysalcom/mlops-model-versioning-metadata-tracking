import os
import json
import pickle
import uuid
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a directory to store models and metadata
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_log_model(run_name, hyperparameters):
    """
    Simulates training an ML model and logs its metadata and artifact.
    This demonstrates a core MLOps concept: tracking different runs.
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    print(f"\n--- Starting ML Run: {run_name} (ID: {run_id}) ---")

    # 1. Simulate Data Generation
    # In a real scenario, this would involve loading and preprocessing actual data.
    X = np.random.rand(100, 5) * 10 # 100 samples, 5 features
    y = (X[:, 0] + X[:, 1] > 10).astype(int) # Simple binary classification target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data simulated and split.")

    # 2. Model Training
    # Using scikit-learn's Logistic Regression as a simple example.
    model = LogisticRegression(**hyperparameters, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained with hyperparameters: {hyperparameters}")

    # 3. Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    # 4. Log Metadata and Save Model Artifact (Core MLOps step)
    # This is crucial for reproducibility, debugging, and deployment.
    model_filename = f"model_{run_id}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "model_type": "LogisticRegression",
        "hyperparameters": hyperparameters,
        "metrics": {
            "accuracy": accuracy
        },
        "model_artifact_path": model_path,
        "data_info": { # Simplified data info
            "num_samples": len(X),
            "num_features": X.shape[1],
            "target_distribution": {
                "class_0": np.sum(y == 0),
                "class_1": np.sum(y == 1)
            }
        }
    }

    metadata_filename = f"metadata_{run_id}.json"
    metadata_path = os.path.join(MODELS_DIR, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata logged to: {metadata_path}")

    return metadata

def review_logged_runs():
    """
    Reviews all logged model runs in the MODELS_DIR.
    """
    print("\n--- Reviewing All Logged ML Runs ---")
    run_summaries = []
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith("metadata_") and filename.endswith(".json"):
            filepath = os.path.join(MODELS_DIR, filename)
            with open(filepath, 'r') as f:
                metadata = json.load(f)
                run_summaries.append({
                    "run_id": metadata["run_id"],
                    "run_name": metadata["run_name"],
                    "timestamp": metadata["timestamp"],
                    "accuracy": metadata["metrics"]["accuracy"],
                    "hyperparameters": metadata["hyperparameters"]
                })
    
    if not run_summaries:
        print("No ML runs found.")
        return

    # Sort by timestamp for better readability
    run_summaries.sort(key=lambda x: x["timestamp"], reverse=True)

    for summary in run_summaries:
        print(f"  Run ID: {summary['run_id']}")
        print(f"  Name: {summary['run_name']}")
        print(f"  Timestamp: {summary['timestamp']}")
        print(f"  Accuracy: {summary['accuracy']:.4f}")
        print(f"  Hyperparameters: {summary['hyperparameters']}")
        print("-" * 30)

def load_and_predict(run_id, new_data):
    """
    Loads a specific model by its run_id and makes predictions.
    This simulates deploying a specific version of a model.
    """
    print(f"\n--- Loading Model for Run ID: {run_id} ---")
    model_path = None
    metadata_path = None
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"metadata_{run_id}") and filename.endswith(".json"):
            metadata_path = os.path.join(MODELS_DIR, filename)
            break
    
    if not metadata_path:
        print(f"Metadata for run ID {run_id} not found.")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        model_path = metadata["model_artifact_path"]

    if not os.path.exists(model_path):
        print(f"Model artifact not found at {model_path} for run ID {run_id}.")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model '{metadata['run_name']}' (ID: {run_id}) loaded successfully.")

    predictions = model.predict(new_data)
    print(f"Predictions for new data: {predictions}")
    return predictions


if __name__ == "__main__":
    # Simulate multiple training runs with different hyperparameters
    # This highlights the need for MLOps to track these variations.
    run1_metadata = train_and_log_model(
        "Baseline Model",
        {"C": 1.0, "solver": "liblinear"}
    )

    run2_metadata = train_and_log_model(
        "Tuned Model - Higher Regularization",
        {"C": 0.1, "solver": "liblinear"} # Different C parameter
    )

    run3_metadata = train_and_log_model(
        "Tuned Model - Different Solver",
        {"C": 1.0, "solver": "lbfgs"} # Different solver
    )

    # Review all logged runs
    review_logged_runs()

    # Simulate loading and using a specific model for prediction
    # This shows how MLOps enables deploying a chosen version.
    if run1_metadata:
        sample_new_data = np.random.rand(3, 5) * 10
        print(f"\n--- Simulating deployment of '{run1_metadata['run_name']}' (ID: {run1_metadata['run_id']}) ---")
        load_and_predict(run1_metadata["run_id"], sample_new_data)
