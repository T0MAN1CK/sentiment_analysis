import os
import argparse
import pickle
import logging
import json
import pandas as pd
import sys
from tensorflow.keras.models import Sequential, load_model

# Ensure logging directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"),  # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Define paths for test datasets
VECTOR_DATA_PATHS = {
    "TF-IDF": "data/vectorized/test_tfidf.csv",
    "CountVectorizer": "data/vectorized/test_count.csv",
    "Word2Vec": "data/vectorized/test_word2vec.csv"
}

ORIGINAL_TEST_DATA = "data/processed/test_cleaned.csv"  # Original test data with text and sentiment

# Define model paths
MODEL_PATHS = {
    "LogisticRegression_TF-IDF": "outputs/models/logisticregression_tf-idf.pkl",
    "LogisticRegression_CountVectorizer": "outputs/models/logisticregression_countvectorizer.pkl",
    "RandomForestClassifier_TF-IDF": "outputs/models/randomforestclassifier_tf-idf.pkl",
    "RandomForestClassifier_CountVectorizer": "outputs/models/randomforestclassifier_countvectorizer.pkl",
    "MLP_Word2Vec": "outputs/models/mlp_word2vec.h5"
}

def load_model_file(model_name):
    """Loads a trained model from disk."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}")

    logging.info(f"Loading model: {model_name}")
    
    # Load Scikit-Learn models
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    # Load TensorFlow Keras model
    elif model_path.endswith(".h5"):
        return load_model(model_path)

def load_vectorized_test_data(model_name):
    """Loads pre-vectorized test data for the selected model."""
    if "TF-IDF" in model_name:
        test_data_path = VECTOR_DATA_PATHS["TF-IDF"]
    elif "CountVectorizer" in model_name:
        test_data_path = VECTOR_DATA_PATHS["CountVectorizer"]
    elif "Word2Vec" in model_name:
        test_data_path = VECTOR_DATA_PATHS["Word2Vec"]
    else:
        raise ValueError(f"Unknown vectorization type for model '{model_name}'.")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    logging.info(f"Loading test data from {test_data_path}")
    return pd.read_csv(test_data_path).values  # Convert to NumPy array for inference

def run_inference(model_name):
    """Runs inference using a trained model and saves predictions."""
    logging.info(f"Starting inference with model: {model_name}")

    # Load trained model
    model = load_model_file(model_name)

    # Load vectorized test data
    X_test = load_vectorized_test_data(model_name)

    # Load original test dataset (for text and actual sentiment)
    if not os.path.exists(ORIGINAL_TEST_DATA):
        raise FileNotFoundError(f"Original test dataset not found: {ORIGINAL_TEST_DATA}")
    
    test_df = pd.read_csv(ORIGINAL_TEST_DATA)
    
    # Ensure correct columns exist
    if "cleaned_review" not in test_df.columns or "sentiment" not in test_df.columns:
        raise ValueError("Original test dataset must contain 'cleaned_review' and 'sentiment' columns.")

    # Make predictions
    logging.info("Running predictions...")
    if isinstance(model, Sequential):  # MLP Model
        predictions = model.predict(X_test)
        predictions = (predictions > 0.5).astype(int).flatten()  # Convert to binary labels
    else:  # Scikit-Learn models (Logistic Regression, Random Forest)
        predictions = model.predict(X_test)

    # Convert numerical predictions to labels (0 -> negative, 1 -> positive)
    predictions_labels = ["negative" if pred == 0 else "positive" for pred in predictions]

    # Prepare output DataFrame
    output_df = test_df[["cleaned_review", "sentiment"]].copy()  # Keep original text and sentiment
    output_df.rename(columns={"cleaned_review": "review", "sentiment": "actual_sentiment"}, inplace=True)
    output_df["predicted_sentiment"] = predictions_labels  # Add predictions

    # Save predictions
    output_path = f"outputs/predictions/predictions_{model_name}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logging.info(f"Inference completed. Predictions saved to: {output_path}")
    print(f"Inference completed! Predictions saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained models.")
    parser.add_argument("--model", type=str, required=False, 
                        help="Specify the trained model to use for inference. If not provided, the best model is selected automatically.")
    args = parser.parse_args()

    if args.model:
        selected_model = args.model
    else:
        # Automatically select best model from metrics.json
        metrics_path = "outputs/predictions/metrics.json"
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Cannot select best model automatically.")
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding metrics.json: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading metrics.json: {e}")
            raise

        best_model = None
        best_accuracy = -1
        for model_name, metrics in metrics_data.items():
            # Skip models not present in MODEL_PATHS
            if model_name not in MODEL_PATHS:
                logging.warning(f"Model '{model_name}' from metrics.json not found in available models. Skipping.")
                continue
            # Get accuracy, default to -1 if not found
            current_accuracy = metrics.get("accuracy", -1)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_model = model_name

        if not best_model:
            raise ValueError("No valid model found in metrics.json with accuracy metrics.")
        
        selected_model = best_model
        logging.info(f"Automatically selected best model: {selected_model} (Accuracy: {best_accuracy})")
        print(f"Automatically selected best model: {selected_model} (Accuracy: {best_accuracy})")

    # Run inference with selected model
    run_inference(selected_model)


# ##Run Models like this
# ##python src/inference/run_inference.py --model <MODEL_NAME>

# # LR
# python src/inference/run_inference.py --model LogisticRegression_TF-IDF
# python src/inference/run_inference.py --model LogisticRegression_CountVectorizer

# # RF
# python src/inference/run_inference.py --model RandomForestClassifier_TF-IDF
# python src/inference/run_inference.py --model RandomForestClassifier_CountVectorizer

# # MLP
# python src/inference/run_inference.py --model MLP_Word2Vec
